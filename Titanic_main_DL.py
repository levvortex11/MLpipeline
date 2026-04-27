import os
import sys
import copy
import random
import pickle
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, RobustScaler, FunctionTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, TransformerMixin

# Импорты для продвинутой стратегии pseudo_labeling
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from Titanic_config_DL import get_dl_config
from Titanic_model_DL import TitanicDLDataset, FlexibleTitanicNet


def seed_everything(seed=42):
    """
    Фиксация всех генераторов псевдослучайных чисел для обеспечения
    полной воспроизводимости (одинаковых метрик при повторных запусках).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Заставляет cuDNN использовать детерминированные алгоритмы свертки/операций
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Фиксирует seed для подпроцессов (worker) DataLoader'а.
    Это предотвращает дублирование данных при num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MetricFactory:
    """Утилитный класс для удобного расчета пула метрик классификации."""

    @staticmethod
    def calculate(targets, probs, metrics_list):
        # Flatten (сжатие в 1D массив) защищает от конфликтов размерностей (N, 1) в функциях sklearn
        t_flat = np.array(targets).flatten()
        p_flat = np.array(probs).flatten()
        # Жесткие классы (0 или 1) для подсчета Accuracy, F1 и т.д.
        preds = (p_flat > 0.5).astype(int)

        results = {}
        # Маппинг строковых названий метрик из конфига на функции sklearn
        mapping = {
            'auc': lambda t, p, cl: roc_auc_score(t, p),
            'accuracy': lambda t, p, cl: accuracy_score(t, cl),
            'f1': lambda t, p, cl: f1_score(t, cl),
            'precision': lambda t, p, cl: precision_score(t, cl),
            'recall': lambda t, p, cl: recall_score(t, cl)
        }
        for m in metrics_list:
            if m in mapping:
                results[m] = mapping[m](t_flat, p_flat, preds)
        return results


def setup_logging(cfg):
    """
    Настройка логирования: вывод в системную консоль и опциональная
    запись в лог-файл (формирует историю экспериментов).
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, cfg.logging.log_level.upper()))

    if not logger.handlers:
        # Хэндлер для вывода в консоль
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(log_fmt))
        logger.addHandler(h)

        # Хэндлер для записи в файл
        if cfg.logging.log_to_file:
            os.makedirs(cfg.general.artifacts_path, exist_ok=True)
            # os.path.join делает склеивание путей безопасным для любой ОС
            log_path = os.path.join(cfg.general.artifacts_path, "dl_training.log")
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_fmt))
            logger.addHandler(fh)
    return logger


class AdvancedCategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Продвинутый заполнитель пропусков для категориальных признаков.
    Реализует стратегии заполнения самым частым значением или специальным тегом "Unknown".
    """

    def __init__(self, strategy='unknown'):
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            series = X_df[col].dropna()
            # Псевдолейблинг для категорий здесь реализован как fallback на моду
            if self.strategy == 'most_common' or self.strategy == 'pseudo_labeling':
                self.fill_values_[col] = series.mode()[0] if not series.empty else "Unknown"
            else:
                # Стратегия unknown просто выделяет пропуски в отдельный класс
                self.fill_values_[col] = "Unknown"
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, val in self.fill_values_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(val)
        return X_df.values


class AdvancedNumericalImputer(BaseEstimator, TransformerMixin):
    """
    Продвинутый заполнитель пропусков для числовых признаков.
    Помимо базовых статистик поддерживает математические сдвиги и предсказание пропусков (MICE).
    """

    def __init__(self, strategy='median', tree_const=100, seed=42):
        self.strategy = strategy
        self.tree_const = tree_const
        self.seed = seed
        self.fill_values_ = {}
        self.iterative_imputer = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)

        # Стратегия pseudo_labeling обучает ML-модель для предсказания отсутствующих значений
        if self.strategy == 'pseudo_labeling':
            self.iterative_imputer = IterativeImputer(random_state=self.seed)
            self.iterative_imputer.fit(X_df)
            return self

        # Стратегии, вычисляющие константы для заполнения
        for col in X_df.columns:
            series = X_df[col]
            if self.strategy == 'tree_custom':
                self.fill_values_[col] = series.min() - self.tree_const
            elif self.strategy == 'min_mean_diff':
                self.fill_values_[col] = series.min() - series.mean()
            elif self.strategy == 'max_mean_sum':
                self.fill_values_[col] = series.max() + series.mean()
            elif self.strategy == 'mean':
                self.fill_values_[col] = series.mean()
            else:
                self.fill_values_[col] = series.median()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if self.strategy == 'pseudo_labeling':
            return self.iterative_imputer.transform(X_df)

        for col, val in self.fill_values_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(val)
        return X_df.values


class FeatureEngineer:
    """Генерация новых признаков (Feature Engineering) до подачи в нейросеть."""

    def transform(self, df):
        df = df.copy()
        # Вычисление размера семьи (Сиблинги + Родители/Дети + Сам пассажир)
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Бинаризация строкового пола в числовой (необходимо для работы сети)
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).fillna(0)
        return df


class EarlyStopping:
    """
    Механизм ранней остановки (Early Stopping).
    Прерывает обучение, если целевая метрика не улучшается в течение заданного числа эпох (patience),
    защищая модель от переобучения. Сохраняет лучшие найденные веса в память (deepcopy).
    """

    def __init__(self, patience=15, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        self.best_metrics = {}

    def __call__(self, monitor_value, model, current_metrics):
        # Инициализация на первой эпохе
        if self.best_score is None:
            self.best_score = monitor_value
            self.best_metrics = current_metrics
            self.save_checkpoint(model)
        else:
            # Определение того, стало ли лучше (в зависимости от поиска максимума или минимума)
            if self.mode == 'min':
                is_better = monitor_value < self.best_score - self.min_delta
            else:
                is_better = monitor_value > self.best_score + self.min_delta

            if is_better:
                # Результат улучшился: сбрасываем счетчик "терпения"
                self.best_score = monitor_value
                self.best_metrics = current_metrics
                self.counter = 0
                self.save_checkpoint(model)
            else:
                # Результат не улучшился: увеличиваем счетчик
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model):
        """Сохранение лучших весов в оперативную/GPU память."""
        self.best_weights = copy.deepcopy(model.state_dict())


class DLPipeline:
    """
    Главный оркестратор пайплайна обучения:
    Управляет разделением данных, препроцессингом, обучением нейросети (фолд за фолдом)
    и формированием финального сабмита (инференсом).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(self.cfg.general.seed)
        self.logger = setup_logging(cfg)

        # Безопасный выбор устройства (Device). Fallback на CPU, если GPU нет.
        if self.cfg.general.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA is not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.cfg.general.device)

        self.fe = FeatureEngineer()
        # Список для хранения метаданных размерностей категорий каждого фолда
        self.fold_info = []

        # Выделенный генератор для Dataloader'а для воспроизводимости шаффлинга батчей
        self.g = torch.Generator()
        self.g.manual_seed(self.cfg.general.seed)

        self.dl_kwargs = {
            'batch_size': self.cfg.training.batch_size,
            'num_workers': self.cfg.training.num_workers,
            'pin_memory': self.cfg.training.pin_memory,
            'worker_init_fn': seed_worker,
            'generator': self.g
        }

    def _prepare_data_split(self, X_tr_raw, X_val_raw, fold_idx):
        """
        Изолированный препроцессинг (обучается строго на Train, применяется на Val).
        Предотвращает Data Leakage (утечку данных из валидации в обучение).
        """
        num_cols = list(self.cfg.features.num_cols)
        cat_cols = list(self.cfg.features.cat_cols)

        # 1. Заполнение пропусков в числовых фичах
        num_imp_strat = self.cfg.features.get('num_impute_strategy', 'median')
        num_imputer = AdvancedNumericalImputer(
            strategy=num_imp_strat,
            tree_const=self.cfg.features.get('tree_const', 100),
            seed=self.cfg.general.seed
        )

        # 2. Масштабирование чисел
        scaler_type = self.cfg.features.get('scaler_type', 'standard')
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'none':
            scaler = FunctionTransformer()
        else:
            scaler = StandardScaler()

        X_tr_num_imp = num_imputer.fit_transform(X_tr_raw[num_cols])
        X_val_num_imp = num_imputer.transform(X_val_raw[num_cols])

        X_tr_num = scaler.fit_transform(X_tr_num_imp)
        X_val_num = scaler.transform(X_val_num_imp)

        # 3. Заполнение пропусков в категориях
        cat_imp_strat = self.cfg.features.get('cat_impute_strategy', 'most_frequent')
        cat_imputer = AdvancedCategoricalImputer(strategy=cat_imp_strat)

        X_tr_cat_imp = cat_imputer.fit_transform(X_tr_raw[cat_cols])
        X_val_cat_imp = cat_imputer.transform(X_val_raw[cat_cols])

        # 4. Кодирование категорий в числа (индексы).
        # unknown_value=-1 заменяет невидимые классы на -1. Прибавляя 1, мы сдвигаем все индексы
        # так, чтобы неизвестные категории стали нулями (padding_idx в слое nn.Embedding).
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr_cat = encoder.fit_transform(X_tr_cat_imp).astype(int) + 1
        X_val_cat = encoder.transform(X_val_cat_imp).astype(int) + 1

        # Сохранение обученных препроцессоров для этапа инференса (теста)
        fold_path = os.path.join(self.cfg.general.artifacts_path, f"fold_{fold_idx}")
        os.makedirs(fold_path, exist_ok=True)

        with open(os.path.join(fold_path, "num_imputer.pkl"), 'wb') as f:
            pickle.dump(num_imputer, f)
        with open(os.path.join(fold_path, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(fold_path, "cat_imputer.pkl"), 'wb') as f:
            pickle.dump(cat_imputer, f)
        with open(os.path.join(fold_path, "encoder.pkl"), 'wb') as f:
            pickle.dump(encoder, f)

        # Расчет размеров словаря для каждой категориальной фичи (нужно для nn.Embedding)
        cat_dims = [len(encoder.categories_[i]) + 1 for i in range(len(cat_cols))]

        return X_tr_num, X_tr_cat, X_val_num, X_val_cat, cat_dims

    def _train_one_fold(self, tr_loader, val_loader, cat_dims, cont_dim, fold_idx):
        """Обучение нейросети на одном разбиении данных."""
        model = FlexibleTitanicNet(cont_dim, cat_dims, self.cfg).to(self.device)

        # Инициализация оптимизатора
        opt_name = self.cfg.training.get('optimizer', 'adam').lower()
        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.training.lr,
                                          weight_decay=self.cfg.training.weight_decay)
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.training.lr, momentum=0.9,
                                        weight_decay=self.cfg.training.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.training.lr,
                                         weight_decay=self.cfg.training.weight_decay)

        # Функция потерь для бинарной классификации. Ожидает на вход "сырые" логиты сети.
        criterion = nn.BCEWithLogitsLoss()

        # Инициализация планировщика скорости обучения (Learning Rate Scheduler)
        sched_type = self.cfg.training.get('scheduler_type', 'plateau').lower()
        if sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.epochs)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=self.cfg.training.monitor_mode, factor=self.cfg.training.factor,
                patience=self.cfg.training.patience
            )

        es = EarlyStopping(
            patience=self.cfg.training.early_stopping_patience,
            min_delta=self.cfg.training.min_delta,
            mode=self.cfg.training.monitor_mode
        )

        for epoch in range(self.cfg.training.epochs):
            # --- Этап обучения (Тренировка) ---
            model.train()
            for x_cont, x_cat, y in tr_loader:
                x_cont, x_cat, y = x_cont.to(self.device), x_cat.to(self.device), y.to(self.device)

                # Обнуляем градиенты с предыдущего шага
                optimizer.zero_grad()

                # Прямой проход (Forward pass)
                logits = model(x_cont, x_cat)

                # Расчет ошибки
                loss = criterion(logits, y)

                # Обратный проход (Backward pass - расчет градиентов)
                loss.backward()

                # Защита от "взрывающихся градиентов" (Gradient Clipping)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Обновление весов
                optimizer.step()

            # --- Этап оценки (Валидация) ---
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():  # Отключаем граф вычислений для экономии памяти
                for x_cont, x_cat, y in val_loader:
                    x_cont, x_cat = x_cont.to(self.device), x_cat.to(self.device)
                    logits = model(x_cont, x_cat)
                    # Пропускаем логиты через сигмоиду для получения вероятностей (0..1)
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    all_targets.extend(y.numpy())

            # Расчет метрик на всей валидационной выборке
            current_metrics = MetricFactory.calculate(
                np.array(all_targets),
                np.array(all_preds),
                self.cfg.training.metrics
            )
            monitor_val = current_metrics.get(self.cfg.training.monitor_metric, 0)

            # Обновление планировщика LR
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_val)
            else:
                scheduler.step()

            # Проверка Early Stopping
            es(monitor_val, model, current_metrics)

            # Логирование
            if (epoch + 1) % self.cfg.logging.verbose_step == 0:
                metrics_str = " | ".join([f"{k.upper()}: {v:.4f}" for k, v in current_metrics.items()])
                self.logger.info(f"  Epoch {epoch + 1:03d} | {metrics_str} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if es.early_stop:
                self.logger.info(f"  Early stopping triggered at epoch {epoch + 1}")
                break

        # Сохранение лучших весов на жесткий диск
        if es.best_weights is not None:
            model_path = os.path.join(self.cfg.general.artifacts_path, f"fold_{fold_idx}", "model.pth")
            torch.save(es.best_weights, model_path)

        # Сохранение мета-информации для последующего инференса
        self.fold_info.append({'cat_dims': cat_dims, 'cont_dim': cont_dim})

        return es.best_metrics

    def predict_on_test(self):
        """Инференс: предсказание на отложенной (тестовой) выборке с усреднением по всем фолдам."""
        self.logger.info("Starting OOM-safe inference on test.csv...")
        test_df_raw = pd.read_csv(self.cfg.inference.test_data_path)
        test_df = self.fe.transform(test_df_raw)

        num_cols = list(self.cfg.features.num_cols)
        cat_cols = list(self.cfg.features.cat_cols)

        # Список списков: предсказания от каждой из N моделей
        fold_probs = []

        for i, info in enumerate(self.fold_info):
            path = os.path.join(self.cfg.general.artifacts_path, f"fold_{i}")

            # Загружаем препроцессоры, обученные на тренировочном сплите i-го фолда
            with open(os.path.join(path, "num_imputer.pkl"), 'rb') as f:
                num_imputer = pickle.load(f)
            with open(os.path.join(path, "scaler.pkl"), 'rb') as f:
                scaler = pickle.load(f)
            with open(os.path.join(path, "cat_imputer.pkl"), 'rb') as f:
                cat_imputer = pickle.load(f)
            with open(os.path.join(path, "encoder.pkl"), 'rb') as f:
                encoder = pickle.load(f)

            # Трансформация (Только transform, без fit!)
            X_num_imp = num_imputer.transform(test_df[num_cols])
            X_num = scaler.transform(X_num_imp)

            X_cat_imp = cat_imputer.transform(test_df[cat_cols])
            X_cat = encoder.transform(X_cat_imp).astype(int) + 1

            # Помещение в DataLoader (OOM-safe подход для огромных датасетов)
            test_dataset = TitanicDLDataset(X_num, X_cat, y=None)
            test_loader = DataLoader(test_dataset, shuffle=False, **self.dl_kwargs)

            # Инициализация и загрузка весов
            model = FlexibleTitanicNet(info['cont_dim'], info['cat_dims'], self.cfg).to(self.device)
            model.load_state_dict(torch.load(os.path.join(path, "model.pth"), weights_only=True))
            model.eval()

            current_fold_preds = []
            with torch.no_grad():
                for x_num_t, x_cat_t in test_loader:
                    x_num_t = x_num_t.to(self.device)
                    x_cat_t = x_cat_t.to(self.device)
                    logits = model(x_num_t, x_cat_t)
                    current_fold_preds.extend(torch.sigmoid(logits).cpu().numpy())

            fold_probs.append(current_fold_preds)

        # Усреднение вероятностей со всех фолдов (простой ансамбль)
        avg_probs = np.mean(fold_probs, axis=0)

        id_col = str(self.cfg.general.id_col)
        target_col = str(self.cfg.general.target_col)

        # Формирование итогового DataFrame для Kaggle
        submission = pd.DataFrame({
            id_col: test_df_raw[id_col],
            # Конвертация усредненной вероятности в жесткий класс (0/1)
            target_col: (avg_probs > 0.5).astype(int).flatten()
        })
        submission.to_csv(self.cfg.inference.submission_path, index=False)
        self.logger.info(f"Submission saved to {self.cfg.inference.submission_path}")

    def run(self):
        """Точка входа. Чтение данных, инициализация разбиений и запуск цикла по фолдам."""
        self.logger.info(f"--- Start Experiment: {self.cfg.general.experiment_name} ---")
        self.logger.info(f"Device set to: {self.device}")

        df = self.fe.transform(pd.read_csv(self.cfg.general.data_path))
        target_col = str(self.cfg.general.target_col)

        if target_col not in df.columns:
            error_msg = f"ОШИБКА: Целевая колонка '{target_col}' не найдена."
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        drop_cols = [target_col] + list(self.cfg.features.dropped_features)
        X = df.drop(columns=drop_cols, errors='ignore')
        y = df[target_col].values

        fold_results = []

        # Настройка стратегии валидации
        if self.cfg.training.use_cv:
            self.logger.info(f"Running Stratified CV with {self.cfg.training.cv_folds} folds...")
            skf = StratifiedKFold(n_splits=self.cfg.training.cv_folds, shuffle=True, random_state=self.cfg.general.seed)
            splits = skf.split(X, y)
        else:
            self.logger.info(f"CV DISABLED. Running single Hold-Out (val_size={self.cfg.training.val_size})...")
            indices = np.arange(len(X))
            tr_idx, val_idx = train_test_split(
                indices, test_size=self.cfg.training.val_size, random_state=self.cfg.general.seed, stratify=y
            )
            splits = [(tr_idx, val_idx)]

        # Цикл обучения
        for fold, (tr_idx, val_idx) in enumerate(splits):
            fold_label = f"Fold {fold + 1}" if self.cfg.training.use_cv else "Single Split"
            self.logger.info(f"\n--- {fold_label} ---")

            # Получаем отмасштабированные и закодированные тензоры
            X_tr_n, X_tr_c, X_val_n, X_val_c, cat_dims = self._prepare_data_split(
                X.iloc[tr_idx], X.iloc[val_idx], fold
            )

            # Оборачиваем данные в объекты PyTorch Dataset
            tr_dataset = TitanicDLDataset(X_tr_n, X_tr_c, y[tr_idx])
            val_dataset = TitanicDLDataset(X_val_n, X_val_c, y[val_idx])

            # Инициализация DataLoader'ов для пакетной (batch) выдачи
            tr_loader = DataLoader(tr_dataset, shuffle=True, **self.dl_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **self.dl_kwargs)

            # Обучение одной модели на одном фолде
            fold_best_metrics = self._train_one_fold(tr_loader, val_loader, cat_dims, X_tr_n.shape[1], fold)
            fold_results.append(fold_best_metrics)

        # Вывод итоговой статистики по всем разбиениям
        self.logger.info("\n=== Final Results ===")
        for m in self.cfg.training.metrics:
            vals = [res[m] for res in fold_results]
            if self.cfg.training.use_cv:
                self.logger.info(f"  {m.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
            else:
                self.logger.info(f"  {m.upper()}: {vals[0]:.4f}")

        # Запуск финального инференса
        if self.cfg.inference.run_inference:
            self.predict_on_test()


if __name__ == "__main__":
    cfg = get_dl_config()
    pipeline = DLPipeline(cfg)
    pipeline.run()