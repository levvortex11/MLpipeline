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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, RobustScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Импорт для экспериментального IterativeImputer (стратегия pseudo_labeling)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin

from HP_config_DL import get_dl_config
from HP_model_DL import HousePricesDLDataset, FlexibleHousePricesNet


def seed_everything(seed=42):
    """
    Глобальная фиксация всех генераторов псевдослучайных чисел.
    Обеспечивает воспроизводимость (детерминированность) результатов обучения.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Фиксация seed внутри потоков (воркеров) DataLoader'а.
    Защищает от дублирования аугментаций/порядка данных при многопоточной загрузке.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MetricFactory:
    """Утилитный класс для удобного и безопасного расчета метрик sklearn."""

    @staticmethod
    def calculate(targets, preds, metrics_list):
        results = {}
        # ИСПРАВЛЕНИЕ: Добавлен .flatten() для защиты от багов в sklearn.
        # sklearn иногда конфликтует с массивами (N, 1), считая метрику поколоночно.
        t_flat = np.array(targets).flatten()
        p_flat = np.array(preds).flatten()

        mapping = {
            'rmse': lambda t, p: np.sqrt(mean_squared_error(t, p)),
            'mae': lambda t, p: mean_absolute_error(t, p),
            'r2': lambda t, p: r2_score(t, p)
        }
        for m in metrics_list:
            if m in mapping:
                results[m] = mapping[m](t_flat, p_flat)
        return results


def setup_logging(cfg):
    """Настройка логирования: выводит в консоль и параллельно сохраняет в лог-файл."""
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, cfg.logging.log_level.upper()))

    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(log_fmt))
        logger.addHandler(h)
        if cfg.logging.log_to_file:
            os.makedirs(cfg.general.artifacts_path, exist_ok=True)
            # Безопасное склеивание пути логов через os.path.join
            log_path = os.path.join(cfg.general.artifacts_path, "dl_training.log")
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(log_fmt))
            logger.addHandler(fh)
    return logger


class AdvancedCategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Продвинутый заполнитель пропусков категорий.
    Позволяет гибко задавать стратегию (в отличие от SimpleImputer).
    """

    def __init__(self, strategy='unknown'):
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            series = X_df[col].dropna()
            if self.strategy == 'most_common' or self.strategy == 'pseudo_labeling':
                # Для категорий pseudo_labeling делает fallback на моду (самое частое значение)
                self.fill_values_[col] = series.mode()[0] if not series.empty else "Unknown"
            else:
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
    Продвинутый заполнитель пропусков для чисел.
    Поддерживает псевдолейблирование (предсказание пропусков) через IterativeImputer.
    """

    def __init__(self, strategy='median', tree_const=100, seed=42):
        self.strategy = strategy
        self.tree_const = tree_const
        self.seed = seed
        self.fill_values_ = {}
        self.iterative_imputer = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        if self.strategy == 'pseudo_labeling':
            # ML-метод: предсказывает пропуск на основе остальных признаков
            self.iterative_imputer = IterativeImputer(random_state=self.seed)
            self.iterative_imputer.fit(X_df)
            return self

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
        if self.strategy == 'pseudo_labeling': return self.iterative_imputer.transform(X_df)
        for col, val in self.fill_values_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(val)
        return X_df.values


class FeatureEngineer:
    """Бизнес-логика создания новых фичей перед подачей в нейросеть."""

    def transform(self, df):
        df = df.copy()
        # Общая площадь дома — сильнейший признак для предсказания цены
        if '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns and 'TotalBsmtSF' in df.columns:
            df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
        return df


class EarlyStopping:
    """
    Алгоритм ранней остановки. Предотвращает переобучение, останавливая цикл,
    если целевая метрика на валидации не улучшается в течение patience эпох.
    Сохраняет лучшие веса в оперативной памяти.
    """

    def __init__(self, patience=15, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        self.best_metrics = {}

    def __call__(self, monitor_value, model, current_metrics):
        if self.best_score is None:
            self.best_score = monitor_value
            self.best_metrics = current_metrics
            self.save_checkpoint(model)
        else:
            if self.mode == 'min':
                is_better = monitor_value < self.best_score - self.min_delta
            else:
                is_better = monitor_value > self.best_score + self.min_delta

            if is_better:
                self.best_score = monitor_value
                self.best_metrics = current_metrics
                self.counter = 0
                self.save_checkpoint(model)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model):
        """Делает 'снимок' весов модели."""
        self.best_weights = copy.deepcopy(model.state_dict())


class DLPipeline:
    """
    Главный класс пайплайна обучения: управляет загрузкой, трансформацией,
    тренировкой модели и инференсом.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(self.cfg.general.seed)
        self.logger = setup_logging(cfg)

        # Безопасный выбор устройства (Device)
        if self.cfg.general.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA is not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.cfg.general.device)

        self.fe = FeatureEngineer()
        # Список для метаданных каждого фолда (сохраняем размеры словарей категорий)
        self.fold_info = []

        # Выделенный генератор для контролируемого шаффлинга данных в DataLoader
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
        Изолированный препроцессинг. Импутеры и скейлеры обучаются (fit) СТРОГО
        на тренировочной части, и лишь применяются (transform) к валидационной.
        Предотвращает Data Leakage (утечку знаний).
        """
        num_cols = list(self.cfg.features.num_cols)
        cat_cols = list(self.cfg.features.cat_cols)

        # 1. Заполнение чисел
        num_imp_strat = self.cfg.features.get('num_impute_strategy', 'median')
        num_imputer = AdvancedNumericalImputer(
            strategy=num_imp_strat,
            tree_const=self.cfg.features.get('tree_const', 100),
            seed=self.cfg.general.seed
        )

        # 2. Скейлинг
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

        # 3. Заполнение категорий
        cat_imp_strat = self.cfg.features.get('cat_impute_strategy', 'unknown')
        cat_imputer = AdvancedCategoricalImputer(strategy=cat_imp_strat)

        X_tr_cat_imp = cat_imputer.fit_transform(X_tr_raw[cat_cols])
        X_val_cat_imp = cat_imputer.transform(X_val_raw[cat_cols])

        # 4. Кодирование категорий в индексы (для слоя Embedding)
        # Класс "-1" (неизвестный класс) сдвигается в "0" и становится padding_idx
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr_cat = encoder.fit_transform(X_tr_cat_imp).astype(int) + 1
        X_val_cat = encoder.transform(X_val_cat_imp).astype(int) + 1

        fold_path = os.path.join(self.cfg.general.artifacts_path, f"fold_{fold_idx}")
        os.makedirs(fold_path, exist_ok=True)

        # Сохранение обученных препроцессоров для инференса на тесте
        with open(os.path.join(fold_path, "num_imputer.pkl"), 'wb') as f:
            pickle.dump(num_imputer, f)
        with open(os.path.join(fold_path, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(fold_path, "cat_imputer.pkl"), 'wb') as f:
            pickle.dump(cat_imputer, f)
        with open(os.path.join(fold_path, "encoder.pkl"), 'wb') as f:
            pickle.dump(encoder, f)

        # Подсчет размеров словаря для каждой фичи (нужно для инициализации nn.Embedding)
        cat_dims = [len(encoder.categories_[i]) + 1 for i in range(len(cat_cols))]

        return X_tr_num, X_tr_cat, X_val_num, X_val_cat, cat_dims

    def _train_one_fold(self, tr_loader, val_loader, cat_dims, cont_dim, fold_idx):
        """Обучение нейросети на одном разбиении данных."""
        model = FlexibleHousePricesNet(cont_dim, cat_dims, self.cfg).to(self.device)

        # Выбор оптимизатора
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

        # Функция потерь: MSE (Mean Squared Error) стандартна для регрессии
        criterion = nn.MSELoss()

        # Планировщик скорости обучения
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
            # --- Этап обучения ---
            model.train()
            for x_cont, x_cat, y in tr_loader:
                x_cont, x_cat, y = x_cont.to(self.device), x_cat.to(self.device), y.to(self.device)

                # Очистка градиентов с предыдущего шага
                optimizer.zero_grad()

                # Прямой проход (вычисление логитов)
                logits = model(x_cont, x_cat)

                # Вычисление потерь
                loss = criterion(logits, y)

                # Обратный проход (вычисление градиентов)
                loss.backward()

                # Защита от "взрывающихся" градиентов
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Обновление весов нейросети
                optimizer.step()

            # --- Этап валидации ---
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():  # Отключение градиентов (экономит память и ускоряет инференс)
                for x_cont, x_cat, y in val_loader:
                    x_cont, x_cat = x_cont.to(self.device), x_cat.to(self.device)
                    logits = model(x_cont, x_cat)
                    all_preds.extend(logits.cpu().numpy())
                    all_targets.extend(y.numpy())

            if self.cfg.general.get('log_target', False):
                eval_targets = np.expm1(all_targets)
                eval_preds = np.expm1(all_preds)
            else:
                eval_targets = all_targets
                eval_preds = all_preds

            current_metrics = MetricFactory.calculate(
                np.array(eval_targets),
                np.array(eval_preds),
                self.cfg.training.metrics
            )

            monitor_val = current_metrics.get(self.cfg.training.monitor_metric, 0)

            # Шаг планировщика
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_val)
            else:
                scheduler.step()

            # Проверка Early Stopping
            es(monitor_val, model, current_metrics)

            if (epoch + 1) % self.cfg.logging.verbose_step == 0:
                metrics_str = " | ".join([f"{k.upper()}: {v:.4f}" for k, v in current_metrics.items()])
                self.logger.info(f"  Epoch {epoch + 1:03d} | {metrics_str} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if es.early_stop:
                self.logger.info(f"  Early stopping triggered at epoch {epoch + 1}")
                break

        # Сохранение финальных весов на диск
        if es.best_weights is not None:
            torch.save(es.best_weights, os.path.join(self.cfg.general.artifacts_path, f"fold_{fold_idx}/model.pth"))

        self.fold_info.append({'cat_dims': cat_dims, 'cont_dim': cont_dim})

        return es.best_metrics

    def predict_on_test(self):
        """Инференс: делает предсказания на отложенной выборке, усредняя ответы N обученных моделей."""
        self.logger.info("Starting OOM-safe inference on test.csv...")
        test_df_raw = pd.read_csv(self.cfg.inference.test_data_path)
        test_df = self.fe.transform(test_df_raw)

        num_cols = list(self.cfg.features.num_cols)
        cat_cols = list(self.cfg.features.cat_cols)
        fold_preds = []

        for i, info in enumerate(self.fold_info):
            path = os.path.join(self.cfg.general.artifacts_path, f"fold_{i}")

            # Загружаем препроцессоры именно того фолда, которым модель обучалась
            with open(f"{path}/num_imputer.pkl", 'rb') as f:
                num_imputer = pickle.load(f)
            with open(f"{path}/scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            with open(f"{path}/cat_imputer.pkl", 'rb') as f:
                cat_imputer = pickle.load(f)
            with open(f"{path}/encoder.pkl", 'rb') as f:
                encoder = pickle.load(f)

            # Трансформация данных (только transform!)
            X_num_imp = num_imputer.transform(test_df[num_cols])
            X_num = scaler.transform(X_num_imp)

            X_cat_imp = cat_imputer.transform(test_df[cat_cols])
            X_cat = encoder.transform(X_cat_imp).astype(int) + 1

            # DataLoader делает инференс OOM-safe (защищает от Out Of Memory)
            test_dataset = HousePricesDLDataset(X_num, X_cat, y=None)
            test_loader = DataLoader(test_dataset, shuffle=False, **self.dl_kwargs)

            model = FlexibleHousePricesNet(info['cont_dim'], info['cat_dims'], self.cfg).to(self.device)
            # weights_only=True — безопасный метод десериализации весов в новых версиях PyTorch
            model.load_state_dict(torch.load(f"{path}/model.pth", weights_only=True))
            model.eval()

            current_fold_preds = []
            with torch.no_grad():
                for x_num_t, x_cat_t in test_loader:
                    x_num_t = x_num_t.to(self.device)
                    x_cat_t = x_cat_t.to(self.device)
                    logits = model(x_num_t, x_cat_t)
                    current_fold_preds.extend(logits.cpu().numpy())

            fold_preds.append(current_fold_preds)

        # Усреднение предсказаний всех фолдов (создание простого ансамбля)
        avg_preds = np.mean(fold_preds, axis=0)

        # Если целевая переменная логарифмировалась при обучении, нужно вернуть её в реальные доллары
        if self.cfg.general.get('log_target', False):
            final_predictions = np.expm1(avg_preds.flatten())
        else:
            final_predictions = avg_preds.flatten()

        id_col = str(self.cfg.general.id_col)
        target_col = str(self.cfg.general.target_col)

        submission = pd.DataFrame({
            id_col: test_df_raw[id_col],
            target_col: final_predictions
        })
        submission.to_csv(self.cfg.inference.submission_path, index=False)
        self.logger.info(f"Submission saved to {self.cfg.inference.submission_path}")

    def run(self):
        """Главный метод запуска всего пайплайна."""
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
        y_raw = df[target_col].values

        # Логарифмирование целевой переменной (защита от искажений из-за домов-выбросов)
        is_log_target = self.cfg.general.get('log_target', False)
        if is_log_target:
            self.logger.info("Applying log1p transformation to target variable.")
            y = np.log1p(y_raw)
        else:
            y = y_raw

        fold_results = []

        # Разделение данных
        if self.cfg.training.use_cv:
            self.logger.info(f"Running CV with {self.cfg.training.cv_folds} folds...")
            kf = KFold(n_splits=self.cfg.training.cv_folds, shuffle=True, random_state=self.cfg.general.seed)
            splits = kf.split(X, y)
        else:
            self.logger.info(f"CV DISABLED. Running single Hold-Out (val_size={self.cfg.training.val_size})...")
            indices = np.arange(len(X))
            tr_idx, val_idx = train_test_split(
                indices, test_size=self.cfg.training.val_size, random_state=self.cfg.general.seed
            )
            splits = [(tr_idx, val_idx)]

        for fold, (tr_idx, val_idx) in enumerate(splits):
            fold_label = f"Fold {fold + 1}" if self.cfg.training.use_cv else "Single Split"
            self.logger.info(f"\n--- {fold_label} ---")

            X_tr_n, X_tr_c, X_val_n, X_val_c, cat_dims = self._prepare_data_split(
                X.iloc[tr_idx], X.iloc[val_idx], fold
            )

            # Инициализация датасетов и лоадеров PyTorch
            tr_dataset = HousePricesDLDataset(X_tr_n, X_tr_c, y[tr_idx])
            val_dataset = HousePricesDLDataset(X_val_n, X_val_c, y[val_idx])

            tr_loader = DataLoader(tr_dataset, shuffle=True, **self.dl_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **self.dl_kwargs)

            fold_best_metrics = self._train_one_fold(tr_loader, val_loader, cat_dims, X_tr_n.shape[1], fold)
            fold_results.append(fold_best_metrics)

        # Вывод финальных результатов
        self.logger.info("\n=== Final Results ===")
        for m in self.cfg.training.metrics:
            vals = [res[m] for res in fold_results]
            if self.cfg.training.use_cv:
                self.logger.info(f"  {m.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
            else:
                self.logger.info(f"  {m.upper()}: {vals[0]:.4f}")

        # Инференс на тестовых данных
        if self.cfg.inference.run_inference:
            self.predict_on_test()


if __name__ == "__main__":
    cfg = get_dl_config()
    pipeline = DLPipeline(cfg)
    pipeline.run()