import pandas as pd
import numpy as np
import os
import logging
import sys
import joblib

# Импорты sklearn для препроцессинга и валидации
from sklearn import metrics, pipeline, compose, preprocessing, impute
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import make_column_selector

# Импорт IterativeImputer для стратегии pseudo_labeling на числах
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Импорты наших фабрик
from factory import get_model, plot_importance, plot_learning_curve
from config import get_config


def seed_everything(seed=42):
    """
    Фиксация всех случайных факторов для воспроизводимости результатов.
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(cfg):
    """
    Настройка логирования: вывод в консоль и запись в файл execution.log.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(console_handler)

    if not os.path.exists(cfg.general.artifacts_path):
        os.makedirs(cfg.general.artifacts_path)

    # ИСПРАВЛЕНИЕ: Безопасное склеивание пути (защита от забытого слеша)
    log_path = os.path.join(cfg.general.artifacts_path, "execution.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(file_handler)


class AdvancedCategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Кастомный трансформер для обработки категориальных пропусков.
    Реализует логику most_common, unknown и pseudo_labeling.
    """

    def __init__(self, strategy='unknown'):
        self.strategy = strategy
        self.fill_values_ = {}
        self.feature_names_in_ = []  # Запоминаем имена для графиков

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()

        for col in X_df.columns:
            series = X_df[col].dropna()
            if self.strategy == 'most_common' or self.strategy == 'pseudo_labeling':
                # Для pseudo_labeling на категориях пока используем fallback на моду,
                # так как sklearn не поддерживает это из коробки без сложных оберток.
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

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(self.feature_names_in_)


class AdvancedNumericalImputer(BaseEstimator, TransformerMixin):
    """
    Кастомный заполнитель пропусков для чисел.
    Поддерживает псевдолейблирование через IterativeImputer.
    """

    def __init__(self, strategy='median', tree_const=100, seed=42):
        self.strategy = strategy
        self.tree_const = tree_const
        self.seed = seed
        self.fill_values_ = {}
        self.iterative_imputer = None
        self.feature_names_in_ = []  # Запоминаем имена для защиты от краша

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()

        # Если выбрана стратегия pseudo_labeling, обучаем IterativeImputer на всей матрице X
        if self.strategy == 'pseudo_labeling':
            self.iterative_imputer = IterativeImputer(random_state=self.seed)
            self.iterative_imputer.fit(X_df)
            return self

        # Классические стратегии: обрабатываем каждую колонку индивидуально
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

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(self.feature_names_in_)


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Трансформер для создания новых признаков (Размер семьи и флаг 'Одиночка').
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'SibSp' in X.columns and 'Parch' in X.columns:
            X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
            X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return None
        features = list(input_features)
        if 'FamilySize' not in features: features.append('FamilySize')
        if 'IsAlone' not in features: features.append('IsAlone')
        return np.array(features)


class ProPipeline:
    """
    Главный оркестратор ML-процесса.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(self.cfg.general.seed)
        setup_logging(self.cfg)
        self.logger = logging.getLogger(__name__)
        self.final_drop_cols = []

        # Маппер метрик для автоматизации
        self.METRIC_MAPPER = {
            'roc_auc': (metrics.roc_auc_score, 'roc_auc', 'proba'),
            'accuracy': (metrics.accuracy_score, 'accuracy', 'labels'),
            'f1': (metrics.f1_score, 'f1', 'labels')
        }

    def _prepare_df(self, df, is_train=True):
        """
        Предварительная чистка данных и защита от Data Leakage.
        """
        df = df.copy()
        target = self.cfg.general.target_col

        if is_train:
            if target not in df.columns:
                self.logger.error(f"Колонка {target} не найдена в данных! Проверьте конфиг или путь к файлу.")
                raise KeyError(target)

            df = df.drop_duplicates().reset_index(drop=True)
            const_cols = [c for c in df.columns if df[c].nunique() <= 1]
            to_drop = list(self.cfg.features.dropped_features) + const_cols

            # Удаление мультиколлинеарности (только на трейне!)
            num_df = df.drop(columns=[target], errors='ignore').select_dtypes(include=[np.number])
            if not num_df.empty:
                corr = num_df.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop += [col for col in upper.columns if any(upper[col] > self.cfg.features.corr_threshold)]
            self.final_drop_cols = list(set(to_drop))

        # Применяем выученный список удаления ко всем данным
        df = df.drop(columns=self.final_drop_cols, errors='ignore')

        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

        for col in self.cfg.features.force_categorical:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    def _get_preprocessing(self):
        """
        Сборка ColumnTransformer с учетом выбранного в конфиге Scaler и Encoder.
        """
        # --- 1. Выбор Скейлера ---
        s_type = self.cfg.scaling.type
        if s_type == 'minmax':
            scaler = preprocessing.MinMaxScaler()
        elif s_type == 'robust':
            scaler = preprocessing.RobustScaler()
        elif s_type == 'none':
            scaler = preprocessing.FunctionTransformer()
        else:
            scaler = preprocessing.StandardScaler()

        num_pipe = pipeline.Pipeline([
            ('imputer', AdvancedNumericalImputer(strategy=self.cfg.imputation.num_strategy,
                                                 tree_const=self.cfg.imputation.tree_const,
                                                 seed=self.cfg.general.seed)),
            ('scaler', scaler)
        ])

        # --- 2. Выбор Энкодера и Импутера категорий ---
        cat_imputer = AdvancedCategoricalImputer(strategy=self.cfg.imputation.cat_strategy)

        enc_type = self.cfg.encoding.type

        if enc_type == 'target':
            # TargetEncoder: target_type='binary' для Титаника, 'continuous' для регрессии
            t_type = 'binary' if self.cfg.general.metric in ['roc_auc', 'accuracy', 'f1'] else 'continuous'
            encoder = preprocessing.TargetEncoder(target_type=t_type, random_state=self.cfg.general.seed)
        elif enc_type == 'ordinal':
            encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        cat_pipe = pipeline.Pipeline([('imputer', cat_imputer), ('encoder', encoder)])

        # Собираем всё в ColumnTransformer через селекторы типов
        return compose.ColumnTransformer([
            ('num', num_pipe, make_column_selector(dtype_include=np.number)),
            ('cat', cat_pipe, make_column_selector(dtype_exclude=np.number))
        ])

    def run(self):
        """
        Запуск полного цикла: Подготовка -> CV-обучение -> Оценка -> Инференс.
        """
        self.logger.info(f"🚀 Запуск Титаника: {self.cfg.general.experiment_name}")
        metric_name = self.cfg.general.metric
        metric_func, sklearn_scoring, pred_mode = self.METRIC_MAPPER.get(metric_name, self.METRIC_MAPPER['roc_auc'])

        if not os.path.exists(self.cfg.general.data_path):
            self.logger.error("Файл данных не найден!")
            return

        df = self._prepare_df(pd.read_csv(self.cfg.general.data_path), is_train=True)
        X = df.drop(columns=[self.cfg.general.target_col])
        y = df[self.cfg.general.target_col]

        # Итоговый пайплайн
        full_pipe = pipeline.Pipeline([
            ('fe', TitanicFeatureEngineer()),
            ('preprocessor', self._get_preprocessing()),
            ('model', get_model(self.cfg, sklearn_scoring))
        ])

        # Выбор режима валидации
        cv_folds = self.cfg.general.get('cv_folds', 5)
        if self.cfg.general.get('use_cv', True):
            self.logger.info("Запуск Cross-Validation...")
            kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.cfg.general.seed)
            splits = kf.split(X, y)
        else:
            self.logger.info("CV отключена. Одиночное разбиение (Hold-out)...")
            tr_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y,
                                               random_state=self.cfg.general.seed)
            splits = [(tr_idx, val_idx)]

        scores = []
        for fold, (tr_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Обучение
            full_pipe.fit(X_tr, y_tr)

            # Выбор метода предсказания (вероятности для AUC, классы для Accuracy)
            if pred_mode == 'proba':
                preds = full_pipe.predict_proba(X_val)[:, 1]
            else:
                preds = full_pipe.predict(X_val)

            score = metric_func(y_val, preds)
            scores.append(score)
            self.logger.info(f"Сплит {fold} | {metric_name.upper()}: {score:.4f}")

        self.logger.info(f"🏆 Итоговый средний {metric_name.upper()}: {np.mean(scores):.4f}")

        # Отрисовка графиков
        if self.cfg.plots.show_learning_curve:
            plot_learning_curve(full_pipe, X, y, self.cfg, sklearn_scoring)

        if self.cfg.plots.show_feature_importance:
            full_pipe.fit(X, y)
            feature_names = full_pipe.named_steps['preprocessor'].get_feature_names_out()
            plot_importance(full_pipe, feature_names)

        # Сохранение артефактов (ИСПРАВЛЕНИЕ: безопасное склеивание пути)
        model_path = os.path.join(self.cfg.general.artifacts_path, "final_model.joblib")
        joblib.dump(full_pipe, model_path)

        # Инференс
        if self.cfg.general.run_inference and os.path.exists(self.cfg.inference.test_data_path):
            test_raw = pd.read_csv(self.cfg.inference.test_data_path)
            test_df = self._prepare_df(test_raw, is_train=False)

            p_labels = full_pipe.predict(test_df) if self.cfg.inference.output_type == 'labels' else \
            full_pipe.predict_proba(test_df)[:, 1]

            pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': p_labels}).to_csv(
                self.cfg.inference.submission_name, index=False)
            self.logger.info(f"✅ Сабмит {self.cfg.inference.submission_name} создан.")


if __name__ == "__main__":
    ProPipeline(get_config()).run()