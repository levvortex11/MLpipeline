import pandas as pd
import numpy as np
import os
import logging
import sys
import joblib

from sklearn import metrics, pipeline, compose, preprocessing, impute
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import make_column_selector

# Импорт для IterativeImputer (числовое псевдолейблирование)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from factoryHP import get_model, plot_importance, plot_learning_curve
from configHP import get_config


def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(cfg):
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(console_handler)

    if not os.path.exists(cfg.general.artifacts_path):
        os.makedirs(cfg.general.artifacts_path)

    # ИСПРАВЛЕНИЕ: Безопасное склеивание пути
    log_path = os.path.join(cfg.general.artifacts_path, "executionHP.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(file_handler)


class AdvancedCategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='unknown'):
        self.strategy = strategy
        self.fill_values_ = {}
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()

        for col in X_df.columns:
            series = X_df[col].dropna()
            if self.strategy == 'most_common' or self.strategy == 'pseudo_labeling':
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
    Продвинутый заполнитель чисел для House Prices.
    Поддерживает псевдолейблирование через IterativeImputer.
    """

    def __init__(self, strategy='median', tree_const=100, seed=42):
        self.strategy = strategy
        self.tree_const = tree_const
        self.seed = seed
        self.fill_values_ = {}
        self.iterative_imputer = None
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()

        if self.strategy == 'pseudo_labeling':
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
        if self.strategy == 'pseudo_labeling':
            return self.iterative_imputer.transform(X_df)

        for col, val in self.fill_values_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(val)
        return X_df.values

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(self.feature_names_in_)


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if all(c in X.columns for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return None
        features = list(input_features)
        if 'TotalSF' not in features: features.append('TotalSF')
        return np.array(features)


class ProPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.final_drop_cols = []
        seed_everything(self.cfg.general.seed)
        setup_logging(self.cfg)
        self.logger = logging.getLogger(__name__)

        self.METRIC_MAPPER = {
            'mae': (metrics.mean_absolute_error, 'neg_mean_absolute_error', 'labels'),
            'mse': (metrics.mean_squared_error, 'neg_mean_squared_error', 'labels'),
            'rmse': (lambda y, p: np.sqrt(metrics.mean_squared_error(y, p)), 'neg_root_mean_squared_error', 'labels'),
            'r2': (metrics.r2_score, 'r2', 'labels')
        }

    def _prepare_df(self, df, is_train=True):
        df = df.copy()
        target = self.cfg.general.target_col

        if is_train:
            if target not in df.columns:
                raise KeyError(f"Колонка {target} не найдена в данных!")

            df = df.drop_duplicates().reset_index(drop=True)
            const_cols = [c for c in df.columns if df[c].nunique() <= 1]
            to_drop = list(self.cfg.features.dropped_features) + const_cols

            num_df = df.drop(columns=[target], errors='ignore').select_dtypes(include=[np.number])
            if not num_df.empty:
                corr = num_df.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop += [col for col in upper.columns if any(upper[col] > self.cfg.features.corr_threshold)]
            self.final_drop_cols = list(set(to_drop))

        df = df.drop(columns=self.final_drop_cols, errors='ignore')

        for col in self.cfg.features.force_categorical:
            if col in df.columns: df[col] = df[col].astype(str)
        return df

    def _get_preprocessing(self):
        stype = self.cfg.scaling.type
        if stype == 'minmax':
            scaler = preprocessing.MinMaxScaler()
        elif stype == 'robust':
            scaler = preprocessing.RobustScaler()
        elif stype == 'none':
            scaler = preprocessing.FunctionTransformer()
        else:
            scaler = preprocessing.StandardScaler()

        num_pipe = pipeline.Pipeline([
            ('imputer', AdvancedNumericalImputer(strategy=self.cfg.imputation.num_strategy,
                                                 tree_const=self.cfg.imputation.tree_const,
                                                 seed=self.cfg.general.seed)),
            ('scaler', scaler)
        ])

        cat_imputer = AdvancedCategoricalImputer(strategy=self.cfg.imputation.cat_strategy)

        if self.cfg.encoding.type == 'target':
            encoder = preprocessing.TargetEncoder(target_type='continuous', random_state=self.cfg.general.seed)
        elif self.cfg.encoding.type == 'ordinal':
            encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        cat_pipe = pipeline.Pipeline([('imputer', cat_imputer), ('encoder', encoder)])

        return compose.ColumnTransformer([
            ('num', num_pipe, make_column_selector(dtype_include=np.number)),
            ('cat', cat_pipe, make_column_selector(dtype_exclude=np.number))
        ])

    def run(self):
        self.logger.info(f"🚀 Запуск House Prices: {self.cfg.general.experiment_name}")
        metric_name = self.cfg.general.metric
        metric_func, sklearn_scoring, pred_mode = self.METRIC_MAPPER.get(metric_name, self.METRIC_MAPPER['rmse'])

        if not os.path.exists(self.cfg.general.data_path):
            self.logger.error("Файл данных не найден!")
            return

        df = self._prepare_df(pd.read_csv(self.cfg.general.data_path), is_train=True)
        X = df.drop(columns=[self.cfg.general.target_col])
        y = df[self.cfg.general.target_col].values

        is_log_target = self.cfg.general.get('log_target', False)
        if is_log_target:
            y = np.log1p(y)

        full_pipe = pipeline.Pipeline([
            ('fe', HouseFeatureEngineer()),
            ('preprocessor', self._get_preprocessing()),
            ('model', get_model(self.cfg, sklearn_scoring))
        ])

        cv_folds = self.cfg.general.get('cv_folds', 5)
        if self.cfg.general.get('use_cv', True):
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.cfg.general.seed)
            splits = kf.split(X, y)
        else:
            tr_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=self.cfg.general.seed)
            splits = [(tr_idx, val_idx)]

        scores = []
        for fold, (tr_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            full_pipe.fit(X_tr, y_tr)
            preds = full_pipe.predict(X_val)

            if is_log_target:
                preds = np.expm1(preds)
                y_val_eval = np.expm1(y_val)
            else:
                y_val_eval = y_val

            score = metric_func(y_val_eval, preds)
            scores.append(score)
            self.logger.info(f"Сплит {fold} | {metric_name.upper()}: {score:.4f}")

        self.logger.info(f"🏆 Средний {metric_name.upper()}: {np.mean(scores):.4f}")

        if self.cfg.plots.show_feature_importance:
            full_pipe.fit(X, y)
            feature_names = full_pipe.named_steps['preprocessor'].get_feature_names_out()
            plot_importance(full_pipe, feature_names)

        # ИСПРАВЛЕНИЕ: Безопасное склеивание пути
        model_path = os.path.join(self.cfg.general.artifacts_path, "final_modelHP.joblib")
        joblib.dump(full_pipe, model_path)

        if self.cfg.general.run_inference and os.path.exists(self.cfg.inference.test_data_path):
            test_raw = pd.read_csv(self.cfg.inference.test_data_path)
            test_df = self._prepare_df(test_raw, is_train=False)
            p_labels = full_pipe.predict(test_df)

            if is_log_target:
                p_labels = np.expm1(p_labels)

            id_col = next((c for c in test_raw.columns if c.lower() == 'id'), test_raw.columns[0])
            pd.DataFrame({id_col: test_raw[id_col], self.cfg.general.target_col: p_labels}).to_csv(
                self.cfg.inference.submission_name, index=False)
            self.logger.info(f"✅ Сабмит готов.")


if __name__ == "__main__":
    ProPipeline(get_config()).run()