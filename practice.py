

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Модели из чеклиста
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# ==========================================
# 1. КОНФИГУРАЦИЯ (Всё управление здесь)
# ==========================================
@dataclass
class Config:
    seed: int = 42
    n_splits: int = 5
    target_col: str = 'target'

    # Списки колонок (обновляешь после EDA)
    num_cols: tuple = ('feature_1', 'feature_2')
    cat_cols: tuple = ('category_1',)
    drop_cols: tuple = ('id', 'constant_feature')  # Для скоррелированных фичей

    # Какие модели из чеклиста гоняем прямо сейчас
    active_models: tuple = ('ridge', 'rf', 'catboost')


# ==========================================
# 2. ПАЙПЛАЙН (Логика чеклиста)
# ==========================================
class KagglePipeline:
    def __init__(self, config: Config):
        self.cfg = config
        self.models_dict = self._init_models()

    def _init_models(self):
        """Пункт 4: Инициализация всех моделей из чеклиста"""
        return {
            'logistic': LogisticRegression(random_state=self.cfg.seed, max_iter=1000),
            'ridge': RidgeClassifier(random_state=self.cfg.seed),
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'tree': DecisionTreeClassifier(random_state=self.cfg.seed),
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.cfg.seed),
            'xgb': xgb.XGBClassifier(random_state=self.cfg.seed, eval_metric='logloss'),
            'lgb': lgb.LGBMClassifier(random_state=self.cfg.seed, verbose=-1),
            'catboost': CatBoostClassifier(random_state=self.cfg.seed, verbose=0, cat_features=list(self.cfg.cat_cols))
        }

    def run_eda(self, df: pd.DataFrame):
        """Пункт 1: Базовый EDA"""
        print("--- [EDA] Базовая статистика ---")
        print(df.describe())
        print(f"\nПропуски:\n{df.isna().sum()[df.isna().sum() > 0]}")
        # Здесь можно добавить plt.hist() для распределений

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Пункт 2: Предобработка"""
        df = df.copy()

        # a. Удаляем ненужное (константы, корреляции)
        df = df.drop(columns=list(self.cfg.drop_cols), errors='ignore')

        # b. Обработка пропусков (Упрощенно: числа - медиана, категории - 'missing')
        for col in self.cfg.num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if is_train else df[col].median())

        for col in self.cfg.cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('missing').astype(str)

        # (Опционально) Здесь добавляется StandardScaler, OneHotEncoding и т.д.
        # Для CatBoost можно оставить категории как есть.
        return df

    def train_validate(self, df: pd.DataFrame):
        """Пункт 3: Валидация (Stratified K-fold) и Пункт 4: Обучение"""
        X = self.preprocess(df.drop(columns=[self.cfg.target_col]))
        y = df[self.cfg.target_col]

        skf = StratifiedKFold(n_splits=self.cfg.n_splits, shuffle=True, random_state=self.cfg.seed)

        results = {}

        # Прогоняем только те модели, которые включены в конфиге
        for model_name in self.cfg.active_models:
            print(f"\n🚀 Обучение модели: {model_name}")
            model = self.models_dict[model_name]
            oof_preds = np.zeros(len(df))

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                # Для CatBoost и LGBM можно тут же передать eval_set для Early Stopping
                model.fit(X_tr, y_tr)

                # Сохраняем предсказания для OOF
                if hasattr(model, "predict_proba"):
                    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
                else:
                    oof_preds[val_idx] = model.predict(X_val)

            # Считаем итоговый скор для модели
            score = roc_auc_score(y, oof_preds)
            results[model_name] = score
            print(f"✅ {model_name} OOF ROC-AUC: {score:.4f}")

        return results


# ==========================================
# 3. ТОЧКА ВХОДА (main.py)
# ==========================================
if __name__ == "__main__":
    # Генерация синтетики для проверки кода
    dummy_data = pd.DataFrame({
        'id': range(100),
        'feature_1': np.random.randn(100),
        'feature_2': np.random.rand(100),
        'category_1': np.random.choice(['A', 'B', 'C'], 100),
        'constant_feature': [1] * 100,
        'target': np.random.choice([0, 1], 100)
    })

    # 1. Инициализация конфига
    cfg = Config()

    # 2. Создание пайплайна
    pipeline = KagglePipeline(cfg)

    # 3. Запуск по чеклисту
    pipeline.run_eda(dummy_data)
    results = pipeline.train_validate(dummy_data)