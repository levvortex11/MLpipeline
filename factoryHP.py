import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn import metrics, linear_model, ensemble, model_selection, neighbors, tree
from sklearn.ensemble import StackingRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


def get_base_model_instance(name, cfg, params=None):
    """Фабрика базовых моделей для регрессии."""
    p = params if params else dict(cfg.model.params)
    seed = cfg.general.seed

    # Early stopping в sklearn Pipeline без eval_set вызывает проблемы у CatBoost/XGB.
    # Поэтому мы игнорируем его на уровне конструктора, если работаем в пайплайне.
    # Для LGBM и XGB он не ломает код так агрессивно, но CatBoost требует eval_set.

    model_map = {
        'catboost': CatBoostRegressor(
            iterations=p.get('iterations', 1000), depth=p.get('depth', 6),
            verbose=False, random_state=seed, l2_leaf_reg=p.get('l2_leaf_reg', 5)
        ),
        'lgbm': LGBMRegressor(
            n_estimators=p.get('iterations', 1000), max_depth=p.get('depth', 6),
            verbosity=-1, random_state=seed
        ),
        'xgb': XGBRegressor(
            n_estimators=p.get('iterations', 1000), max_depth=p.get('depth', 6),
            random_state=seed
        ),
        'knn': neighbors.KNeighborsRegressor(n_neighbors=p.get('n_neighbors', 5)),
        'rf': ensemble.RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1),
        'dt': tree.DecisionTreeRegressor(max_depth=5, random_state=seed),
        'lasso': linear_model.Lasso(alpha=p.get('alpha', 1.0), random_state=seed),
        'ridge': linear_model.Ridge(alpha=p.get('alpha', 1.0), random_state=seed),
        'elasticnet': linear_model.ElasticNet(alpha=p.get('alpha', 1.0), l1_ratio=0.5, random_state=seed)
    }
    return model_map.get(name, model_map['ridge'])


def get_model(cfg, scoring_string='neg_root_mean_squared_error'):
    """Сборка регрессионной модели (со стекингом и тюнингом)."""
    seed = cfg.general.seed
    n_jobs = cfg.tuning.n_jobs

    if cfg.general.use_stacking:
        estimators = [(m, get_base_model_instance(m, cfg)) for m in cfg.stacking.base_models]
        meta = get_base_model_instance(cfg.stacking.meta_model, cfg, params=cfg.stacking.meta_params)
        base_obj = StackingRegressor(estimators=estimators, final_estimator=meta, cv=cfg.stacking.cv_folds)

        if cfg.tuning.enabled and cfg.stacking.stack_tuning_mode != 'none':
            grid = {'final_estimator__alpha': [0.1, 1.0, 10.0]} if cfg.stacking.stack_tuning_mode == 'meta_only' else {}
            return model_selection.RandomizedSearchCV(base_obj, grid, n_iter=cfg.tuning.n_iter, cv=cfg.tuning.cv,
                                                      scoring=scoring_string, random_state=seed, n_jobs=n_jobs)
        return base_obj
    else:
        base_obj = get_base_model_instance(cfg.model.name, cfg)
        if cfg.tuning.enabled:
            grid = {'max_depth': [3, 4, 5, 8, 10]} if hasattr(base_obj, 'max_depth') else {}
            return model_selection.RandomizedSearchCV(base_obj, grid, n_iter=cfg.tuning.n_iter, cv=cfg.tuning.cv,
                                                      scoring=scoring_string, random_state=seed, n_jobs=n_jobs)
        return base_obj


def plot_importance(model, features):
    """Отрисовка важности признаков (Топ-20)."""
    m = model
    if hasattr(m, 'best_estimator_'): m = m.best_estimator_
    if hasattr(m, 'named_steps'): m = m.named_steps['model']

    if hasattr(m, 'feature_importances_'):
        imp = m.feature_importances_
        idx = np.argsort(imp)[-20:]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(idx)), imp[idx], align='center', color='steelblue')
        plt.yticks(range(len(idx)), [features[i] for i in idx])
        plt.title("Importance of Features (Top 20)")
        plt.tight_layout()
        plt.show()


def plot_learning_curve(estimator, X, y, cfg, scoring_string='neg_root_mean_squared_error'):
    """График обучения."""
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cfg.tuning.cv, n_jobs=cfg.tuning.n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring=scoring_string
    )
    # Инвертируем отрицательные метрики sklearn
    mult = -1 if 'neg_' in scoring_string else 1

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, mult * np.mean(train_scores, axis=1), 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, mult * np.mean(test_scores, axis=1), 'o-', color="g", label="Validation Score")
    plt.title(f"Model Learning Curve ({cfg.general.metric})")
    plt.xlabel("Samples")
    plt.ylabel(cfg.general.metric)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()