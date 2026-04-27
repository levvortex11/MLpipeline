import numpy as np
import matplotlib.pyplot as plt
import warnings

# Импортируем все необходимые инструменты sklearn
from sklearn import metrics, linear_model, ensemble, model_selection, neighbors, tree
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Бустинг-библиотеки
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


def get_base_model_instance(name, cfg, params=None):
    """
    Фабрика создания базовых моделей по имени из конфига.
    Эта функция возвращает "сырой" инстанс алгоритма с заданными параметрами.
    """
    p = params if params else dict(cfg.model.params)
    seed = cfg.general.seed

    # ИСПРАВЛЕНИЕ: Early stopping отключаем во время тюнинга, а также
    # для CatBoost внутри пайплайна (так как Pipeline не передает eval_set, что вызывает ошибку).
    es = p.get('early_stopping_rounds') if not cfg.tuning.enabled else None

    # Словарь со всеми поддерживаемыми алгоритмами
    model_map = {
        'catboost': CatBoostClassifier(
            iterations=p.get('iterations', 100), depth=p.get('depth', 3),
            verbose=False, random_state=seed, l2_leaf_reg=5
        ),
        'lgbm': LGBMClassifier(
            n_estimators=p.get('iterations', 100), max_depth=p.get('depth', 3),
            verbosity=-1, random_state=seed
        ),
        'xgb': XGBClassifier(
            n_estimators=p.get('iterations', 100), max_depth=p.get('depth', 3),
            random_state=seed
        ),
        'knn': neighbors.KNeighborsClassifier(n_neighbors=p.get('n_neighbors', 5)),
        'rf': ensemble.RandomForestClassifier(n_estimators=1000, random_state=seed),
        'dt': tree.DecisionTreeClassifier(max_depth=5, random_state=seed),
        'lasso': linear_model.LogisticRegression(penalty='l1', solver='liblinear', random_state=seed),
        'ridge': linear_model.RidgeClassifier(random_state=seed),
        'elasticnet': linear_model.LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                                      random_state=seed),
        'logistic': linear_model.LogisticRegression(C=p.get('C', 1.0), random_state=seed)
    }

    base_model = model_map.get(name, model_map['logistic'])

    # Важный фикс: если модель не умеет выводить вероятности (predict_proba), например Ridge,
    # мы оборачиваем её в CalibratedClassifierCV. Это необходимо для метрики ROC-AUC и мягких ансамблей.
    if not hasattr(base_model, "predict_proba"):
        base_model = CalibratedClassifierCV(base_model, cv=3, method='sigmoid')
    return base_model


def get_model(cfg, scoring_string='roc_auc'):
    """
    Сборка итоговой модели: либо одиночной, либо стекинга, с опциональным тюнингом параметров.
    """
    seed = cfg.general.seed
    n_jobs = cfg.tuning.n_jobs

    if cfg.general.use_stacking:
        # Собираем список базовых оценщиков (первый уровень)
        estimators = [(m, get_base_model_instance(m, cfg)) for m in cfg.stacking.base_models]
        # Создаем мета-модель (второй уровень)
        meta = get_base_model_instance(cfg.stacking.meta_model, cfg, params=cfg.stacking.meta_params)

        # Собираем StackingClassifier (внутри он сам разобьет данные на cv_folds для обучения базовых моделей)
        base_obj = StackingClassifier(estimators=estimators, final_estimator=meta, cv=cfg.stacking.cv_folds)

        # Оборачиваем в RandomizedSearchCV, если тюнинг включен
        if cfg.tuning.enabled and cfg.stacking.stack_tuning_mode != 'none':
            grid = {
                'final_estimator__C': [0.01, 0.1, 1.0, 10.0]} if cfg.stacking.stack_tuning_mode == 'meta_only' else {}
            return model_selection.RandomizedSearchCV(base_obj, grid, n_iter=cfg.tuning.n_iter, cv=cfg.tuning.cv,
                                                      scoring=scoring_string, random_state=seed, n_jobs=n_jobs)
        return base_obj
    else:
        # Собираем одиночную модель
        base_obj = get_base_model_instance(cfg.model.name, cfg)
        if cfg.tuning.enabled:
            # Демонстрационная сетка параметров (для реальных задач расширяется)
            grid = {'max_depth': [3, 4, 5, 8, 10]} if hasattr(base_obj, 'max_depth') else {}
            return model_selection.RandomizedSearchCV(base_obj, grid, n_iter=cfg.tuning.n_iter, cv=cfg.tuning.cv,
                                                      scoring=scoring_string, random_state=seed, n_jobs=n_jobs)
        return base_obj


def plot_importance(model, features):
    """
    Визуализация важности признаков.
    Функция умеет "докапываться" до реальной модели сквозь обертки (Search / Pipeline).
    """
    m = model
    # Извлекаем лучшую модель, если был тюнинг
    if hasattr(m, 'best_estimator_'): m = m.best_estimator_
    # Извлекаем модель из пайплайна
    if hasattr(m, 'named_steps'): m = m.named_steps['model']

    if hasattr(m, 'feature_importances_'):
        imp = m.feature_importances_
        # Берем только топ-20, чтобы график не превратился в кашу после OneHot-кодирования
        idx = np.argsort(imp)[-20:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(idx)), imp[idx], align='center', color='steelblue')
        plt.yticks(range(len(idx)), [features[i] for i in idx])
        plt.title("Importance of Features (Top 20)")
        plt.xlabel("Score")
        plt.tight_layout()
        plt.show()


def plot_learning_curve(estimator, X, y, cfg, scoring_string='roc_auc'):
    """
    Рисует график кривой обучения (Learning Curve).
    Показывает, страдает ли модель от переобучения (разрыв между Training и Validation)
    или от недообучения (обе линии низко).
    """
    # ИСПРАВЛЕНИЕ: Берем количество фолдов из general (или 5 по умолчанию), а не из блока тюнинга
    cv_folds = cfg.general.get('cv_folds', 5)
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv_folds, n_jobs=cfg.tuning.n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring=scoring_string
    )
    # sklearn возвращает отрицательные значения для некоторых метрик (MSE/MAE), инвертируем для красоты
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