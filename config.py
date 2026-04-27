from omegaconf import OmegaConf

def get_config():
    """
    Центральный пульт управления экспериментом Titanic ML.
    Все параметры из этого словаря управляют логикой в main_pipeline.py и factory.py.
    """
    config = {
        'general': {
            # Название эксперимента (используется для логов)
            'experiment_name': 'titanic_target_v5',
            # Фиксатор случайности для воспроизводимости результатов
            'seed': 42,
            # Название целевой колонки, которую мы предсказываем
            'target_col': 'Survived',
            # Основная метрика оптимизации. Варианты: 'roc_auc', 'accuracy', 'f1'
            'metric': 'roc_auc',
            # Относительный или абсолютный путь к обучающему датасету
            'data_path': 'Data/train.csv',
            # Использовать ли ансамбль Stacking (True) или одиночную модель (False)
            'use_stacking': False,
            # ИСПРАВЛЕНИЕ: Переключатель режима валидации: True = кросс-валидация, False = одиночный Hold-Out
            'use_cv': True,
            # ИСПРАВЛЕНИЕ: Количество фолдов для основной валидации (используется, если use_cv=True)
            'cv_folds': 5,
            # Включение режима генерации файла submission.csv для тестовой выборки
            'run_inference': True,
            # Папка, куда сохраняются веса моделей (joblib) и логи
            'artifacts_path': 'outputs/artifacts/'
        },
        'features': {
            # Список колонок, которые принудительно конвертируются в строки (категории)
            'force_categorical': ['Pclass'],
            # Колонки, удаляемые на самом раннем этапе (не несут пользы или вызывают утечки)
            'dropped_features': ['PassengerId', 'Name', 'Ticket', 'Cabin'],
            # Порог очистки от мультиколлинеарности. Если две фичи коррелируют > 0.95, одна удаляется
            'corr_threshold': 0.95
        },
        'imputation': {
            # Стратегии заполнения категориальных пропусков (из схемы):
            # 'most_common' - заполнение самым частым значением (мода / most lazy)
            # 'unknown'     - заполнение словом "Unknown" (если пустые значения имеют логику)
            # 'pseudo_labeling' - продвинутый подход (Feature Pseudo Labeling)
            'cat_strategy': 'unknown',

            # Стратегии заполнения числовых пропусков (из схемы):
            # 'mean' / 'median' - классика (most lazy)
            # 'tree_custom'     - [MIN] - Const (отлично для деревьев)
            # 'min_mean_diff'   - [MIN] - [AVG] (для линейных и других моделей)
            # 'max_mean_sum'    - [MAX] + [AVG] (аналогично)
            # 'pseudo_labeling' - итеративное предсказание пропусков через ML (IterativeImputer)
            'num_strategy': 'tree_custom',

            # Константа, которая вычитается из минимума в стратегии 'tree_custom'
            'tree_const': 100
        },
        'scaling': {
            # Тип нормализации числовых данных:
            # 'standard' - StandardScaler (z-score)
            # 'minmax'   - MinMaxScaler (от 0 до 1)
            # 'robust'   - RobustScaler (устойчив к выбросам, использует квантили)
            # 'none'     - без масштабирования (хорошо для деревьев)
            'type': 'standard'
        },
        'encoding': {
            # Тип кодирования категориальных данных:
            # 'target'  - TargetEncoder (кодирует средним значением таргета, отлично против переобучения)
            # 'onehot'  - OneHotEncoder (классические дамми-переменные 0/1)
            # 'ordinal' - OrdinalEncoder (простая нумерация 0, 1, 2...)
            'type': 'target'
        },
        'stacking': {
            # Список базовых моделей для первого слоя ансамбля
            'base_models': ['rf', 'lgbm', 'catboost', 'xgb', 'knn', 'elasticnet'],
            # Мета-модель, которая учится на предсказаниях базовых моделей (второй слой)
            'meta_model': 'logistic',
            # Параметры для мета-модели
            'meta_params': {'C': 0.1},
            # Количество фолдов при обучении стекинга (чтобы избежать утечки)
            'cv_folds': 5,
            # Режим тюнинга в стекинге: 'meta_only' (тюним только мета-модель), 'none', 'all'
            'stack_tuning_mode': 'meta_only'
        },
        'tuning': {
            # Флаг включения автоматического подбора гиперпараметров (RandomizedSearchCV)
            'enabled': False,
            # Количество итераций (комбинаций параметров) для проверки
            'n_iter': 30,
            # Количество фолдов кросс-валидации во время тюнинга
            'cv': 5,
            # Количество используемых ядер процессора (-1 = все доступные)
            'n_jobs': -1
        },
        'model': {
            # Название одиночной модели, которая обучается, если use_stacking = False
            'name': 'xgb',
            # Статические параметры для выбранной модели
            'params': {
                'iterations': 1000,
                'learning_rate': 0.01,
                'l2_leaf_reg': 10,
                'depth': 3,
                'early_stopping_rounds': 50,
                'verbose': False
            }
        },
        'plots': {
            # Отрисовка кривых обучения (Learning Curve) для диагностики переобучения
            'show_learning_curve': True,
            # Отрисовка столбчатой диаграммы важности признаков
            'show_feature_importance': True,
            # Отрисовка изменения метрик по эпохам (задел на будущее)
            'show_metrics_curves': False
        },
        'inference': {
            # Путь к тестовому датасету (на котором нет колонки Survived)
            'test_data_path': 'Data/test.csv',
            # Имя итогового файла для Kaggle / бизнеса
            'submission_name': 'submission.csv',
            # Формат вывода: 'labels' (0 или 1) либо 'probs' (вероятности от 0.0 до 1.0)
            'output_type': 'labels'
        }
    }
    return OmegaConf.create(config)