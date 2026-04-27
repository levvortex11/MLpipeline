from omegaconf import OmegaConf

def get_config():
    """
    Центральный пульт управления экспериментом House Prices (Регрессия).
    """
    config = {
        'general': {
            'experiment_name': 'hp_full_production_v1',
            'seed': 42,
            'target_col': 'SalePrice',
            # Флаг логарифмирования таргета. MUST HAVE для цен на недвижимость!
            # Снижает влияние выбросов (очень дорогих домов) и делает RMSE адекватным (~0.13)
            'log_target': True,
            'metric': 'rmse',  # Доступно: 'rmse', 'mae', 'r2'
            'data_path': 'Data/trainHP.csv',
            'use_stacking': False,
            'use_cv': True, # Использовать кросс-валидацию вместо Hold-Out
            'cv_folds': 5,  # Количество фолдов для основной валидации
            'run_inference': True,
            'artifacts_path': 'outputs/artifacts/'
        },
        'features': {
            # Принудительно делаем эти числовые колонки категориальными (т.к. это просто коды)
            'force_categorical': ['MSSubClass', 'OverallQual', 'OverallCond'],
            'dropped_features': ['Id'],
            # Жесткость удаления мультиколлинеарных признаков
            'corr_threshold': 0.98
        },
        'imputation': {
            # Варианты: 'most_common', 'unknown', 'pseudo_labeling'
            'cat_strategy': 'unknown',

            # Варианты: 'median', 'mean', 'tree_custom', 'min_mean_diff', 'max_mean_sum', 'pseudo_labeling'
            'num_strategy': 'median',
            'tree_const': 100
        },
        'scaling': {
            'type': 'standard'  # Варианты: 'standard', 'minmax', 'robust', 'none'
        },
        'encoding': {
            'type': 'target'  # Варианты: 'target' (кодирует средним таргетом), 'onehot', 'ordinal'
        },
        'stacking': {
            'base_models': ['rf', 'lgbm', 'catboost', 'xgb', 'ridge'],
            'meta_model': 'ridge',
            'meta_params': {'alpha': 1.0},
            'cv_folds': 5, # Фолды специально для сборки мета-признаков стекинга
            'stack_tuning_mode': 'meta_only'
        },
        'tuning': {
            'enabled': False,
            'n_iter': 30,
            'cv': 5, # Фолды для RandomizedSearchCV
            'n_jobs': -1
        },
        'model': {
            'name': 'xgb',
            'params': {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 5,
                'early_stopping_rounds': 50,
                'verbose': False
            }
        },
        'plots': {
            'show_learning_curve': True,
            'show_feature_importance': True
        },
        'inference': {
            'test_data_path': 'Data/testHP.csv',
            'submission_name': 'submissionHP.csv'
        }
    }
    return OmegaConf.create(config)