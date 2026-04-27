from omegaconf import OmegaConf


def get_dl_config():
    """
    Центральный конфиг для Deep Learning пайплайна House Prices.
    Управляет архитектурой нейросети, тренировочным циклом и препроцессингом.
    Использование OmegaConf позволяет легко менять параметры эксперимента
    в одном месте без модификации основного кода логики.
    """
    config = {
        'general': {
            # Название эксперимента для формирования папок и логов
            'experiment_name': 'houseprices_dl_production_v1',
            # Глобальный seed для полной воспроизводимости вычислений
            'seed': 42,
            # Путь к тренировочным данным
            'data_path': 'Data/trainHP.csv',
            # Колонка-идентификатор (исключается из обучения)
            'id_col': 'Id',
            # Целевая переменная (то, что предсказывает нейросеть)
            'target_col': 'SalePrice',
            # ИСПРАВЛЕНИЕ: Добавлен рубильник логарифма для контроля из конфига.
            # Если True, целевая переменная логарифмируется (np.log1p), что крайне важно для цен.
            'log_target': True,
            # Папка для сохранения весов (model.pth) и скейлеров
            'artifacts_path': 'outputs/dl_artifacts/',
            # Устройство для вычислений ('cuda' для GPU, 'cpu' для процессора)
            'device': 'cuda'
        },

        'inference': {
            # Путь к тестовым данным (Kaggle hold-out)
            'test_data_path': 'Data/testHP.csv',
            # Путь для сохранения итогового сабмита
            'submission_path': 'outputs/submission_dl.csv',
            # Флаг запуска предсказаний после завершения обучения
            'run_inference': True
        },

        'logging': {
            # Флаг для дублирования логов из консоли в файл dl_training.log
            'log_to_file': True,
            # Уровень детализации логов
            'log_level': 'INFO',
            # Частота логирования метрик в консоль (каждые N эпох)
            'verbose_step': 10
        },

        'features': {
            # Непрерывные (числовые) фичи. Пройдут через импутер и скейлер напрямую в сеть.
            'num_cols': [
                'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
                'TotalSF'  # Сгенерированная фича (создается в FeatureEngineer)
            ],
            # Категориальные фичи. Пройдут через OrdinalEncoder и поступят в слои nn.Embedding
            'cat_cols': [
                'MSZoning', 'Street', 'LotShape', 'Neighborhood', 'BldgType',
                'HouseStyle', 'ExterQual', 'BsmtQual', 'HeatingQC', 'CentralAir',
                'KitchenQual', 'GarageType', 'SaleCondition'
            ],
            # Фичи, подлежащие безусловному удалению в начале обработки
            'dropped_features': ['Id'],

            # --- Стратегии препроцессинга DL-пайплайна ---
            # Варианты: 'median', 'mean', 'tree_custom', 'min_mean_diff', 'max_mean_sum', 'pseudo_labeling'
            'num_impute_strategy': 'median',
            'tree_const': 100,  # Константа для сдвига в стратегии tree_custom

            # Варианты: 'most_common', 'unknown', 'pseudo_labeling'
            'cat_impute_strategy': 'unknown',

            # Масштабирование: 'standard' (Z-score), 'minmax' (0..1), 'robust', 'none'
            # Нейросети очень чувствительны к масштабу, standard работает надежнее всего
            'scaler_type': 'standard'
        },

        'architecture': {
            # Динамическая генерация скрытых слоев полносвязной сети (MLP)
            # В данном случае 3 слоя: на 256, 128 и 64 нейрона
            'hidden_layers': [128, 64, 32, 16],
            # Функция активации между слоями: 'relu', 'leaky_relu', 'elu', 'tanh'
            'activation': 'leaky_relu',
            # Использование Batch Normalization для стабилизации обучения и градиентов
            'use_batchnorm': True,
            # Вероятность зануления нейронов (Dropout) для борьбы с переобучением (0.0 = отключить)
            'dropout_p': 0.2,
            # Ограничитель размера вектора Entity Embedding для одной категориальной фичи
            'max_embedding_dim': 50
        },

        'training': {
            # Если True — используем кросс-валидацию, если False — простой train_test_split
            'use_cv': True,
            # Количество разбиений для кросс-валидации
            'cv_folds': 5,
            # Доля валидационной выборки при use_cv = False
            'val_size': 0.2,
            # Количество полных проходов по обучающей выборке (эпохи)
            'epochs': 300,
            # Количество образцов в одном пакете (батче)
            'batch_size': 32,
            # Начальная скорость обучения (Learning Rate)
            'lr': 1e-3,
            # Оптимизатор: 'adamw' (Adam с правильным Weight Decay), 'adam', 'sgd'
            'optimizer': 'adamw',
            # L2-регуляризация (штраф за большие веса)
            'weight_decay': 1e-4,

            # Количество параллельных воркеров для загрузки данных в DataLoader (0 = главный поток)
            'num_workers': 0,
            # Флаг для более быстрого переноса тензоров из оперативной памяти в память GPU
            'pin_memory': True,

            # Список метрик для оценки модели на валидации
            'metrics': ['rmse', 'mae', 'r2'],
            # Целевая метрика, по которой работает Early Stopping и Scheduler
            'monitor_metric': 'rmse',
            # Ищем минимум для ошибки (rmse) или максимум (например, для r2)
            'monitor_mode': 'min',

            # Планировщик LR: 'plateau' (уменьшает LR на плато) или 'cosine'
            'scheduler_type': 'plateau',
            # Во сколько раз уменьшать LR при срабатывании планировщика
            'factor': 0.5,
            # Эпохи ожидания без улучшения для срабатывания планировщика
            'patience': 10,
            # Эпохи ожидания без улучшения для полной остановки обучения (Early Stopping)
            'early_stopping_patience': 20,
            # Минимальное изменение метрики, чтобы считать это реальным улучшением
            'min_delta': 1e-4
        }
    }
    return OmegaConf.create(config)