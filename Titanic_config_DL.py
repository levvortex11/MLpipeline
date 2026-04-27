from omegaconf import OmegaConf


def get_dl_config():
    """
    Центральный пульт управления экспериментом Titanic DL (Deep Learning).
    Использование словаря и OmegaConf позволяет легко версионировать эксперименты
    и изменять гиперпараметры без необходимости жестко кодировать их в основном скрипте.
    """
    config = {
        'general': {
            # Название эксперимента, используемое для создания папок логов и артефактов
            'experiment_name': 'titanic_senior_final',
            # Глобальный seed для обеспечения полной воспроизводимости вычислений
            'seed': 42,
            # Относительный путь к обучающей выборке
            'data_path': 'Data/train.csv',
            # Колонка-идентификатор (не используется в обучении, нужна для сабмита)
            'id_col': 'PassengerId',
            # Наша целевая переменная (0 - погиб, 1 - выжил)
            'target_col': 'Survived',
            # Базовая директория для сохранения моделей, логов и препроцессоров
            'artifacts_path': 'outputs/dl_artifacts/',
            # Устройство для вычислений: 'cuda' для GPU, 'cpu' для процессора
            'device': 'cuda'
        },

        'inference': {
            # Путь к тестовому датасету (Kaggle hold-out)
            'test_data_path': 'Data/test.csv',
            # Путь и имя файла для сохранения финальных предсказаний
            'submission_path': 'outputs/submission.csv',
            # Флаг: выполнять ли предсказание на тесте после завершения кросс-валидации
            'run_inference': True
        },

        'logging': {
            # Дублировать ли логи из консоли в текстовый файл (dl_training.log)
            'log_to_file': True,
            # Уровень детализации логов (INFO, DEBUG, WARNING, ERROR)
            'log_level': 'INFO',
            # Как часто (в эпохах) выводить метрики обучения в консоль
            'verbose_step': 10
        },

        'features': {
            # Список числовых (непрерывных) признаков. Будут отмасштабированы (Scaler)
            'num_cols': ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch'],
            # Список категориальных признаков. Пойдут в слои nn.Embedding
            'cat_cols': ['Pclass', 'Sex', 'Embarked', 'IsAlone'],
            # Признаки, которые будут безусловно удалены из DataFrame перед обработкой
            'dropped_features': ['PassengerId', 'Name', 'Ticket', 'Cabin'],

            # --- Стратегии заполнения пропусков (Imputation) ---
            # Для чисел доступны: 'median', 'mean', 'tree_custom', 'min_mean_diff', 'max_mean_sum', 'pseudo_labeling'
            'num_impute_strategy': 'median',
            # Для категорий доступны: 'most_common', 'unknown', 'pseudo_labeling'
            'cat_impute_strategy': 'most_frequent',

            # Тип масштабирования чисел: 'standard' (Z-score), 'minmax' (0..1), 'robust', 'none'
            'scaler_type': 'standard'
        },

        'architecture': {
            # Конфигурация полносвязной сети (MLP). Количество и размер скрытых слоев
            'hidden_layers': [128, 64],
            # Функция активации: 'relu', 'leaky_relu', 'elu', 'tanh'
            'activation': 'leaky_relu',
            # Флаг использования слоев BatchNorm1d для стабилизации обучения
            'use_batchnorm': True,
            # Вероятность отключения нейронов в слоях Dropout (0.0 - отключить)
            'dropout_p': 0.3,
            # Ограничитель размерности вектора для Entity Embeddings
            'max_embedding_dim': 50
        },

        'training': {
            # Использовать ли StratifiedKFold (True) или обычный train_test_split (False)
            'use_cv': True,
            # Количество разбиений (фолдов) для кросс-валидации
            'cv_folds': 5,
            # Доля валидационной выборки, если use_cv = False
            'val_size': 0.2,
            # Максимальное количество эпох обучения для каждого фолда
            'epochs': 200,
            # Размер батча (количество примеров, обрабатываемых за один проход)
            'batch_size': 32,
            # Начальная скорость обучения (Learning Rate)
            'lr': 1e-3,
            # Оптимизатор: 'adam', 'adamw', 'sgd'
            'optimizer': 'adam',
            # Коэффициент L2-регуляризации (штраф за слишком большие веса)
            'weight_decay': 1e-5,

            # Количество параллельных потоков загрузки данных в DataLoader (0 = главный поток)
            'num_workers': 0,
            # Ускоряет перенос тензоров в GPU-память
            'pin_memory': True,

            # Список метрик для оценки на валидации
            'metrics': ['auc', 'accuracy', 'f1'],
            # Метрика, по которой работает Early Stopping и планировщик LR
            'monitor_metric': 'auc',
            # Режим мониторинга: 'max' (ищем максимум AUC/Accuracy) или 'min' (ищем минимум Loss)
            'monitor_mode': 'max',

            # Планировщик скорости обучения: 'plateau' (ReduceLROnPlateau) или 'cosine'
            'scheduler_type': 'plateau',
            # Во сколько раз уменьшать LR при срабатывании планировщика 'plateau'
            'factor': 0.5,
            # Сколько эпох без улучшений ждать перед снижением LR
            'patience': 7,
            # Сколько эпох без улучшений ждать перед полной остановкой обучения (Early Stopping)
            'early_stopping_patience': 15,
            # Минимальное изменение метрики, чтобы считать это реальным улучшением
            'min_delta': 1e-4
        }
    }
    return OmegaConf.create(config)