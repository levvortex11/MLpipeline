import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TitanicDLDataset(Dataset):
    """
    Кастомный класс датасета для PyTorch.
    Служит для загрузки, хранения и пакетной выдачи данных в модель.
    Отделяет числовые признаки от категориальных для специфичной архитектуры сети.
    """
    def __init__(self, X_cont, X_cat, y=None):
        # Числовые признаки хранятся как Float тензоры (стандарт для вычислений весов)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        # Категориальные признаки хранятся как Long (целые числа), так как слои
        # nn.Embedding ожидают индексы категорий, а не вещественные числа.
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)

        # Если мы находимся в режиме обучения/валидации (таргет передан)
        if y is not None:
            # Преобразуем массив (N,) в матрицу (N, 1). Это критически важно,
            # так как выходной слой сети выдает размерность (N, 1), и функция
            # потерь BCEWithLogitsLoss требует совпадения размерностей.
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            # Режим инференса (предсказание на test.csv)
            self.y = None

    def __len__(self):
        """Возвращает общее количество строк в датасете."""
        return len(self.X_cont)

    def __getitem__(self, idx):
        """
        Возвращает один образец (строку) по индексу.
        DataLoader автоматически вызывает этот метод для сборки батчей.
        """
        if self.y is not None:
            return self.X_cont[idx], self.X_cat[idx], self.y[idx]
        return self.X_cont[idx], self.X_cat[idx]


class FlexibleTitanicNet(nn.Module):
    """
    Гибкая нейросетевая архитектура (MLP + Entity Embeddings).
    Слои генерируются динамически на основе параметров из конфигурации.
    """
    def __init__(self, cont_dim, cat_dims, cfg):
        """
        Инициализация архитектуры сети.
        :param cont_dim: Количество непрерывных (числовых) признаков
        :param cat_dims: Список, содержащий количество уникальных классов для каждой категориальной фичи
        :param cfg: Объект конфигурации (блок architecture)
        """
        super().__init__()
        self.cfg = cfg.architecture

        # --- Создание слоев Embedding для категорий ---
        # nn.ModuleList используется для того, чтобы PyTorch зарегистрировал веса
        # этих слоев во внутреннем графе вычислений.
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        for num_classes in cat_dims:
            # Эвристическое правило для размера вектора эмбеддинга (формула fast.ai)
            # Ограничиваем сверху значением max_embedding_dim, чтобы избежать перерасхода памяти.
            emb_dim = min(self.cfg.max_embedding_dim, (num_classes + 1) // 2)
            self.embeddings.append(nn.Embedding(
                num_embeddings=num_classes,
                embedding_dim=emb_dim,
                # padding_idx=0 означает, что индекс 0 отдает вектор из нулей (для неизвестных категорий)
                padding_idx=0
            ))
            total_emb_dim += emb_dim

        # --- Создание полносвязных слоев (MLP) ---
        self.layers = nn.ModuleList()
        # Входной размер для первого скрытого слоя = сумма размерностей чисел и выученных эмбеддингов
        current_dim = cont_dim + total_emb_dim

        # Словарь функций активации с использованием лямбда-функций.
        # Это гарантирует создание НОВОГО экземпляра функции активации для каждого слоя,
        # что необходимо для корректной работы графа PyTorch в сложных архитектурах.
        act_dict = {
            'relu': lambda: nn.ReLU(),
            'leaky_relu': lambda: nn.LeakyReLU(),
            'elu': lambda: nn.ELU(),
            'tanh': lambda: nn.Tanh()
        }
        activation_fn = act_dict.get(self.cfg.activation, lambda: nn.ReLU())

        # Динамическая генерация скрытых блоков: Linear -> [BatchNorm] -> Activation -> [Dropout]
        for hidden_dim in self.cfg.hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim))

            if self.cfg.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))

            self.layers.append(activation_fn())

            if self.cfg.dropout_p > 0:
                self.layers.append(nn.Dropout(p=self.cfg.dropout_p))

            current_dim = hidden_dim

        # --- Финальный слой предсказания ---
        # Возвращает один логит (сырое значение). Функция активации Sigmoid
        # будет применена на этапе расчета потерь или при валидации.
        self.output = nn.Linear(current_dim, 1)

    def forward(self, x_cont, x_cat):
        """
        Прямой проход данных через сеть.
        """
        # 1. Прогоняем каждый категориальный признак через его собственный слой Embedding
        x_emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]

        # 2. Склеиваем числовые признаки и векторы эмбеддингов по размерности фичей (dim=1)
        if len(x_emb) > 0:
            x = torch.cat([x_cont] + x_emb, dim=1)
        else:
            x = x_cont

        # 3. Пропускаем склеенный тензор через последовательность скрытых слоев
        for layer in self.layers:
            x = layer(x)

        # 4. Выдаем итоговое предсказание (логит)
        return self.output(x)