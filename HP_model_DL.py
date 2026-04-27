import torch
import torch.nn as nn
from torch.utils.data import Dataset

class HousePricesDLDataset(Dataset):
    """
    Класс датасета для загрузки данных в DataLoader.
    Специфика в том, что он разделяет непрерывные и категориальные признаки
    для раздельной обработки в различных ветвях нейросети.
    """
    def __init__(self, X_cont, X_cat, y=None):
        # Непрерывные фичи конвертируем в float32 (стандартный тип для весов сети)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        # Категории передаются как LongTensor (int64), так как слой nn.Embedding
        # воспринимает на вход только целочисленные индексы словаря.
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)

        if y is not None:
            # Формируем 2D тензор размерности (N, 1) для таргета,
            # чтобы он корректно считался с выходом финального слоя nn.Linear(..., 1)
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            self.y = None

    def __len__(self):
        """Возвращает общий размер датасета."""
        return len(self.X_cont)

    def __getitem__(self, idx):
        """Возвращает один образец по индексу для сборки батчей."""
        if self.y is not None:
            return self.X_cont[idx], self.X_cat[idx], self.y[idx]
        return self.X_cont[idx], self.X_cat[idx]


class FlexibleHousePricesNet(nn.Module):
    """
    Универсальная полносвязная нейросеть (MLP) с поддержкой Entity Embeddings
    для категориальных переменных. Архитектура (количество слоев, активации)
    собирается динамически из конфигурационного файла.
    """
    def __init__(self, cont_dim, cat_dims, cfg):
        """
        :param cont_dim: количество числовых фичей
        :param cat_dims: список, содержащий размер словаря для каждой категориальной фичи
        :param cfg: блок architecture из конфигурации
        """
        super().__init__()
        self.cfg = cfg.architecture

        # ModuleList хранит слои так, чтобы PyTorch регистрировал их веса
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        # Динамическое создание слоев nn.Embedding для каждой категориальной колонки
        for num_classes in cat_dims:
            # Эвристика fast.ai: размер вектора эмбеддинга не больше 50 или половины классов
            emb_dim = min(self.cfg.max_embedding_dim, (num_classes + 1) // 2)
            self.embeddings.append(nn.Embedding(
                num_embeddings=num_classes,
                embedding_dim=emb_dim,
                # padding_idx=0 зарезервирован под unknown классы (-1 сдвигается в 0 после OrdinalEncoder)
                padding_idx=0
            ))
            total_emb_dim += emb_dim

        # Создаем список скрытых слоев
        self.layers = nn.ModuleList()
        # Входной размер = размер числовых фичей + размер всех склеенных эмбеддингов
        current_dim = cont_dim + total_emb_dim

        # Словарь функций активации. Лямбды гарантируют, что каждый слой получит свой,
        # независимый экземпляр активации (это важно для сложных графов).
        act_dict = {
            'relu': lambda: nn.ReLU(),
            'leaky_relu': lambda: nn.LeakyReLU(),
            'elu': lambda: nn.ELU(),
            'tanh': lambda: nn.Tanh()
        }
        activation_fn = act_dict.get(self.cfg.activation, lambda: nn.ReLU())

        # Динамическая сборка скрытых блоков (Linear -> BatchNorm -> Activation -> Dropout)
        for hidden_dim in self.cfg.hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim))

            if self.cfg.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))

            self.layers.append(activation_fn())

            if self.cfg.dropout_p > 0:
                self.layers.append(nn.Dropout(p=self.cfg.dropout_p))

            current_dim = hidden_dim

        # Финальный слой предсказания (1 выходной нейрон для регрессии - предсказывает цену/лог-цену)
        self.output = nn.Linear(current_dim, 1)

    def forward(self, x_cont, x_cat):
        """Прямой проход (вычисление предсказания)."""
        # 1. Пропускаем каждую категориальную колонку через свой эмбеддинг-слой
        x_emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]

        # 2. Склеиваем непрерывные фичи с полученными векторами эмбеддингов (dim=1 - по столбцам)
        if len(x_emb) > 0:
            x = torch.cat([x_cont] + x_emb, dim=1)
        else:
            x = x_cont

        # 3. Прогон склеенного вектора через все полносвязные слои
        for layer in self.layers:
            x = layer(x)

        # 4. Возвращаем результат
        return self.output(x)