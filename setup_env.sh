#!/bin/bash

# Создание виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# Обновление pip
pip install --upgrade pip

# Установка библиотек
# Примечание: Если у тебя есть GPU NVIDIA, PyTorch может потребовать специфичную версию CUDA.
# Эта команда устанавливает стандартную стабильную версию.
pip install -r requirements_full.txt

echo "------------------------------------------------"
echo "✅ Полное ML/DL окружение успешно установлено!"
echo "Активируй его командой: source .venv/bin/activate"