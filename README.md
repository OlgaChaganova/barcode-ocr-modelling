# OCR

Optical Character Recognition при помощи модели CRNN

Детекция производится с использованием модели **yolov5s**.

## 1 - Подготовка данных

1. Скачать [архив](https://disk.yandex.ru/d/nk-h0vv20EZvzg) с данными (gorai);


2. Распаковать его в папку `data`;

Отдельно проходить детектором для выделения баркодов и сохранения их в отдельную директорию не нужно, но такой скрипт существует. Можно выполнить:

```
make download_models_dvc
make prepare_data
```

3. Пример изображений в датасете: `notebooks/dataset.ipynb`

## 2 - Запуск обучения

1. Подготовить виртуальное окружение (Python >= 3.9):


```commandline
make install
clearml-init
```

3. Отредактировать конфиг для обучения в `src/configs/config.py`:


4. Выполнить (`python src/train.py --help` для вывода списка возможных аргументов):

```commandline
python src/train.py --log --logger <clearml / wandb>
```

или (с дефолтными параметрами)

```commandline
make train
```

Поддерживаются два логгера: ClearML и WandB.


## 3 - Эксперименты

-  TODO

## 4 - Инференс и веса моделей

1. Для модели написан wrapper (`src/ocr.py`). Пример, как запускать модель - `notebooks/inference.ipynb`;

2. Скачать веса модели (предварительно настроить доступ к удаленному серверу 91.206.15.25):

```commandline
make download_models_dvc
```

Веса OCR модели сковерчены в формат ONNX под CPU.

