# MNIST classifier

Модели MLP и простой CNN для классификации рукописных цифр.

## Описание проекта

Данный проект решает задачу классификации рукописных цифр.
Модели обучаются на датасете MNIST и подходят для интеграции в веб-интерфейс или графическое приложение.

## Структура проекта

```
MNIST_classifier/
├── venv/                  # Корневая папка виртуального окружения
├── config/                # Конфигурационные файлы
│   └── pipeline_config.yaml
├── data/                  # Датасеты
│   └── MNIST/             # Датасет MNIST
│       ├── raw/           # Сырые данные
│       │   ├── t10k-images-idx3-ubyte
│       │   ├── t10k-images-idx3-ubyte.gz
│       │   ├── t10k-labels-idx1-ubyte
│       │   ├── t10k-labels-idx1-ubyte.gz
│       │   ├── train-images-idx3-ubyte
│       │   ├── train-images-idx3-ubyte.gz
│       │   ├── train-labels-idx1-ubyte
│       │   └── train-labels-idx1-ubyte.gz
├── inference/             # Изображения для инференса
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.jpg
│   ├── 8.jpg
│   └── 9.jpg
├── logs/                  # Файлы логов
│   └── mnist_pipeline.log.txt
├── models/                # Файлы state_dict обученных моделей
│   ├── mlp_final.pkl
│   └── cnn_final.pkl
├── static/                # Файлы веб приложения (CSS/JS)
│   ├── style.css
├── templates/             # HTML-шаблоны
│   └── index.html
├── utils/                 # Модули для построения пайплайна
│   ├── data_preprocessing.py
│   ├── logger.py
│   ├── model_training.py
│   └── models.py
├── app.py                 # Скрипт веб приложения
├── Dockerfile             # Файл конфигурации Docker
├── pipeline.py            # Скрипт пайплайна
├── README.md              # Документация
└── requirements.txt       # Файл с модулями для установки
```

## Установка зависимостей
В консоли IDE программы перейдите в корневую папку проекта и активируйте виртуальную среду, если она еще не была активирована.
Введите команду
      ```
      pip install -r requirements.txt
      ``` 
для установки модулей из файла requirements.txt

## Структура конфигурационных файлов модели
В проекте используется конфигурационный файл пайплайна, находящиеся в папке configs:

### 'pipeline_config.yaml'
Ниже будет представлено описание ключей и значений .yaml словаря в формате "ключ: тип данных (диапазон значений) - описание"
 - "used_architecture": str("mlp" or "cnn")  название используемой архитектуры модели
 - "epochs": int(> 0) - количество эпох обучения модели
 - "batch": int(> 0) - размер батча
 - "lr": float(> 0) - шаг обучения
 - "device": str("cuda" or "cpu") - устройство, на котором будет осуществляться обучение ("cuda", если доступно обучение на GPU, иначе "cpu")
 - "output_model_filename": str - имя файла state_dict обученной модели в формате name.pkl
 - "in_app_used_model_filename": str - имя файла модели, используемой в веб приложении

## Работа с проектом
Для запуска пайплайна перейдите в корневую папку с проектом и введите команду: 
      ``` 
      python run pipeline.py
      ``` 

После успешного завершения работы пайплайна обученная модель будет сохранена в папке models.

Проект контейнеризован с помощью Docker. Для запуска:

1. Находясь в папке репозитория (MNIST_classifier) соберите Docker образ:
   ```pwsh
   docker build -t mnist-web-app .
2. После сборки запустите контейнер:
   ```pwsh
   docker run -p 5000:5000 mnist-web-app
3. После запуска API будет доступно:
   http://localhost:5000