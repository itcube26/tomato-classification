# camera-pomidor

## Сайт с обученными моделями

https://universe.roboflow.com

## Клонирование репозитория (в етрминале)

git clone https://github.com/itcube26/tomato-classification

cd tomato-classification/


## Создание Python окружения

sudo apt install python3.10-venv

python3 -m venv venv

## Активация окружения

cd tomato-classification/

source venv/bin/activate


ДОЛЖНО БЫТЬ ТАК:
(venv) comp01@comp01-DPA156:~/tomato-classification$ 


## Установка зависимостей на Zorin OS

pip install opencv-python numpy paho-mqtt requests



....



## Запуск обучения

python3 train.py

## Запуск тестирования

python3 webcam.py
