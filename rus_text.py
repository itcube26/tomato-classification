"""
Рисование русского текста на кадрах OpenCV (cv2.putText кириллицу не умеет).

Используется PIL + шрифт DejaVuSans (есть почти во всех Linux-дистрибутивах,
включая Zorin OS / Ubuntu). Если вдруг шрифта нет — скачаем его автоматически.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}

_CANDIDATE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_FALLBACK_URL = (
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/version_2_37/"
    "dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf"
)


def _find_font_file():
    for path in _CANDIDATE_FONTS:
        if os.path.exists(path):
            return path
    # Пробуем скачать в каталог рядом со скриптом
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "DejaVuSans-Bold.ttf")
    if not os.path.exists(local):
        try:
            import urllib.request
            print(f"Шрифт не найден, скачиваю: {_FALLBACK_URL}")
            urllib.request.urlretrieve(_FALLBACK_URL, local)
        except Exception as e:
            print(f"Не удалось скачать шрифт: {e}. "
                  f"Установите пакет fonts-dejavu-core.")
            return None
    return local


def get_font(size):
    if size not in _FONT_CACHE:
        path = _find_font_file()
        if path:
            _FONT_CACHE[size] = ImageFont.truetype(path, size)
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def put_russian_text(frame, text, org, font_size=48, color_bgr=(255, 255, 255),
                     bg=True):
    """Рисует русский текст на кадре (BGR numpy array) и возвращает кадр.

    org — координаты левого верхнего угла текста (x, y).
    color_bgr — цвет в формате (B, G, R).
    """
    pil_img = Image.fromarray(cv2_to_rgb(frame))
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size)

    x, y = org
    if bg:
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 6
        rgb_color = (color_bgr[2], color_bgr[1], color_bgr[0])
        # полупрозрачная чёрная подложка для читаемости сделаем просто тёмной
        draw.rectangle([bbox[0] - pad, bbox[1] - pad,
                        bbox[2] + pad, bbox[3] + pad], fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=rgb_color)
    else:
        rgb_color = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text((x, y), text, font=font, fill=rgb_color)

    return np.array(pil_img)[:, :, ::-1].copy()


def cv2_to_rgb(frame):
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
