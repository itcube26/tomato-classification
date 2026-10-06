import time

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw

from rus_text import put_russian_text, get_font

# фиктивная картинка только для замера ширины текста
_dummy_img = Image.new("RGB", (10, 10))

# ============================================================
# 1. Загрузка модели
# ============================================================
device = torch.device("cpu")  # Для вебкамеры достаточно CPU

# Инициализируем архитектуру (должна совпадать с тем, что было при обучении)
model = models.mobilenet_v2()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 3)  # 3 класса: bad, diseases, good

# Загружаем веса
model.load_state_dict(torch.load('oves_model.pth', map_location=device))
model.eval()  # Переводим в режим оценивания

# Классы (в алфавитном порядке, как их увидел ImageFolder)
class_names = ['bad', 'diseases', 'good']

# Какой класс считаем "овёс в кадре" и минимальная уверенность
OVES_CLASSES = {'good', 'diseases'}   # любая трава (здоровая или больная) — это овёс
MIN_CONFIDENCE = 0.5                  # ниже этой уверенности считаем, что овса нет

# Сколько секунд нужно продержать овёс в кадре подряд, чтобы началась "загрузка"
LOADING_SECONDS = 2.0
# Длительность самой анимации "ЗАГРУЗКА..." (секунды), после неё пишем "ОВЁС"
LOAD_ANIM_SECONDS = 2.0

# ============================================================
# 2. Преобразования для вебкамеры (как при обучении, но без аугментации)
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# 3. Настройки камеры
# ============================================================
# Если камера не откроется с 0, попробуйте 1 или 2
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Ошибка: не удалось открыть веб-камеру!")
    exit()

print("Вебкамера запущена. Нажмите 'q' для выхода.")
print("Логика:")
print(f"  - овёс в кадре >= {LOADING_SECONDS:.0f} сек -> 'ЗАГРУЗКА...', затем 'ОВЁС'")
print("  - овёс убрали из кадра -> белым 'ТРАВЫ НЕТ'")

# ============================================================
# 4. Состояние конечного автомата
#   state:
#     'no_herb'    — овса в кадре нет
#     'pending'    — овёс появился, но ещё меньше LOADING_SECONDS секунд
#     'loading'    — идёт анимация "загрузка"
#     'ready'      — загрузка закончена, показываем 'ОВЁС'
# ============================================================
state = 'no_herb'
herb_since = None      # время, с которого непрерывно видим овёс
loading_start = None   # время начала анимации загрузки


def classify(frame):
    """Возвращает (имя класса, уверенность в %)."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch.to(device))
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence = probabilities[predicted_idx.item()].item() * 100

    return predicted_class, confidence


while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    predicted_class, confidence = classify(frame)
    herb_present = (predicted_class in OVES_CLASSES and
                    confidence >= MIN_CONFIDENCE * 100)

    # --- переходы между состояниями ---
    if herb_present:
        if state == 'no_herb':
            state = 'pending'
            herb_since = now
        if state == 'pending' and now - herb_since >= LOADING_SECONDS:
            state = 'loading'
            loading_start = now
    else:
        # овса нет — сразу сбрасываем всё
        state = 'no_herb'
        herb_since = None
        loading_start = None

    # --- отрисовка надписи сверху ---
    h, w = frame.shape[:2]
    text = None
    color = (255, 255, 255)  # BGR, по умолчанию белый

    if state == 'no_herb':
        text = "ТРАВЫ НЕТ"
        color = (255, 255, 255)  # белый
    elif state == 'pending':
        # овёс только что появился, ещё не прошло 2 секунды — показываем загрузку
        text = "ЗАГРУЗКА..."
        color = (0, 255, 255)  # жёлтый (BGR)
    elif state == 'loading':
        progress = min((now - loading_start) / LOAD_ANIM_SECONDS, 1.0)
        dots = "." * (int(progress * 3) + 1)
        text = f"ЗАГРУЗКА{dots}"
        color = (0, 255, 255)  # жёлтый (BGR)
        if progress >= 1.0:
            state = 'ready'
    elif state == 'ready':
        text = "ОВЁС"
        color = (0, 255, 0)  # зелёный

    if text is not None:
        font_size = 48
        # рисуем по центру сверху (ширину текста меряем через PIL)
        font = get_font(font_size)
        bbox = ImageDraw.Draw(_dummy_img).textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x = max(10, (w - tw) // 2)
        frame = put_russian_text(frame, text, (x, 15),
                                 font_size=font_size, color_bgr=color)

    # маленькая служебная строка внизу (можно убрать)
    cv2.putText(frame, f"{predicted_class} {confidence:.0f}%", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow('Oves Classifier (Press Q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
