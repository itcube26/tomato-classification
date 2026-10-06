import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# 1. Загрузка модели
device = torch.device("cpu") # Для вебкамеры достаточно CPU

# Инициализируем архитектуру (должна совпадать с тем, что было при обучении)
model = models.mobilenet_v2()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 3) # 2 класса

# Загружаем веса
model.load_state_dict(torch.load('oves_model.pth', map_location=device))
model.eval() # Переводим в режим оценивания

# Классы (важно: они должны быть в алфавитном порядке, как их увидел ImageFolder)
class_names = ['bad', 'diseases', "good"] 

# 2. Преобразования для вебкамеры (должны совпадать с обучением, но без аугментации)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Запуск веб-камеры
cap = cv2.VideoCapture(0) # 0 - это обычно встроенная камера ноутбука

if not cap.isOpened():
    print("Ошибка: не удалось открыть веб-камеру!")
    exit()

print("Вебкамера запущена. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV читает в BGR, а PyTorch/PIL ждут RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Преобразуем в PIL Image и прогоняем через трансформации
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0) # Добавляем размерность batch

    # Предсказание
    with torch.no_grad():
        output = model(input_batch.to(device))
        
        # Получаем индекс класса с максимальной вероятностью
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        
        # Получаем уверенность (softmax)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence = probabilities[predicted_idx.item()].item() * 100

    # Рисуем результат на кадре
    color = (0, 255, 0) if predicted_class == 'good' else (0, 0, 255) # Зеленый для good, красный для bad
    text = f"{predicted_class.upper()} ({confidence:.1f}%)"
    
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Показываем видео
    cv2.imshow('Tomato Classifier (Press Q to quit)', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
