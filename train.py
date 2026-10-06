import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 1. Настройка преобразований для картинок
# Аугментация поможет модели лучше учиться на 2 картинках
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Загрузка данных
# ImageFolder автоматически поймет, что папки 'bad' и 'good' - это классы
dataset_dir = 'dataset-oves'
image_dataset = datasets.ImageFolder(dataset_dir, data_transforms)

# Batch size = 2, так как у нас всего 4 картинки
dataloader = DataLoader(image_dataset, batch_size=4, shuffle=True)

# Получаем имена классов (отсортированы по алфавиту: 0 - bad, 1 - good)
class_names = image_dataset.classes
print(f"Найдены классы: {class_names}")

# 3. Загрузка предобученной модели MobileNetV2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Замораживаем все слои, кроме последнего (чтобы учить только на наших помидорах)
for param in model.parameters():
    param.requires_grad = False

# Заменяем классификатор на наш (2 класса: bad и good)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# 4. Настройка обучения
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 5. Цикл обучения
num_epochs = 30 # Можно увеличить, если добавишь больше картинок

print("Начинаем обучение...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Эпоха [{epoch+1}/{num_epochs}], Ошибка (Loss): {running_loss/len(dataloader):.4f}")

# 6. Сохранение модели
torch.save(model.state_dict(), 'oves_model.pth')
print("Обучение завершено! Модель сохранена в 'oves_model.pth'")
