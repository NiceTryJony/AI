import os
import subprocess
from pathlib import Path
from datetime import datetime
import torch

# Путь для сохранения временных изображений
SAVE_PATH = Path("captured_images")
SAVE_PATH.mkdir(exist_ok=True)

# Параметры камеры
DEVICE = "/dev/video3"  # Укажи правильное устройство камеры
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def capture_image():
    """Захват изображения с помощью v4l2-ctl."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = SAVE_PATH / f"image_{timestamp}.jpg"
    
    # Команда для захвата изображения
    command = [
        "v4l2-ctl",
        f"--device={DEVICE}",
        f"--stream-mmap=3",
        f"--stream-count=1",
        f"--stream-to={image_path}"
    ]
    
    try:
        subprocess.run(command, check=True)
        return image_path
    except subprocess.CalledProcessError as e:
        print(f"Ошибка захвата изображения: {e}")
        return None

def analyze_image(image_path):
    """Анализ изображения с использованием YOLOv5."""
    results = model(image_path)
    detections = results.pandas().xyxy[0]  # Получение данных в формате pandas DataFrame
    
    if detections.empty:
        print("Объекты не обнаружены.")
    else:
        print("Обнаруженные объекты:")
        print(detections[['name', 'confidence']])  # Имя объекта и уровень уверенности
        
        # Сохранение изображения с результатами
        results.save()
        print(f"Результаты сохранены в: {results.files}")

def main():
    print("Программа начата. Нажмите Ctrl+C для выхода.")
    
    try:
        while True:
            input("Нажмите Enter для захвата изображения...")
            image_path = capture_image()
            
            if image_path and image_path.exists():
                print(f"Изображение захвачено: {image_path}")
                analyze_image(image_path)
            else:
                print("Не удалось захватить изображение.")
    except KeyboardInterrupt:
        print("\nПрограмма завершена.")

if __name__ == "__main__":
    main()
