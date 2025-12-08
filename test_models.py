"""
Тестирование моделей детекции:
1. YOLO-World (yolov8x-worldv2.pt) - для зданий по текстовому описанию
2. YOLO11l-OBB (yolo11l-obb.pt) - для аэроснимков с повёрнутыми рамками (DOTA)
3. YOLOv8l-VisDrone (yolov8l-visdrone.pt) - для дронов (транспорт, люди)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os


def test_yolo_world(image_path, output_path="test_output/yolo_world_result.jpg"):
    """Тест YOLO-World для детекции зданий и других объектов по текстовому описанию."""
    print("\n" + "="*60)
    print("🌍 YOLO-World (Open-Vocabulary Detection)")
    print("="*60)
    
    model = YOLO('yolov8x-worldv2.pt')
    
    # Устанавливаем классы для детекции
    classes = [
        'building', 'house', 'roof', 'car', 'truck', 'bus', 
        'person', 'tree', 'road', 'pool', 'boat'
    ]
    model.set_classes(classes)
    print(f"Классы для детекции: {classes}")
    
    # Детекция
    results = model.predict(
        image_path,
        conf=0.1,
        imgsz=1280,
        save=True,
        project="test_output",
        name="yolo_world"
    )
    
    print(f"\nНайдено объектов: {len(results[0].boxes)}")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        print(f"  - {class_name}: {conf:.2%}")
    
    return results


def test_yolo11_obb(image_path, output_path="test_output/yolo11_obb_result.jpg"):
    """Тест YOLO11-OBB для детекции объектов с повёрнутыми рамками (DOTA)."""
    print("\n" + "="*60)
    print("📐 YOLO11l-OBB (Oriented Bounding Boxes - DOTA)")
    print("="*60)
    
    model = YOLO('yolo11l-obb.pt')
    
    # Классы DOTA
    dota_classes = {
        0: 'plane', 1: 'ship', 2: 'storage tank', 3: 'baseball diamond',
        4: 'tennis court', 5: 'basketball court', 6: 'ground track field',
        7: 'harbor', 8: 'bridge', 9: 'large vehicle', 10: 'small vehicle',
        11: 'helicopter', 12: 'roundabout', 13: 'soccer ball field', 
        14: 'swimming pool'
    }
    print(f"Классы DOTA: {list(dota_classes.values())}")
    
    # Детекция
    results = model.predict(
        image_path,
        conf=0.25,
        imgsz=1024,
        save=True,
        project="test_output",
        name="yolo11_obb"
    )
    
    print(f"\nНайдено объектов: {len(results[0].obb) if hasattr(results[0], 'obb') and results[0].obb is not None else 0}")
    
    if hasattr(results[0], 'obb') and results[0].obb is not None:
        for i, (cls, conf) in enumerate(zip(results[0].obb.cls, results[0].obb.conf)):
            cls_id = int(cls)
            confidence = float(conf)
            class_name = dota_classes.get(cls_id, f"class_{cls_id}")
            print(f"  - {class_name}: {confidence:.2%}")
    
    return results


def test_yolov8_visdrone(image_path, output_path="test_output/visdrone_result.jpg"):
    """Тест YOLOv8-VisDrone для детекции объектов с дрона."""
    print("\n" + "="*60)
    print("🚁 YOLOv8l-VisDrone (Drone Detection)")
    print("="*60)
    
    model = YOLO('yolov8l-visdrone.pt')
    
    # Классы VisDrone
    visdrone_classes = {
        0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
        5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'
    }
    print(f"Классы VisDrone: {list(visdrone_classes.values())}")
    
    # Детекция
    results = model.predict(
        image_path,
        conf=0.25,
        imgsz=640,
        save=True,
        project="test_output",
        name="visdrone"
    )
    
    print(f"\nНайдено объектов: {len(results[0].boxes)}")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = visdrone_classes.get(cls_id, f"class_{cls_id}")
        print(f"  - {class_name}: {conf:.2%}")
    
    return results


def test_all_models(image_path):
    """Запуск тестов всех моделей на одном изображении."""
    print("\n" + "🔬"*30)
    print("       ТЕСТИРОВАНИЕ МОДЕЛЕЙ ДЕТЕКЦИИ")
    print("🔬"*30)
    print(f"\nИзображение: {image_path}")
    
    os.makedirs("test_output", exist_ok=True)
    
    # Тест всех моделей
    try:
        test_yolo_world(image_path)
    except Exception as e:
        print(f"❌ Ошибка YOLO-World: {e}")
    
    try:
        test_yolo11_obb(image_path)
    except Exception as e:
        print(f"❌ Ошибка YOLO11-OBB: {e}")
    
    try:
        test_yolov8_visdrone(image_path)
    except Exception as e:
        print(f"❌ Ошибка YOLOv8-VisDrone: {e}")
    
    print("\n" + "="*60)
    print("✅ Результаты сохранены в папке test_output/")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Тестовое изображение по умолчанию
        # Можно использовать любое аэрофото
        test_images = [
            "mosaic.jpg"
        ]
        
        # Проверяем наличие локальных файлов
        image_path = None
        for img in test_images[:-1]:
            if os.path.exists(img):
                image_path = img
                break
        
        if image_path is None:
            # Используем онлайн изображение для теста
            image_path = test_images[-1]
            print(f"Используем тестовое изображение: {image_path}")
    
    test_all_models(image_path)
