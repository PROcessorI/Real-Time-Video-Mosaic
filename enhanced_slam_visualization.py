#!/usr/bin/env python3
"""
Улучшенная визуализация SLAM траектории.
Запуск: python enhanced_slam_visualization.py
"""

import cv2
import numpy as np
import os
import sys
import time

# Добавление текущей директории в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from slam import SLAMVisualizer, SimpleSLAM


def main():
    """
    Демонстрация улучшенной визуализации SLAM.
    """
    print("=" * 60)
    print("УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ SLAM")
    print("=" * 60)

    # Проверка наличия траектории
    traj_path = "test_output/slam_trajectory_final.npy"
    if not os.path.exists(traj_path):
        print("Ошибка: Файл траектории не найден!")
        print("Запустите SLAM на видео сначала (python slam.py -> опция 1)")
        return

    # Загрузка траектории
    trajectory = np.load(traj_path)
    print(f"✓ Загружено {len(trajectory)} точек траектории")

    # Создание SLAM системы для визуализации
    slam = SimpleSLAM()
    visualizer = SLAMVisualizer()

    # Имитация ключевых кадров (каждые 10 точек)
    for i in range(0, len(trajectory), 10):
        pose = np.eye(4)
        pose[:3, 3] = trajectory[i]
        kf = {
            'pose': pose,
            'keypoints': [],
            'descriptors': np.array([]),
            'frame_id': i
        }
        slam.keyframes.append(kf)
        slam.stats['keyframes'] = len(slam.keyframes)

    slam.stats['total_frames'] = len(trajectory)
    slam.vo.trajectory = trajectory.tolist()

    print("✓ Создана SLAM система для визуализации")
    print("✓ Инициализирован визуализатор с множественными видами")

    # Создание демо кадра для визуализации
    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    demo_frame.fill(100)  # Серый фон

    print("\n" + "=" * 60)
    print("КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ SLAM")
    print("=" * 60)
    print("Окна визуализации:")
    print("  • Enhanced SLAM Visualization Dashboard - комплексная панель")
    print("  • SLAM - Video - видео с наложенной траекторией")
    print("  • SLAM - Map (Top View) - карта сверху")
    print("  • SLAM - Multi-View Dashboard - множественные виды")
    print("\nУправление:")
    print("  'q' - выход")
    print("  's' - сохранить скриншот")
    print("  't' - показать 3D траекторию (Open3D)")
    print("  'r' - создать отчёт")
    print("-" * 60)

    frame_count = 0
    last_save_time = 0

    # Основной цикл визуализации
    while True:
        frame_count += 1

        # Создание комплексной визуализации
        multi_view = visualizer.create_multi_view_visualization(slam, demo_frame, size=(1600, 900))

        # Дополнительные визуализации
        traj_frame = slam.vo.visualize_trajectory(demo_frame, scale=50.0)
        map_vis = slam.get_map_visualization(size=(400, 400))

        # Отображение окон
        cv2.imshow('Enhanced SLAM Visualization Dashboard', multi_view)
        cv2.imshow('SLAM - Video', traj_frame)
        cv2.imshow('SLAM - Map (Top View)', map_vis)

        # Обработка клавиш
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            print("\nВыход из визуализации")
            break

        elif key == ord('s'):
            # Сохранение скриншота с временной меткой
            timestamp = int(time.time())
            screenshot_path = f"test_output/slam_visualization_{timestamp}.png"
            cv2.imwrite(screenshot_path, multi_view)
            print(f"✓ Скриншот сохранён: {screenshot_path}")

        elif key == ord('t'):
            # 3D визуализация траектории
            print("Запуск 3D визуализации...")
            cv2.destroyAllWindows()

            try:
                from slam import visualize_trajectory_3d
                visualize_trajectory_3d(trajectory)
            except Exception as e:
                print(f"Ошибка 3D визуализации: {e}")
                print("Установите Open3D: pip install open3d")

            # Перезапуск окон после 3D визуализации
            continue

        elif key == ord('r'):
            # Создание и отображение отчёта
            report = visualizer.create_slam_report(slam, trajectory)
            print("\n" + "="*50)
            print("ОТЧЁТ SLAM СИСТЕМЫ")
            print("="*50)
            for key, value in report.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
            print("="*50)

        # Автосохранение каждые 30 секунд
        current_time = time.time()
        if current_time - last_save_time > 30:
            auto_save_path = f"test_output/slam_auto_{int(current_time)}.png"
            cv2.imwrite(auto_save_path, multi_view)
            last_save_time = current_time
            print(f"✓ Автосохранение: {auto_save_path}")

    cv2.destroyAllWindows()

    # Финальный отчёт
    report = visualizer.create_slam_report(slam, trajectory)
    print("\nФинальный отчёт:")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ Визуализация завершена")


if __name__ == "__main__":
    main()