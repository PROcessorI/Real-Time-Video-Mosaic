"""
Модуль для реализации SLAM (Simultaneous Localization and Mapping).
Одновременная локализация и картографирование.

SLAM позволяет:
- Определять положение камеры в пространстве
- Строить карту окружающей среды
- Работать в реальном времени

Основные подходы:
1. Visual SLAM (vSLAM) - на основе камер
2. LiDAR SLAM - на основе лазерных дальномеров
3. Visual-Inertial SLAM - камера + IMU
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
import time
import os


class VisualOdometry:
    """
    Визуальная одометрия - оценка движения камеры по последовательности изображений.
    Это базовый компонент Visual SLAM.
    """
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        """
        Args:
            camera_matrix: матрица внутренних параметров камеры 3x3
        """
        # Параметры камеры по умолчанию
        if camera_matrix is None:
            # Примерные параметры для HD камеры
            self.camera_matrix = np.array([
                [700, 0, 640],
                [0, 700, 360],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            self.camera_matrix = camera_matrix
        
        # Детектор особенностей
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # Для оптического потока
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Параметры детектора FAST
        self.fast = cv2.FastFeatureDetector_create(threshold=25)
        
        # История
        self.prev_frame = None
        self.prev_keypoints = None
        self.trajectory = []  # Траектория камеры
        self.current_pose = np.eye(4)  # Текущая позиция (4x4 матрица)
        
    def detect_features(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обнаружение особенностей на изображении.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Используем ORB для обнаружения и описания
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        
        return points, descriptors
    
    def track_features_optical_flow(self, prev_frame: np.ndarray, 
                                     curr_frame: np.ndarray,
                                     prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Отслеживание особенностей с помощью оптического потока Lucas-Kanade.
        """
        # Преобразование в градации серого
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
        
        # Отслеживание точек
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self.lk_params
        )
        
        # Обратная проверка для фильтрации
        prev_points_check, status_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, curr_points, None, **self.lk_params
        )
        
        # Вычисление ошибки обратного потока
        diff = np.abs(prev_points - prev_points_check).reshape(-1, 2).max(axis=1)
        good_mask = (status.flatten() == 1) & (status_back.flatten() == 1) & (diff < 1.0)
        
        return curr_points, status, good_mask
    
    def estimate_motion(self, prev_points: np.ndarray, 
                        curr_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Оценка движения камеры по соответствиям точек.
        Использует Essential Matrix.
        """
        # Вычисление Essential Matrix
        E, mask = cv2.findEssentialMat(
            prev_points, curr_points,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        # Восстановление позы (R, t) из Essential Matrix
        _, R, t, mask_pose = cv2.recoverPose(
            E, prev_points, curr_points, self.camera_matrix, mask=mask
        )
        
        return R, t
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Обработка нового кадра и обновление позиции камеры.
        
        Returns:
            dict с информацией о движении
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        result = {
            'num_features': 0,
            'num_inliers': 0,
            'pose': self.current_pose.copy(),
            'translation': np.zeros(3),
            'rotation': np.eye(3)
        }
        
        if self.prev_frame is None:
            # Первый кадр
            self.prev_frame = gray
            self.prev_keypoints, _ = self.detect_features(frame)
            result['num_features'] = len(self.prev_keypoints)
            return result
        
        # Отслеживание точек
        curr_points, status, good_mask = self.track_features_optical_flow(
            self.prev_frame, gray, self.prev_keypoints
        )
        
        # Фильтрация хороших точек
        prev_good = self.prev_keypoints[good_mask]
        curr_good = curr_points[good_mask]
        
        result['num_features'] = len(curr_good)
        
        if len(prev_good) < 8:
            # Недостаточно точек, переинициализация
            self.prev_frame = gray
            self.prev_keypoints, _ = self.detect_features(frame)
            return result
        
        # Оценка движения
        R, t = self.estimate_motion(prev_good, curr_good)
        
        result['rotation'] = R
        result['translation'] = t.flatten()
        result['num_inliers'] = len(prev_good)
        
        # Обновление глобальной позы
        # T_new = T_old * T_relative
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t.flatten()
        
        self.current_pose = self.current_pose @ T_relative
        result['pose'] = self.current_pose.copy()
        
        # Сохранение траектории
        self.trajectory.append(self.current_pose[:3, 3].copy())
        
        # Обновление предыдущего кадра
        self.prev_frame = gray
        
        # Переобнаружение точек, если их мало
        if len(curr_good) < 500:
            self.prev_keypoints, _ = self.detect_features(frame)
        else:
            self.prev_keypoints = curr_good.reshape(-1, 1, 2)
        
        return result
    
    def get_trajectory(self) -> np.ndarray:
        """Получение траектории камеры."""
        if len(self.trajectory) == 0:
            return np.array([])
        return np.array(self.trajectory)
    
    def visualize_trajectory(self, frame: np.ndarray, scale: float = 10.0) -> np.ndarray:
        """
        Визуализация траектории на изображении.
        """
        vis = frame.copy()
        trajectory = self.get_trajectory()
        
        if len(trajectory) < 2:
            return vis
        
        # Проецирование 3D траектории на 2D
        center_x, center_y = vis.shape[1] // 2, vis.shape[0] // 2
        
        for i in range(1, len(trajectory)):
            pt1 = (int(center_x + trajectory[i-1][0] * scale),
                   int(center_y + trajectory[i-1][2] * scale))
            pt2 = (int(center_x + trajectory[i][0] * scale),
                   int(center_y + trajectory[i][2] * scale))
            
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
        
        # Текущая позиция
        curr_pos = (int(center_x + trajectory[-1][0] * scale),
                    int(center_y + trajectory[-1][2] * scale))
        cv2.circle(vis, curr_pos, 5, (0, 0, 255), -1)
        
        return vis


class SimpleSLAM:
    """
    Упрощённая реализация Visual SLAM.
    
    Компоненты:
    1. Visual Odometry - оценка движения
    2. Mapping - построение карты (облако точек)
    3. Loop Closure - обнаружение замкнутых петель (упрощённо)
    """
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        self.vo = VisualOdometry(camera_matrix)
        
        # Карта (облако точек)
        self.map_points = []  # 3D точки
        self.map_descriptors = []  # Дескрипторы для точек
        
        # Ключевые кадры
        self.keyframes = []
        self.keyframe_threshold = 0.5  # Минимальное перемещение для нового ключевого кадра
        self.last_keyframe_pose = np.eye(4)
        
        # Матчер для поиска соответствий
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Статистика
        self.stats = {
            'total_frames': 0,
            'keyframes': 0,
            'map_points': 0,
            'fps': 0
        }
        
        self.last_time = time.time()
    
    def triangulate_points(self, points1: np.ndarray, points2: np.ndarray,
                           pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """
        Триангуляция 3D точек из двух видов.
        """
        # Проекционные матрицы
        K = self.vo.camera_matrix
        
        P1 = K @ pose1[:3, :]
        P2 = K @ pose2[:3, :]
        
        # Триангуляция
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        
        # Преобразование в 3D координаты
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def should_create_keyframe(self, current_pose: np.ndarray) -> bool:
        """
        Определение, нужно ли создать новый ключевой кадр.
        """
        # Вычисление перемещения от последнего ключевого кадра
        translation = np.linalg.norm(
            current_pose[:3, 3] - self.last_keyframe_pose[:3, 3]
        )
        
        # Вычисление поворота
        R_diff = self.last_keyframe_pose[:3, :3].T @ current_pose[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        
        return translation > self.keyframe_threshold or angle > 0.3
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Обработка кадра для SLAM.
        """
        start_time = time.time()
        
        # Визуальная одометрия
        vo_result = self.vo.process_frame(frame)
        
        self.stats['total_frames'] += 1
        
        # Проверка на ключевой кадр
        if self.should_create_keyframe(vo_result['pose']):
            self._add_keyframe(frame, vo_result['pose'])
        
        # Обновление FPS
        elapsed = time.time() - start_time
        self.stats['fps'] = 1.0 / max(elapsed, 0.001)
        
        return {
            **vo_result,
            'stats': self.stats.copy(),
            'trajectory': self.vo.get_trajectory()
        }
    
    def _add_keyframe(self, frame: np.ndarray, pose: np.ndarray):
        """
        Добавление ключевого кадра.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        keypoints, descriptors = self.vo.orb.detectAndCompute(gray, None)
        
        self.keyframes.append({
            'pose': pose.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'frame_id': self.stats['total_frames']
        })
        
        self.last_keyframe_pose = pose.copy()
        self.stats['keyframes'] = len(self.keyframes)
    
    def get_map_visualization(self, size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """
        Визуализация карты сверху.
        """
        vis = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        trajectory = self.vo.get_trajectory()
        if len(trajectory) == 0:
            return vis
        
        # Масштабирование
        traj_xy = trajectory[:, [0, 2]]  # X и Z координаты
        
        if len(traj_xy) > 1:
            min_xy = traj_xy.min(axis=0)
            max_xy = traj_xy.max(axis=0)
            range_xy = max_xy - min_xy
            range_xy = np.maximum(range_xy, 0.1)  # Избегаем деления на 0
            
            scale = min(size[0] * 0.8, size[1] * 0.8) / max(range_xy)
            offset = np.array([size[0] // 2, size[1] // 2])
            
            # Рисование траектории
            for i in range(1, len(traj_xy)):
                pt1 = ((traj_xy[i-1] - min_xy - range_xy/2) * scale + offset).astype(int)
                pt2 = ((traj_xy[i] - min_xy - range_xy/2) * scale + offset).astype(int)
                cv2.line(vis, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
            
            # Рисование ключевых кадров
            for kf in self.keyframes:
                kf_pos = kf['pose'][:3, 3][[0, 2]]
                pt = ((kf_pos - min_xy - range_xy/2) * scale + offset).astype(int)
                cv2.circle(vis, tuple(pt), 5, (255, 0, 0), -1)
            
            # Текущая позиция
            curr = ((traj_xy[-1] - min_xy - range_xy/2) * scale + offset).astype(int)
            cv2.circle(vis, tuple(curr), 8, (0, 0, 255), -1)
        
        # Статистика
        cv2.putText(vis, f"Frames: {self.stats['total_frames']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Keyframes: {self.stats['keyframes']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"FPS: {self.stats['fps']:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis


# ============================================================
# ИНФОРМАЦИЯ О SLAM БИБЛИОТЕКАХ
# ============================================================

SLAM_LIBRARIES_INFO = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         SLAM БИБЛИОТЕКИ ДЛЯ PYTHON                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  1. ORB-SLAM2/3 (через pyORBSLAM)                                            ║
║     • Самый популярный Visual SLAM                                            ║
║     • Монокулярный, стерео и RGB-D режимы                                     ║
║     • Loop closure и relocalization                                           ║
║     • Требует компиляции C++ части                                            ║
║     • https://github.com/raulmur/ORB_SLAM2                                   ║
║                                                                               ║
║  2. OpenCV (встроенные функции)                                              ║
║     • cv2.SFM модуль для Structure from Motion                               ║
║     • Visual Odometry можно построить вручную                                ║
║     • Triangulation, Essential Matrix                                        ║
║                                                                               ║
║  3. Open3D SLAM                                                              ║
║     • RGB-D SLAM с интеграцией облаков точек                                 ║
║     • TSDF volume integration                                                ║
║     • ICP для регистрации                                                    ║
║     • pip install open3d                                                     ║
║                                                                               ║
║  4. RTAB-Map (через Python wrapper)                                          ║
║     • RGB-D SLAM                                                             ║
║     • Loop closure detection                                                 ║
║     • 3D map building                                                        ║
║     • http://introlab.github.io/rtabmap/                                     ║
║                                                                               ║
║  5. Kimera (MIT)                                                             ║
║     • Semantic SLAM                                                          ║
║     • Метрико-семантическая карта                                            ║
║     • https://github.com/MIT-SPARK/Kimera                                    ║
║                                                                               ║
║  6. GTSAM (для оптимизации графов)                                           ║
║     • Factor graphs для SLAM                                                 ║
║     • Bundle adjustment                                                      ║
║     • pip install gtsam                                                      ║
║                                                                               ║
║  7. g2o (graph optimization)                                                 ║
║     • Оптимизация графов для SLAM                                            ║
║     • pip install g2o                                                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║                            ТИПЫ SLAM                                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  VISUAL SLAM (vSLAM):                                                        ║
║  • Использует только камеры                                                   ║
║  • Монокулярный: 1 камера (масштаб не определяется)                          ║
║  • Стерео: 2 камеры (точная глубина)                                         ║
║  • RGB-D: камера глубины (Kinect, RealSense)                                 ║
║                                                                               ║
║  LIDAR SLAM:                                                                  ║
║  • Использует лазерный сканер                                                 ║
║  • Точные измерения расстояний                                               ║
║  • Google Cartographer, LOAM, LeGO-LOAM                                      ║
║                                                                               ║
║  VISUAL-INERTIAL SLAM:                                                        ║
║  • Камера + IMU (гироскоп + акселерометр)                                    ║
║  • VINS-Mono, ORB-SLAM3 VIO                                                  ║
║  • Более устойчив к быстрым движениям                                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def get_video_files(folder: str = "Data") -> list:
    """Получение списка видеофайлов из папки."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(folder, file))
    
    return video_files


def run_slam_on_video(video_path: str, save_trajectory: bool = True):
    """
    Запуск SLAM на видеофайле.
    
    Args:
        video_path: Путь к видеофайлу
        save_trajectory: Сохранять ли траекторию
    """
    print(f"\n{'='*60}")
    print(f"Обработка видео: {video_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return None
    
    # Получение информации о видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Разрешение: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"Всего кадров: {total_frames}")
    print(f"\nНажмите 'q' для выхода, 'p' для паузы, 's' для сохранения")
    print("-" * 60)
    
    # Создание SLAM с калибровкой под размер кадра
    camera_matrix = np.array([
        [width * 0.8, 0, width / 2],
        [0, width * 0.8, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    slam = SimpleSLAM(camera_matrix)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nВидео завершено!")
                break
            
            frame_count += 1
            
            # Обработка кадра
            result = slam.process_frame(frame)
            
            # Отображение информации на кадре
            info_frame = frame.copy()
            cv2.putText(info_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_frame, f"Features: {result['num_features']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_frame, f"FPS: {result['stats']['fps']:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_frame, f"Keyframes: {result['stats']['keyframes']}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Визуализация траектории на кадре
            traj_frame = slam.vo.visualize_trajectory(info_frame, scale=50.0)
            
            # Визуализация карты сверху
            map_vis = slam.get_map_visualization(size=(400, 400))
            
            # Объединение визуализаций
            cv2.imshow('SLAM - Video', traj_frame)
            cv2.imshow('SLAM - Map (Top View)', map_vis)
            
            # Прогресс
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Прогресс: {progress:.1f}% | Кадр {frame_count}/{total_frames} | "
                      f"Ключевых кадров: {result['stats']['keyframes']}")
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nВыход по запросу пользователя")
            break
        elif key == ord('p'):
            paused = not paused
            print("Пауза" if paused else "Продолжение")
        elif key == ord('s'):
            # Сохранение текущего состояния
            trajectory = slam.vo.get_trajectory()
            if len(trajectory) > 0:
                save_path = f"test_output/slam_trajectory_{frame_count}.npy"
                os.makedirs("test_output", exist_ok=True)
                np.save(save_path, trajectory)
                print(f"Траектория сохранена: {save_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Сохранение финальной траектории
    if save_trajectory:
        trajectory = slam.vo.get_trajectory()
        if len(trajectory) > 0:
            os.makedirs("test_output", exist_ok=True)
            
            # Сохранение как numpy
            np.save("test_output/slam_trajectory_final.npy", trajectory)
            
            # Сохранение как текст
            with open("test_output/slam_trajectory_final.txt", 'w') as f:
                f.write("# SLAM Trajectory\n")
                f.write(f"# Video: {video_path}\n")
                f.write(f"# Total frames: {frame_count}\n")
                f.write(f"# Keyframes: {slam.stats['keyframes']}\n")
                f.write("# X, Y, Z\n")
                for point in trajectory:
                    f.write(f"{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}\n")
            
            print(f"\nТраектория сохранена в test_output/")
            print(f"  - slam_trajectory_final.npy")
            print(f"  - slam_trajectory_final.txt")
    
    return slam


def run_slam_webcam():
    """
    Запуск SLAM на веб-камере в реальном времени.
    """
    print(f"\n{'='*60}")
    print("SLAM на веб-камере (реальное время)")
    print(f"{'='*60}")
    print("\nНажмите 'q' для выхода, 'r' для сброса, 's' для сохранения")
    print("-" * 60)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Веб-камера недоступна!")
        return None
    
    # Настройка камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Разрешение камеры: {width}x{height}")
    
    # Калибровка камеры (примерная)
    camera_matrix = np.array([
        [width * 0.8, 0, width / 2],
        [0, width * 0.8, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    slam = SimpleSLAM(camera_matrix)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            continue
        
        frame_count += 1
        
        # Обработка кадра
        result = slam.process_frame(frame)
        
        # Отображение информации
        info_frame = frame.copy()
        cv2.putText(info_frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_frame, f"Features: {result['num_features']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_frame, f"FPS: {result['stats']['fps']:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_frame, f"Keyframes: {result['stats']['keyframes']}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Визуализация траектории
        traj_frame = slam.vo.visualize_trajectory(info_frame, scale=100.0)
        
        # Визуализация карты
        map_vis = slam.get_map_visualization(size=(400, 400))
        
        cv2.imshow('SLAM - Webcam', traj_frame)
        cv2.imshow('SLAM - Map (Top View)', map_vis)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Сброс SLAM
            slam = SimpleSLAM(camera_matrix)
            frame_count = 0
            print("SLAM сброшен")
        elif key == ord('s'):
            # Сохранение траектории
            trajectory = slam.vo.get_trajectory()
            if len(trajectory) > 0:
                os.makedirs("test_output", exist_ok=True)
                np.save(f"test_output/slam_webcam_{frame_count}.npy", trajectory)
                print(f"Сохранено: test_output/slam_webcam_{frame_count}.npy")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return slam


def visualize_trajectory_3d(trajectory: np.ndarray):
    """
    3D визуализация траектории с помощью Open3D.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D не установлен. Установите: pip install open3d")
        return
    
    if len(trajectory) < 2:
        print("Недостаточно точек для визуализации")
        return
    
    print("\nСоздание 3D визуализации траектории...")
    
    # Создание линий траектории
    points = trajectory.tolist()
    lines = [[i, i+1] for i in range(len(points)-1)]
    colors = [[0, 1, 0] for _ in lines]  # Зелёный цвет
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Создание сферы для начальной точки
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    start_sphere.translate(trajectory[0])
    start_sphere.paint_uniform_color([0, 0, 1])  # Синий - начало
    
    # Создание сферы для конечной точки
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    end_sphere.translate(trajectory[-1])
    end_sphere.paint_uniform_color([1, 0, 0])  # Красный - конец
    
    # Создание системы координат
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    # Визуализация
    print("Синяя сфера - начало, Красная сфера - конец")
    print("Управление: ЛКМ - вращение, СКМ - перемещение, Колесо - зум")
    
    o3d.visualization.draw_geometries(
        [line_set, start_sphere, end_sphere, coord_frame],
        window_name="SLAM Trajectory 3D",
        width=1024,
        height=768
    )


def main_menu():
    """
    Главное меню SLAM.
    """
    while True:
        print("\n" + "=" * 60)
        print("           VISUAL SLAM - Главное меню")
        print("=" * 60)
        print("\n1. Запустить SLAM на видеофайле")
        print("2. Запустить SLAM на веб-камере")
        print("3. Загрузить и визуализировать траекторию (3D)")
        print("4. Информация о SLAM библиотеках")
        print("5. Выход")
        print("-" * 60)
        
        choice = input("Выберите опцию (1-5): ").strip()
        
        if choice == '1':
            # Выбор видеофайла
            video_files = get_video_files("Data")
            
            if not video_files:
                print("\nВидеофайлы не найдены в папке Data/")
                custom_path = input("Введите путь к видео (или Enter для отмены): ").strip()
                if custom_path and os.path.exists(custom_path):
                    run_slam_on_video(custom_path)
                continue
            
            print("\nДоступные видеофайлы:")
            for i, vf in enumerate(video_files, 1):
                print(f"  {i}. {os.path.basename(vf)}")
            print(f"  {len(video_files)+1}. Указать свой путь")
            
            try:
                vid_choice = int(input(f"\nВыберите видео (1-{len(video_files)+1}): "))
                if 1 <= vid_choice <= len(video_files):
                    run_slam_on_video(video_files[vid_choice - 1])
                elif vid_choice == len(video_files) + 1:
                    custom_path = input("Введите путь к видео: ").strip()
                    if os.path.exists(custom_path):
                        run_slam_on_video(custom_path)
                    else:
                        print("Файл не найден!")
            except ValueError:
                print("Некорректный ввод")
        
        elif choice == '2':
            run_slam_webcam()
        
        elif choice == '3':
            # Загрузка траектории
            traj_files = []
            if os.path.exists("test_output"):
                traj_files = [f for f in os.listdir("test_output") 
                              if f.endswith('.npy') and 'trajectory' in f]
            
            if not traj_files:
                print("\nСохранённые траектории не найдены")
                custom_path = input("Введите путь к .npy файлу (или Enter): ").strip()
                if custom_path and os.path.exists(custom_path):
                    trajectory = np.load(custom_path)
                    visualize_trajectory_3d(trajectory)
                continue
            
            print("\nДоступные траектории:")
            for i, tf in enumerate(traj_files, 1):
                print(f"  {i}. {tf}")
            
            try:
                traj_choice = int(input(f"\nВыберите траекторию (1-{len(traj_files)}): "))
                if 1 <= traj_choice <= len(traj_files):
                    traj_path = os.path.join("test_output", traj_files[traj_choice - 1])
                    trajectory = np.load(traj_path)
                    print(f"Загружено {len(trajectory)} точек траектории")
                    visualize_trajectory_3d(trajectory)
            except ValueError:
                print("Некорректный ввод")
        
        elif choice == '4':
            print(SLAM_LIBRARIES_INFO)
        
        elif choice == '5':
            print("\nВыход из программы")
            break
        
        else:
            print("Некорректный выбор, попробуйте снова")


if __name__ == "__main__":
    main_menu()
