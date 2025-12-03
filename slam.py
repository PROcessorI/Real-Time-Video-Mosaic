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


# ============================================================
# ПРОДВИНУТЫЙ АНАЛИЗАТОР ЗЕМЛИ И ПОЧВЫ
# ============================================================

class TerrainSoilAnalyzer:
    """
    Продвинутый анализатор земли и почвы по изображению.
    
    Функции:
    - Классификация типа почвы по цвету и текстуре
    - Оценка влажности почвы
    - Определение плодородности
    - Анализ структуры поверхности
    - Определение типа растительного покрова
    - Выявление эрозии и дефектов
    """
    
    # Справочник типов почв с характеристиками (HSV диапазоны)
    SOIL_TYPES = {
        'chernozem': {
            'name': 'Чернозём',
            'name_en': 'Chernozem (Black Soil)',
            'hsv_range': [(0, 0, 0), (180, 100, 80)],
            'color_desc': 'Тёмно-коричневый до чёрного',
            'fertility': 'Очень высокая',
            'fertility_score': 95,
            'organic_matter': '6-15%',
            'ph_range': '6.5-7.5',
            'water_retention': 'Высокая',
            'suitable_crops': ['пшеница', 'кукуруза', 'подсолнечник', 'сахарная свёкла'],
            'regions': 'Украина, Юг России, Казахстан'
        },
        'podzol': {
            'name': 'Подзолистая',
            'name_en': 'Podzol',
            'hsv_range': [(10, 20, 100), (30, 80, 200)],
            'color_desc': 'Светло-серый',
            'fertility': 'Низкая',
            'fertility_score': 30,
            'organic_matter': '1-4%',
            'ph_range': '4.5-5.5',
            'water_retention': 'Низкая',
            'suitable_crops': ['картофель', 'рожь', 'овёс', 'лён'],
            'regions': 'Северная Россия, Скандинавия, Канада'
        },
        'clay': {
            'name': 'Глинистая',
            'name_en': 'Clay Soil',
            'hsv_range': [(5, 50, 80), (25, 200, 180)],
            'color_desc': 'Красно-коричневый, жёлто-коричневый',
            'fertility': 'Средняя',
            'fertility_score': 55,
            'organic_matter': '2-5%',
            'ph_range': '5.5-7.0',
            'water_retention': 'Очень высокая (плохой дренаж)',
            'suitable_crops': ['рис', 'капуста', 'брокколи'],
            'regions': 'Повсеместно'
        },
        'sandy': {
            'name': 'Песчаная',
            'name_en': 'Sandy Soil',
            'hsv_range': [(15, 30, 150), (35, 120, 255)],
            'color_desc': 'Светло-жёлтый, бежевый',
            'fertility': 'Низкая',
            'fertility_score': 25,
            'organic_matter': '0.5-2%',
            'ph_range': '5.5-7.0',
            'water_retention': 'Очень низкая',
            'suitable_crops': ['морковь', 'картофель', 'арбузы', 'дыни'],
            'regions': 'Пустыни, прибрежные зоны'
        },
        'loam': {
            'name': 'Суглинок',
            'name_en': 'Loam Soil',
            'hsv_range': [(8, 40, 80), (25, 150, 160)],
            'color_desc': 'Коричневый',
            'fertility': 'Высокая',
            'fertility_score': 80,
            'organic_matter': '3-6%',
            'ph_range': '6.0-7.0',
            'water_retention': 'Хорошая (сбалансированная)',
            'suitable_crops': ['томаты', 'перец', 'зерновые', 'бобовые'],
            'regions': 'Умеренный климат повсеместно'
        },
        'red_soil': {
            'name': 'Красная почва (Латерит)',
            'name_en': 'Red Soil (Laterite)',
            'hsv_range': [(0, 100, 80), (15, 255, 200)],
            'color_desc': 'Красный, красно-оранжевый',
            'fertility': 'Средняя (требует удобрений)',
            'fertility_score': 45,
            'organic_matter': '1-3%',
            'ph_range': '5.0-6.5',
            'water_retention': 'Средняя',
            'suitable_crops': ['хлопок', 'арахис', 'табак', 'цитрусовые'],
            'regions': 'Тропики, Индия, Африка, Бразилия'
        },
        'peaty': {
            'name': 'Торфяная',
            'name_en': 'Peaty Soil',
            'hsv_range': [(0, 30, 20), (30, 100, 70)],
            'color_desc': 'Тёмно-коричневый до чёрного (волокнистый)',
            'fertility': 'Высокая (после мелиорации)',
            'fertility_score': 70,
            'organic_matter': '20-80%',
            'ph_range': '3.5-5.5',
            'water_retention': 'Очень высокая (заболоченность)',
            'suitable_crops': ['клюква', 'голубика', 'овощи (после осушения)'],
            'regions': 'Болота, Северная Европа, Канада'
        },
        'calcarite': {
            'name': 'Известковая (Карбонатная)',
            'name_en': 'Calcareous Soil',
            'hsv_range': [(20, 10, 180), (40, 60, 255)],
            'color_desc': 'Светлый, белёсый, серо-белый',
            'fertility': 'Средняя',
            'fertility_score': 50,
            'organic_matter': '1-4%',
            'ph_range': '7.5-8.5',
            'water_retention': 'Средняя',
            'suitable_crops': ['виноград', 'оливки', 'лаванда', 'зерновые'],
            'regions': 'Средиземноморье, степи'
        }
    }
    
    # Типы растительного покрова
    VEGETATION_TYPES = {
        'dense_grass': {'name': 'Густая трава', 'green_ratio': (0.6, 1.0), 'health': 'Отлично'},
        'sparse_grass': {'name': 'Редкая трава', 'green_ratio': (0.3, 0.6), 'health': 'Хорошо'},
        'dry_grass': {'name': 'Сухая трава', 'green_ratio': (0.1, 0.3), 'health': 'Плохо'},
        'bare_soil': {'name': 'Голая почва', 'green_ratio': (0.0, 0.1), 'health': 'Нет растительности'},
        'forest': {'name': 'Лесной покров', 'green_ratio': (0.7, 1.0), 'health': 'Отлично'},
        'shrubs': {'name': 'Кустарники', 'green_ratio': (0.4, 0.7), 'health': 'Хорошо'}
    }
    
    def __init__(self):
        """Инициализация анализатора."""
        self.analysis_results = {}
        
    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        Полный анализ изображения земли/почвы.
        
        Args:
            image: BGR изображение
            
        Returns:
            Словарь с результатами анализа
        """
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'soil_analysis': self._analyze_soil(image),
            'moisture_analysis': self._analyze_moisture(image),
            'vegetation_analysis': self._analyze_vegetation(image),
            'texture_analysis': self._analyze_texture(image),
            'erosion_analysis': self._analyze_erosion(image),
            'recommendations': []
        }
        
        # Генерация рекомендаций
        results['recommendations'] = self._generate_recommendations(results)
        
        self.analysis_results = results
        return results
    
    def _analyze_soil(self, image: np.ndarray) -> Dict:
        """Анализ типа почвы по цвету."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Средние значения HSV
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # Средние значения BGR
        b_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        r_mean = np.mean(image[:, :, 2])
        
        # Определение типа почвы по цветовым характеристикам
        soil_scores = {}
        
        for soil_type, properties in self.SOIL_TYPES.items():
            score = 0
            
            # Проверка яркости (V)
            if soil_type == 'chernozem' and v_mean < 80:
                score += 40
            elif soil_type == 'sandy' and v_mean > 150:
                score += 40
            elif soil_type == 'calcarite' and v_mean > 180:
                score += 35
            elif soil_type == 'podzol' and 100 < v_mean < 180 and s_mean < 60:
                score += 35
            elif soil_type == 'red_soil' and r_mean > g_mean * 1.3 and r_mean > b_mean * 1.5:
                score += 45
            elif soil_type == 'clay' and 80 < v_mean < 160 and r_mean > b_mean:
                score += 30
            elif soil_type == 'loam' and 80 < v_mean < 150:
                score += 25
            elif soil_type == 'peaty' and v_mean < 70 and s_mean < 80:
                score += 35
            
            # Проверка насыщенности
            if soil_type in ['red_soil', 'clay'] and s_mean > 80:
                score += 20
            elif soil_type in ['podzol', 'calcarite'] and s_mean < 50:
                score += 20
            elif soil_type == 'chernozem' and s_mean < 100:
                score += 15
            
            # Проверка оттенка
            if soil_type == 'red_soil' and h_mean < 15:
                score += 20
            elif soil_type in ['sandy', 'loam', 'clay'] and 10 < h_mean < 30:
                score += 15
            
            soil_scores[soil_type] = score
        
        # Определение наиболее вероятного типа
        best_soil = max(soil_scores, key=soil_scores.get)
        confidence = min(100, soil_scores[best_soil])
        
        soil_info = self.SOIL_TYPES[best_soil]
        
        return {
            'type': best_soil,
            'name': soil_info['name'],
            'name_en': soil_info['name_en'],
            'confidence': confidence,
            'color_description': soil_info['color_desc'],
            'fertility': soil_info['fertility'],
            'fertility_score': soil_info['fertility_score'],
            'organic_matter': soil_info['organic_matter'],
            'ph_range': soil_info['ph_range'],
            'water_retention': soil_info['water_retention'],
            'suitable_crops': soil_info['suitable_crops'],
            'typical_regions': soil_info['regions'],
            'color_stats': {
                'hsv_mean': [float(h_mean), float(s_mean), float(v_mean)],
                'rgb_mean': [float(r_mean), float(g_mean), float(b_mean)]
            },
            'all_scores': soil_scores
        }
    
    def _analyze_moisture(self, image: np.ndarray) -> Dict:
        """Анализ влажности почвы по цвету."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Влажная почва обычно темнее
        v_mean = np.mean(hsv[:, :, 2])
        s_mean = np.mean(hsv[:, :, 1])
        
        # Вычисление индекса влажности (чем темнее, тем влажнее)
        # Также влажная почва менее насыщенная по цвету
        darkness_factor = max(0, (100 - v_mean / 2.55)) / 100
        saturation_factor = max(0, 1 - s_mean / 255 * 0.3)
        
        moisture_index = darkness_factor * 0.7 + saturation_factor * 0.3
        moisture_percent = min(100, moisture_index * 100)
        
        # Классификация уровня влажности
        if moisture_percent > 70:
            level = 'Очень высокая (переувлажнение)'
            status = 'warning'
            drainage_needed = True
        elif moisture_percent > 50:
            level = 'Высокая'
            status = 'good'
            drainage_needed = False
        elif moisture_percent > 30:
            level = 'Умеренная (оптимально)'
            status = 'optimal'
            drainage_needed = False
        elif moisture_percent > 15:
            level = 'Низкая'
            status = 'warning'
            drainage_needed = False
        else:
            level = 'Очень низкая (засуха)'
            status = 'critical'
            drainage_needed = False
        
        return {
            'moisture_index': round(moisture_percent, 1),
            'level': level,
            'status': status,
            'drainage_needed': drainage_needed,
            'irrigation_recommendation': 'Требуется полив' if moisture_percent < 30 else 
                                         'Полив не требуется' if moisture_percent < 70 else 
                                         'Требуется дренаж'
        }
    
    def _analyze_vegetation(self, image: np.ndarray) -> Dict:
        """Анализ растительного покрова."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Маска зелёного цвета (растительность)
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Маска жёлтого/коричневого (сухая растительность)
        dry_lower = np.array([15, 40, 80])
        dry_upper = np.array([35, 200, 200])
        dry_mask = cv2.inRange(hsv, dry_lower, dry_upper)
        
        total_pixels = image.shape[0] * image.shape[1]
        green_pixels = np.sum(green_mask > 0)
        dry_pixels = np.sum(dry_mask > 0)
        
        green_ratio = green_pixels / total_pixels
        dry_ratio = dry_pixels / total_pixels
        bare_ratio = 1 - green_ratio - dry_ratio
        
        # Определение типа покрова
        if green_ratio > 0.7:
            cover_type = 'Густая зелёная растительность'
            health = 'Отлично'
            ndvi_estimate = 0.7 + green_ratio * 0.3
        elif green_ratio > 0.4:
            cover_type = 'Умеренная растительность'
            health = 'Хорошо'
            ndvi_estimate = 0.4 + green_ratio * 0.5
        elif green_ratio > 0.2:
            cover_type = 'Редкая растительность'
            health = 'Удовлетворительно'
            ndvi_estimate = 0.2 + green_ratio * 0.5
        elif dry_ratio > 0.3:
            cover_type = 'Сухая/увядающая растительность'
            health = 'Плохо'
            ndvi_estimate = 0.1 + dry_ratio * 0.2
        else:
            cover_type = 'Преимущественно голая почва'
            health = 'Нет растительности'
            ndvi_estimate = -0.1 + green_ratio
        
        return {
            'cover_type': cover_type,
            'health_status': health,
            'green_cover_percent': round(green_ratio * 100, 1),
            'dry_vegetation_percent': round(dry_ratio * 100, 1),
            'bare_soil_percent': round(max(0, bare_ratio) * 100, 1),
            'ndvi_estimate': round(ndvi_estimate, 2),
            'photosynthesis_activity': 'Высокая' if green_ratio > 0.5 else 
                                       'Средняя' if green_ratio > 0.2 else 'Низкая'
        }
    
    def _analyze_texture(self, image: np.ndarray) -> Dict:
        """Анализ текстуры поверхности почвы."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Вычисление градиентов (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Лапласиан для детектирования резкости
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # GLCM-подобные метрики (упрощённо)
        # Контраст
        contrast = np.std(gray)
        
        # Однородность (обратно пропорциональна дисперсии градиента)
        homogeneity = 1 / (1 + np.std(gradient_magnitude) / 100)
        
        # Шероховатость
        roughness = np.mean(gradient_magnitude)
        
        # Классификация текстуры
        if roughness > 50:
            texture_type = 'Очень грубая (комковатая)'
            particle_size = 'Крупные комки'
        elif roughness > 30:
            texture_type = 'Грубая'
            particle_size = 'Крупнозернистая'
        elif roughness > 15:
            texture_type = 'Средняя'
            particle_size = 'Среднезернистая'
        elif roughness > 8:
            texture_type = 'Мелкая'
            particle_size = 'Мелкозернистая'
        else:
            texture_type = 'Очень мелкая (гладкая)'
            particle_size = 'Пылеватая/илистая'
        
        return {
            'texture_type': texture_type,
            'particle_size': particle_size,
            'roughness_index': round(roughness, 2),
            'contrast': round(contrast, 2),
            'homogeneity': round(homogeneity, 3),
            'sharpness': round(laplacian_var, 2),
            'compaction_estimate': 'Высокая' if roughness < 10 else 
                                   'Средняя' if roughness < 25 else 'Низкая (рыхлая)'
        }
    
    def _analyze_erosion(self, image: np.ndarray) -> Dict:
        """Анализ признаков эрозии."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Детектирование линий (потенциальные борозды эрозии)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        num_lines = len(lines) if lines is not None else 0
        
        # Анализ вариации яркости (пятнистость = возможные вымытые участки)
        v_std = np.std(hsv[:, :, 2])
        
        # Определение уровня эрозии
        erosion_indicators = 0
        erosion_types = []
        
        if num_lines > 50:
            erosion_indicators += 30
            erosion_types.append('Линейная (ручейковая)')
        
        if v_std > 60:
            erosion_indicators += 25
            erosion_types.append('Пятнистая (вымывание)')
        
        # Проверка на оголённые участки
        low_sat_mask = hsv[:, :, 1] < 30
        low_sat_ratio = np.sum(low_sat_mask) / (image.shape[0] * image.shape[1])
        if low_sat_ratio > 0.3:
            erosion_indicators += 20
            erosion_types.append('Обнажение подпочвы')
        
        # Классификация
        if erosion_indicators > 50:
            level = 'Высокая'
            status = 'critical'
        elif erosion_indicators > 25:
            level = 'Умеренная'
            status = 'warning'
        elif erosion_indicators > 10:
            level = 'Слабая'
            status = 'attention'
        else:
            level = 'Минимальная или отсутствует'
            status = 'good'
        
        return {
            'erosion_level': level,
            'erosion_index': erosion_indicators,
            'status': status,
            'detected_types': erosion_types if erosion_types else ['Не обнаружено'],
            'linear_features_count': num_lines,
            'surface_variability': round(v_std, 2),
            'protection_recommended': erosion_indicators > 25
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Генерация рекомендаций на основе анализа."""
        recommendations = []
        
        soil = results['soil_analysis']
        moisture = results['moisture_analysis']
        vegetation = results['vegetation_analysis']
        erosion = results['erosion_analysis']
        
        # Рекомендации по почве
        if soil['fertility_score'] < 40:
            recommendations.append(f"⚠️ Низкая плодородность ({soil['name']}). "
                                   "Рекомендуется внесение органических удобрений.")
        
        # Рекомендации по влажности
        if moisture['status'] == 'critical':
            recommendations.append("🔴 Критически низкая влажность! Срочно требуется ирригация.")
        elif moisture['status'] == 'warning' and moisture['moisture_index'] > 70:
            recommendations.append("⚠️ Переувлажнение почвы. Необходим дренаж.")
        elif moisture['moisture_index'] < 30:
            recommendations.append("💧 Рекомендуется регулярный полив.")
        
        # Рекомендации по растительности
        if vegetation['green_cover_percent'] < 20:
            recommendations.append("Низкий растительный покров. "
                                   "Рекомендуется посев покровных культур для защиты почвы.")
        elif vegetation['health_status'] == 'Плохо':
            recommendations.append("Растительность в плохом состоянии. "
                                   "Проверьте питательные вещества и влажность.")
        
        # Рекомендации по эрозии
        if erosion['status'] == 'critical':
            recommendations.append("Высокий риск эрозии! Необходимы срочные меры: "
                                   "террасирование, посадка защитных полос.")
        elif erosion['protection_recommended']:
            recommendations.append("Рекомендуется установка противоэрозионных мер.")
        
        # Рекомендации по культурам
        if soil['fertility_score'] > 60:
            crops = ', '.join(soil['suitable_crops'][:3])
            recommendations.append(f"Подходящие культуры для данной почвы: {crops}")
        
        if not recommendations:
            recommendations.append("Состояние почвы и покрова в норме. "
                                   "Продолжайте текущие агротехнические мероприятия.")
        
        return recommendations
    
    def visualize_analysis(self, image: np.ndarray, results: Dict = None) -> np.ndarray:
        """
        Создание визуализации анализа.
        """
        if results is None:
            results = self.analysis_results
        
        if not results:
            # Если нет результатов, просто вернём изображение с надписью
            vis = image.copy()
            cv2.putText(vis, "Analyzing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 255), 2, cv2.LINE_AA)
            return vis
        
        # Создание визуализации
        h, w = image.shape[:2]
        
        # Панель с информацией справа
        panel_width = 400
        vis = np.zeros((max(h, 700), w + panel_width, 3), dtype=np.uint8)
        vis[:h, :w] = image
        
        # Фон панели
        vis[:, w:] = (40, 40, 40)
        
        # Текст на панели
        y_offset = 30
        line_height = 25
        x_pos = w + 10
        
        def put_text(text, y, color=(255, 255, 255), scale=0.5):
            cv2.putText(vis, str(text), (x_pos, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        scale, color, 1, cv2.LINE_AA)
        
        # Заголовок
        put_text("=== АНАЛИЗ ПОЧВЫ ===", y_offset, (0, 255, 255), 0.7)
        y_offset += line_height + 10
        
        # Тип почвы
        soil = results['soil_analysis']
        put_text(f"Тип: {soil['name']}", y_offset, (100, 255, 100))
        y_offset += line_height
        put_text(f"({soil['name_en']})", y_offset, (150, 150, 150), 0.4)
        y_offset += line_height
        put_text(f"Уверенность: {soil['confidence']}%", y_offset)
        y_offset += line_height
        put_text(f"Плодородность: {soil['fertility']}", y_offset)
        y_offset += line_height
        put_text(f"pH: {soil['ph_range']}", y_offset)
        y_offset += line_height + 10
        
        # Влажность
        put_text("=== ВЛАЖНОСТЬ ===", y_offset, (0, 255, 255), 0.6)
        y_offset += line_height
        moisture = results['moisture_analysis']
        color = (0, 255, 0) if moisture['status'] == 'optimal' else \
                (0, 255, 255) if moisture['status'] == 'good' else \
                (0, 165, 255) if moisture['status'] == 'warning' else (0, 0, 255)
        put_text(f"Уровень: {moisture['level']}", y_offset, color)
        y_offset += line_height
        put_text(f"Индекс: {moisture['moisture_index']}%", y_offset)
        y_offset += line_height + 10
        
        # Растительность
        put_text("=== РАСТИТЕЛЬНОСТЬ ===", y_offset, (0, 255, 255), 0.6)
        y_offset += line_height
        veg = results['vegetation_analysis']
        put_text(f"Покров: {veg['cover_type']}", y_offset, (100, 255, 100))
        y_offset += line_height
        put_text(f"Зелёный: {veg['green_cover_percent']}%", y_offset)
        y_offset += line_height
        put_text(f"NDVI (оценка): {veg['ndvi_estimate']}", y_offset)
        y_offset += line_height + 10
        
        # Текстура
        put_text("=== ТЕКСТУРА ===", y_offset, (0, 255, 255), 0.6)
        y_offset += line_height
        tex = results['texture_analysis']
        put_text(f"Тип: {tex['texture_type']}", y_offset)
        y_offset += line_height
        put_text(f"Частицы: {tex['particle_size']}", y_offset)
        y_offset += line_height + 10
        
        # Эрозия
        put_text("=== ЭРОЗИЯ ===", y_offset, (0, 255, 255), 0.6)
        y_offset += line_height
        eros = results['erosion_analysis']
        color = (0, 255, 0) if eros['status'] == 'good' else \
                (0, 255, 255) if eros['status'] == 'attention' else \
                (0, 165, 255) if eros['status'] == 'warning' else (0, 0, 255)
        put_text(f"Уровень: {eros['erosion_level']}", y_offset, color)
        y_offset += line_height
        
        return vis
    
    def print_report(self, results: Dict = None):
        """Вывод текстового отчёта."""
        if results is None:
            results = self.analysis_results
        
        if not results:
            print("Нет данных для отчёта. Сначала выполните анализ.")
            return
        
        print("\n" + "=" * 70)
        print("           ОТЧЁТ АНАЛИЗА ЗЕМЛИ И ПОЧВЫ")
        print("=" * 70)
        print(f"Дата анализа: {results['timestamp']}")
        print(f"Размер изображения: {results['image_size']}")
        
        # Почва
        soil = results['soil_analysis']
        print("\n" + "-" * 40)
        print("АНАЛИЗ ПОЧВЫ")
        print("-" * 40)
        print(f"  Тип почвы: {soil['name']} ({soil['name_en']})")
        print(f"  Уверенность: {soil['confidence']}%")
        print(f"  Цвет: {soil['color_description']}")
        print(f"  Плодородность: {soil['fertility']} ({soil['fertility_score']}/100)")
        print(f"  Органическое вещество: {soil['organic_matter']}")
        print(f"  pH диапазон: {soil['ph_range']}")
        print(f"  Водоудержание: {soil['water_retention']}")
        print(f"  Типичные регионы: {soil['typical_regions']}")
        print(f"  Подходящие культуры: {', '.join(soil['suitable_crops'])}")
        
        # Влажность
        moisture = results['moisture_analysis']
        print("\n" + "-" * 40)
        print("АНАЛИЗ ВЛАЖНОСТИ")
        print("-" * 40)
        print(f"  Индекс влажности: {moisture['moisture_index']}%")
        print(f"  Уровень: {moisture['level']}")
        print(f"  Рекомендация: {moisture['irrigation_recommendation']}")
        
        # Растительность
        veg = results['vegetation_analysis']
        print("\n" + "-" * 40)
        print("АНАЛИЗ РАСТИТЕЛЬНОСТИ")
        print("-" * 40)
        print(f"  Тип покрова: {veg['cover_type']}")
        print(f"  Состояние: {veg['health_status']}")
        print(f"  Зелёный покров: {veg['green_cover_percent']}%")
        print(f"  Сухая растительность: {veg['dry_vegetation_percent']}%")
        print(f"  Голая почва: {veg['bare_soil_percent']}%")
        print(f"  NDVI (оценка): {veg['ndvi_estimate']}")
        
        # Текстура
        tex = results['texture_analysis']
        print("\n" + "-" * 40)
        print("АНАЛИЗ ТЕКСТУРЫ")
        print("-" * 40)
        print(f"  Тип текстуры: {tex['texture_type']}")
        print(f"  Размер частиц: {tex['particle_size']}")
        print(f"  Шероховатость: {tex['roughness_index']}")
        print(f"  Уплотнение: {tex['compaction_estimate']}")
        
        # Эрозия
        eros = results['erosion_analysis']
        print("\n" + "-" * 40)
        print("АНАЛИЗ ЭРОЗИИ")
        print("-" * 40)
        print(f"  Уровень эрозии: {eros['erosion_level']}")
        print(f"  Индекс: {eros['erosion_index']}")
        print(f"  Обнаруженные типы: {', '.join(eros['detected_types'])}")
        
        # Рекомендации
        print("\n" + "-" * 40)
        print("РЕКОМЕНДАЦИИ")
        print("-" * 40)
        for rec in results['recommendations']:
            print(f"  {rec}")
        
        print("\n" + "=" * 70)


def run_soil_analyzer():
    """Запуск анализатора почвы."""
    print("\n" + "=" * 60)
    print("      ПРОДВИНУТЫЙ АНАЛИЗАТОР ЗЕМЛИ И ПОЧВЫ")
    print("=" * 60)
    
    analyzer = TerrainSoilAnalyzer()
    
    while True:
        print("\n1. Анализ изображения из файла")
        print("2. Анализ с веб-камеры (одиночный кадр)")
        print("3. Анализ с веб-камеры (реальное время)")
        print("4. Анализ всех изображений в папке")
        print("5. Информация о типах почв")
        print("6. Назад в главное меню")
        print("-" * 40)
        
        choice = input("Выберите опцию (1-6): ").strip()
        
        if choice == '1':
            # Анализ из файла
            image_path = input("Введите путь к изображению: ").strip()
            if not os.path.exists(image_path):
                # Проверка в папке Data
                if os.path.exists(os.path.join("Data", image_path)):
                    image_path = os.path.join("Data", image_path)
                else:
                    print("Файл не найден!")
                    continue
            
            image = cv2.imread(image_path)
            if image is None:
                print("Не удалось загрузить изображение!")
                continue
            
            print("\nАнализ изображения...")
            results = analyzer.analyze_image(image)
            analyzer.print_report(results)
            
            # Визуализация
            vis = analyzer.visualize_analysis(image, results)
            cv2.imshow("Soil Analysis", vis)
            print("\nНажмите любую клавишу для закрытия...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Сохранение
            save = input("\nСохранить отчёт? (y/n): ").strip().lower()
            if save == 'y':
                os.makedirs("test_output", exist_ok=True)
                cv2.imwrite("test_output/soil_analysis_result.jpg", vis)
                
                # Сохранение текстового отчёта
                with open("test_output/soil_analysis_report.txt", 'w', encoding='utf-8') as f:
                    f.write(f"Отчёт анализа почвы\n")
                    f.write(f"Дата: {results['timestamp']}\n")
                    f.write(f"Файл: {image_path}\n\n")
                    f.write(f"Тип почвы: {results['soil_analysis']['name']}\n")
                    f.write(f"Влажность: {results['moisture_analysis']['moisture_index']}%\n")
                    f.write(f"Растительность: {results['vegetation_analysis']['green_cover_percent']}%\n")
                
                print("Сохранено в test_output/")
        
        elif choice == '2':
            # Одиночный кадр с камеры
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Камера недоступна!")
                continue
            
            print("Нажмите ПРОБЕЛ для захвата или Q для выхода")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                cv2.imshow("Camera - Press SPACE to capture", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):
                    print("\nАнализ захваченного кадра...")
                    results = analyzer.analyze_image(frame)
                    analyzer.print_report(results)
                    
                    vis = analyzer.visualize_analysis(frame, results)
                    cv2.imshow("Soil Analysis", vis)
                    cv2.waitKey(0)
                    break
                elif key == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '3':
            # Реальное время
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Камера недоступна!")
                continue
            
            print("Анализ в реальном времени. Нажмите Q для выхода.")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Анализ каждые 10 кадров для производительности
                if frame_count % 10 == 0:
                    results = analyzer.analyze_image(frame)
                    vis = analyzer.visualize_analysis(frame, results)
                else:
                    vis = analyzer.visualize_analysis(frame)
                
                cv2.imshow("Soil Analysis - Real-time", vis)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '4':
            # Анализ всех изображений в папке
            folder = input("Введите путь к папке (Enter для 'Data'): ").strip() or "Data"
            
            if not os.path.exists(folder):
                print("Папка не найдена!")
                continue
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            images = [f for f in os.listdir(folder) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
            
            if not images:
                print("Изображения не найдены!")
                continue
            
            print(f"\nНайдено {len(images)} изображений. Начинаем анализ...")
            
            os.makedirs("test_output/soil_batch", exist_ok=True)
            
            for i, img_name in enumerate(images, 1):
                img_path = os.path.join(folder, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"  [{i}/{len(images)}] Ошибка загрузки: {img_name}")
                    continue
                
                results = analyzer.analyze_image(image)
                soil_type = results['soil_analysis']['name']
                moisture = results['moisture_analysis']['moisture_index']
                
                print(f"  [{i}/{len(images)}] {img_name}: {soil_type}, влажность {moisture}%")
                
                # Сохранение визуализации
                vis = analyzer.visualize_analysis(image, results)
                out_name = f"analysis_{os.path.splitext(img_name)[0]}.jpg"
                cv2.imwrite(f"test_output/soil_batch/{out_name}", vis)
            
            print(f"\nРезультаты сохранены в test_output/soil_batch/")
        
        elif choice == '5':
            # Информация о типах почв
            print("\n" + "=" * 70)
            print("              СПРАВОЧНИК ТИПОВ ПОЧВ")
            print("=" * 70)
            
            for soil_type, info in analyzer.SOIL_TYPES.items():
                print(f"\n{info['name']} ({info['name_en']})")
                print("-" * 50)
                print(f"  Цвет: {info['color_desc']}")
                print(f"  Плодородность: {info['fertility']} ({info['fertility_score']}/100)")
                print(f"  Органика: {info['organic_matter']}")
                print(f"  pH: {info['ph_range']}")
                print(f"  Водоудержание: {info['water_retention']}")
                print(f"  Культуры: {', '.join(info['suitable_crops'])}")
                print(f"  Регионы: {info['regions']}")
        
        elif choice == '6':
            break
        
        else:
            print("Некорректный выбор")


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
        print("5. Анализатор земли и почвы")
        print("6. Выход")
        print("-" * 60)
        
        choice = input("Выберите опцию (1-6): ").strip()
        
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
            run_soil_analyzer()
        
        elif choice == '6':
            print("\nВыход из программы")
            break
        
        else:
            print("Некорректный выбор, попробуйте снова")


if __name__ == "__main__":
    main_menu()
