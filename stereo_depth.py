"""
Модуль для определения рельефа местности через стереоскопические камеры.
Включает:
- Калибровку стереокамер
- Построение карты глубины (disparity map)
- Генерацию облака точек (point cloud)
- Визуализацию 3D ландшафта
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import os


class StereoDepthEstimator:
    """
    Класс для оценки глубины с помощью стереоскопических камер.
    
    Стереоскопия работает по принципу триангуляции:
    - Две камеры расположены на известном расстоянии (baseline)
    - Один и тот же объект виден под разными углами
    - По разнице положений (disparity) вычисляется глубина
    
    Формула глубины: Z = (f * B) / d
    где f - фокусное расстояние, B - база (расстояние между камерами), d - диспаратность
    """
    
    def __init__(self, baseline: float = 0.1, focal_length: float = 700.0):
        """
        Args:
            baseline: расстояние между камерами в метрах (по умолчанию 10 см)
            focal_length: фокусное расстояние в пикселях
        """
        self.baseline = baseline
        self.focal_length = focal_length
        
        # Параметры калибровки (заполняются после калибровки)
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.R = None  # Матрица поворота между камерами
        self.T = None  # Вектор переноса между камерами
        self.Q = None  # Матрица для репроекции в 3D
        
        # Карты для ректификации
        self.map_left_x = None
        self.map_left_y = None
        self.map_right_x = None
        self.map_right_y = None
        
        # Создание стерео-матчера
        self._init_stereo_matcher()
    
    def _init_stereo_matcher(self, algorithm: str = "sgbm"):
        """
        Инициализация алгоритма сопоставления стереопар.
        
        Алгоритмы:
        - StereoBM: быстрый, но менее точный (Block Matching)
        - StereoSGBM: более точный, использует Semi-Global Block Matching
        """
        if algorithm == "bm":
            # StereoBM - простой и быстрый алгоритм
            self.stereo = cv2.StereoBM_create(
                numDisparities=16*8,  # Должно быть кратно 16
                blockSize=15  # Размер окна сопоставления (нечётный, 5-21)
            )
        else:
            # StereoSGBM - более точный алгоритм
            window_size = 5
            min_disp = 0
            num_disp = 16*8  # Должно быть кратно 16
            
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=8 * 3 * window_size**2,   # Штраф за изменение диспаратности на 1
                P2=32 * 3 * window_size**2,  # Штраф за изменение диспаратности больше 1
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            
            # Правый матчер для WLS фильтра
            self.stereo_right = cv2.ximgproc.createRightMatcher(self.stereo) if hasattr(cv2, 'ximgproc') else None
            
            # WLS фильтр для сглаживания карты диспаратности
            if hasattr(cv2, 'ximgproc'):
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
                self.wls_filter.setLambda(8000)
                self.wls_filter.setSigmaColor(1.5)
            else:
                self.wls_filter = None
    
    def calibrate_stereo_cameras(self, 
                                  images_left: List[np.ndarray], 
                                  images_right: List[np.ndarray],
                                  pattern_size: Tuple[int, int] = (9, 6),
                                  square_size: float = 0.025) -> bool:
        """
        Калибровка стереокамер по изображениям шахматной доски.
        
        Args:
            images_left: список изображений с левой камеры
            images_right: список изображений с правой камеры
            pattern_size: размер шахматной доски (внутренние углы)
            square_size: размер клетки в метрах
            
        Returns:
            True если калибровка успешна
        """
        # Критерии для уточнения углов
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Подготовка точек объекта
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints = []  # 3D точки в реальном мире
        imgpoints_left = []  # 2D точки на изображении левой камеры
        imgpoints_right = []  # 2D точки на изображении правой камеры
        
        for img_left, img_right in zip(images_left, images_right):
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            
            # Поиск углов шахматной доски
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)
            
            if ret_left and ret_right:
                objpoints.append(objp)
                
                # Уточнение положения углов
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
                
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
        
        if len(objpoints) < 3:
            print("Недостаточно изображений для калибровки")
            return False
        
        img_size = gray_left.shape[::-1]
        
        # Калибровка каждой камеры отдельно
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_left, img_size, None, None
        )
        
        ret_right, self.camera_matrix_right, self.dist_coeffs_right, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_right, img_size, None, None
        )
        
        # Стерео калибровка
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret, _, _, _, _, self.R, self.T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            img_size, criteria=criteria, flags=flags
        )
        
        # Ректификация
        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            img_size, self.R, self.T,
            alpha=0
        )
        
        # Создание карт для ремаппинга
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left,
            R1, P1, img_size, cv2.CV_32FC1
        )
        
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right,
            R2, P2, img_size, cv2.CV_32FC1
        )
        
        # Обновление базы и фокусного расстояния
        self.baseline = abs(self.T[0][0])
        self.focal_length = P1[0, 0]
        
        print(f"Калибровка завершена. RMS ошибка: {ret:.4f}")
        print(f"База: {self.baseline:.4f} м, Фокусное расстояние: {self.focal_length:.2f} пикселей")
        
        return True
    
    def rectify_images(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ректификация изображений (выравнивание эпиполярных линий).
        После ректификации соответствующие точки находятся на одной горизонтальной линии.
        """
        if self.map_left_x is None:
            return img_left, img_right
        
        rectified_left = cv2.remap(img_left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        
        return rectified_left, rectified_right
    
    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Вычисление карты диспаратности (disparity map).
        
        Диспаратность - это разница в положении одной и той же точки
        на левом и правом изображениях.
        
        Большая диспаратность = близкий объект
        Малая диспаратность = далёкий объект
        """
        # Преобразование в градации серого
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        # Вычисление диспаратности
        disparity_left = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Применение WLS фильтра для сглаживания
        if self.wls_filter is not None and self.stereo_right is not None:
            disparity_right = self.stereo_right.compute(gray_right, gray_left).astype(np.float32) / 16.0
            disparity = self.wls_filter.filter(disparity_left, gray_left, None, disparity_right)
        else:
            disparity = disparity_left
        
        return disparity
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Преобразование карты диспаратности в карту глубины.
        
        Формула: Z = (f * B) / d
        где:
        - Z - глубина (расстояние до объекта)
        - f - фокусное расстояние
        - B - база (расстояние между камерами)
        - d - диспаратность
        """
        # Избегаем деления на ноль
        disparity_safe = np.maximum(disparity, 0.1)
        
        depth = (self.focal_length * self.baseline) / disparity_safe
        
        # Ограничение максимальной глубины
        depth = np.clip(depth, 0, 100)  # Максимум 100 метров
        
        return depth
    
    def compute_point_cloud(self, img_left: np.ndarray, disparity: np.ndarray) -> np.ndarray:
        """
        Генерация облака точек из изображения и карты диспаратности.
        
        Returns:
            points: массив Nx6 (X, Y, Z, R, G, B)
        """
        h, w = disparity.shape
        
        # Создание сетки координат
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Преобразование в 3D координаты
        if self.Q is not None:
            # Использование матрицы Q для репроекции
            points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        else:
            # Ручной расчёт
            depth = self.disparity_to_depth(disparity)
            
            # Центр изображения
            cx = w / 2
            cy = h / 2
            
            # Вычисление 3D координат
            Z = depth
            X = (u - cx) * Z / self.focal_length
            Y = (v - cy) * Z / self.focal_length
            
            points_3d = np.dstack([X, Y, Z])
        
        # Фильтрация некорректных точек
        mask = (disparity > 0) & (disparity < 128) & np.isfinite(points_3d[:, :, 2])
        
        # Получение цветов
        if len(img_left.shape) == 3:
            colors = img_left[mask]
        else:
            colors = np.stack([img_left[mask]] * 3, axis=-1)
        
        points = points_3d[mask]
        
        # Объединение координат и цветов
        point_cloud = np.hstack([points, colors])
        
        return point_cloud
    
    def save_point_cloud_ply(self, points: np.ndarray, filename: str):
        """
        Сохранение облака точек в формате PLY.
        """
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(filename, 'w') as f:
            f.write(header)
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(p[5])} {int(p[4])} {int(p[3])}\n")
        
        print(f"Облако точек сохранено: {filename} ({len(points)} точек)")
    
    def visualize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Визуализация карты диспаратности с цветовой картой.
        """
        # Нормализация для отображения
        disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_colored = cv2.applyColorMap(disp_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        return disp_colored
    
    def visualize_depth(self, depth: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
        """
        Визуализация карты глубины.
        """
        # Нормализация
        depth_normalized = np.clip(depth / max_depth, 0, 1) * 255
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
        
        return depth_colored


class StereoTerrainMapper:
    """
    Класс для построения карты рельефа местности на основе стереозрения.
    """
    
    def __init__(self, stereo_estimator: StereoDepthEstimator):
        self.stereo = stereo_estimator
        self.terrain_points = []
        self.accumulated_cloud = None
    
    def process_stereo_frame(self, img_left: np.ndarray, img_right: np.ndarray) -> dict:
        """
        Обработка стереопары и извлечение информации о рельефе.
        """
        # Ректификация
        rect_left, rect_right = self.stereo.rectify_images(img_left, img_right)
        
        # Вычисление диспаратности
        disparity = self.stereo.compute_disparity(rect_left, rect_right)
        
        # Преобразование в глубину
        depth = self.stereo.disparity_to_depth(disparity)
        
        # Генерация облака точек
        point_cloud = self.stereo.compute_point_cloud(rect_left, disparity)
        
        # Визуализации
        disparity_vis = self.stereo.visualize_disparity(disparity)
        depth_vis = self.stereo.visualize_depth(depth)
        
        return {
            'disparity': disparity,
            'depth': depth,
            'point_cloud': point_cloud,
            'disparity_vis': disparity_vis,
            'depth_vis': depth_vis,
            'rectified_left': rect_left,
            'rectified_right': rect_right
        }
    
    def extract_terrain_profile(self, depth: np.ndarray, 
                                  row: Optional[int] = None) -> np.ndarray:
        """
        Извлечение профиля рельефа из карты глубины.
        """
        if row is None:
            row = depth.shape[0] // 2
        
        profile = depth[row, :]
        return profile
    
    def detect_obstacles(self, depth: np.ndarray, 
                         threshold: float = 2.0) -> np.ndarray:
        """
        Обнаружение препятствий на основе карты глубины.
        """
        # Близкие объекты (препятствия)
        obstacles = depth < threshold
        
        # Морфологическая обработка для удаления шума
        kernel = np.ones((5, 5), np.uint8)
        obstacles = cv2.morphologyEx(obstacles.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_CLOSE, kernel)
        
        return obstacles


def demo_stereo_depth():
    """
    Демонстрация работы со стереокамерами.
    """
    # Создание объекта для оценки глубины
    stereo = StereoDepthEstimator(baseline=0.1, focal_length=700)
    
    # Пример: чтение с двух камер
    # cap_left = cv2.VideoCapture(0)  # Левая камера
    # cap_right = cv2.VideoCapture(1)  # Правая камера
    
    print("=" * 60)
    print("Демонстрация работы стереоскопических камер")
    print("=" * 60)
    print("""
    Для работы со стереокамерами необходимо:
    
    1. Калибровка камер:
       - Снять 15-20 изображений шахматной доски с обеих камер
       - Запустить stereo.calibrate_stereo_cameras()
    
    2. Получение глубины:
       - Захват синхронных кадров с обеих камер
       - Ректификация изображений
       - Вычисление карты диспаратности
       - Преобразование в глубину/облако точек
    
    3. Визуализация:
       - Карта диспаратности (disparity map)
       - Карта глубины (depth map)
       - 3D облако точек (point cloud)
    """)
    
    # Симуляция работы с синтетическими данными
    print("\nСоздание синтетических данных для демонстрации...")
    
    # Создание тестовых изображений с искусственной диспаратностью
    h, w = 480, 640
    img_left = np.zeros((h, w), dtype=np.uint8)
    img_right = np.zeros((h, w), dtype=np.uint8)
    
    # Добавление объектов на разных глубинах
    cv2.rectangle(img_left, (100, 100), (200, 200), 255, -1)  # Близкий объект
    cv2.rectangle(img_right, (80, 100), (180, 200), 255, -1)  # Сдвиг 20 пикселей
    
    cv2.rectangle(img_left, (400, 300), (500, 400), 200, -1)  # Далёкий объект
    cv2.rectangle(img_right, (395, 300), (495, 400), 200, -1)  # Сдвиг 5 пикселей
    
    # Вычисление диспаратности
    disparity = stereo.compute_disparity(img_left, img_right)
    depth = stereo.disparity_to_depth(disparity)
    
    print(f"Размер карты диспаратности: {disparity.shape}")
    print(f"Диапазон глубин: {depth.min():.2f} - {depth.max():.2f} м")
    
    return stereo


if __name__ == "__main__":
    demo_stereo_depth()
