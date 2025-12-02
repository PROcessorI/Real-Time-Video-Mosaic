"""
Построение 3D модели из видео с использованием монокулярной оценки глубины.

Пайплайн:
1. Оценка глубины с помощью нейросети (GLPN, MiDaS, Depth Anything)
2. Построение облака точек из RGB-D изображения
3. Генерация полигональной сетки (mesh) алгоритмом Пуассона

Основано на статье о монокулярной оценке глубины: https://habr.com/ru/companies/skillfactory/articles/693338/
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, List
from PIL import Image
import time

# Проверка зависимостей
print("Проверка зависимостей...")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    print("✓ Plotly (интерактивная 3D визуализация)")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("✗ Plotly не установлен: pip install plotly")

try:
    import torch
    TORCH_AVAILABLE = True
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch не установлен: pip install torch torchvision")

try:
    from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("✗ Transformers не установлен: pip install transformers")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("✓ Open3D")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("✗ Open3D не установлен: pip install open3d")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("✓ Matplotlib")
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MonocularDepthEstimator:
    """
    Монокулярная оценка глубины с использованием различных моделей.
    
    Доступные модели:
    - GLPN (vinvino02/glpn-nyu) - легкая и быстрая
    - DPT (Intel/dpt-large) - высокое качество
    - MiDaS (Intel/dpt-hybrid-midas) - хороший баланс
    - Depth Anything (LiheYoung/depth-anything-base-hf) - новейшая модель
    """
    
    MODELS = {
        'glpn': 'vinvino02/glpn-nyu',
        'dpt-large': 'Intel/dpt-large', 
        'midas': 'Intel/dpt-hybrid-midas',
        'depth-anything': 'LiheYoung/depth-anything-base-hf',
        'depth-anything-small': 'LiheYoung/depth-anything-small-hf',
    }
    
    def __init__(self, model_name: str = 'depth-anything-small'):
        """
        Args:
            model_name: название модели (glpn, dpt-large, midas, depth-anything)
        """
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("Требуются PyTorch и Transformers!")
        
        self.model_name = model_name
        self.model_id = self.MODELS.get(model_name, model_name)
        
        print(f"\nЗагрузка модели: {self.model_id}...")
        
        # Определяем устройство
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Устройство: {self.device}")
        
        # Загрузка модели через pipeline (проще)
        try:
            self.pipe = pipeline(
                task="depth-estimation",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1
            )
            self.use_pipeline = True
            print(f"✓ Модель загружена (pipeline)")
        except Exception as e:
            print(f"Pipeline не удался, пробуем прямую загрузку: {e}")
            # Прямая загрузка
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            self.use_pipeline = False
            print(f"✓ Модель загружена (прямая)")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Оценка глубины для одного изображения.
        
        Args:
            image: BGR или RGB изображение (numpy array)
            
        Returns:
            depth_map: карта глубины (float32)
        """
        # Конвертация в PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Предполагаем BGR (OpenCV)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        if self.use_pipeline:
            # Через pipeline
            result = self.pipe(pil_image)
            depth = np.array(result["depth"])
        else:
            # Прямой inference
            inputs = self.image_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Интерполяция до размера входного изображения
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=pil_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            
            depth = prediction.squeeze().cpu().numpy()
        
        return depth.astype(np.float32)
    
    def estimate_depth_video(self, video_path: str, 
                              frame_step: int = 10,
                              max_frames: int = 50) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Оценка глубины для видео.
        
        Returns:
            List of (rgb_image, depth_map) tuples
        """
        print(f"\nОбработка видео: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Всего кадров: {total_frames}")
        print(f"Обработка каждого {frame_step}-го кадра (макс. {max_frames})...")
        
        results = []
        frame_idx = 0
        processed = 0
        
        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_step == 0:
                # Оценка глубины
                depth = self.estimate_depth(frame)
                
                # Сохраняем RGB версию
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results.append((rgb, depth))
                
                processed += 1
                print(f"  Кадр {frame_idx}/{total_frames} -> глубина {depth.min():.2f}-{depth.max():.2f}")
            
            frame_idx += 1
        
        cap.release()
        print(f"✓ Обработано {len(results)} кадров")
        
        return results


class DepthToPointCloud:
    """
    Преобразование RGB-D изображения в облако точек.
    """
    
    def __init__(self, fx: float = 500, fy: float = 500, 
                 cx: float = None, cy: float = None):
        """
        Args:
            fx, fy: фокусные расстояния (в пикселях)
            cx, cy: координаты главной точки (центр изображения)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def create_point_cloud(self, rgb: np.ndarray, depth: np.ndarray,
                           depth_scale: float = 1.0,
                           depth_trunc: float = 10.0) -> 'o3d.geometry.PointCloud':
        """
        Создание облака точек из RGB и карты глубины.
        
        Args:
            rgb: RGB изображение
            depth: карта глубины
            depth_scale: масштаб глубины
            depth_trunc: максимальная глубина для обрезки
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D не установлен!")
        
        height, width = depth.shape[:2]
        
        # Установка центра, если не задан
        cx = self.cx if self.cx is not None else width / 2
        cy = self.cy if self.cy is not None else height / 2
        
        # Нормализация глубины
        depth_normalized = depth.copy()
        if depth_normalized.max() > 0:
            depth_normalized = depth_normalized / depth_normalized.max() * depth_scale
        
        # Преобразование в uint16 для Open3D
        depth_uint16 = (depth_normalized * 1000).astype(np.uint16)
        
        # Создание RGBD изображения
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_uint16)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # Внутренние параметры камеры
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, self.fx, self.fy, cx, cy
        )
        
        # Создание облака точек
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        return pcd
    
    def create_point_cloud_manual(self, rgb: np.ndarray, depth: np.ndarray,
                                   depth_scale: float = 1.0,
                                   subsample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ручное создание облака точек (без Open3D).
        
        Args:
            rgb: RGB изображение
            depth: карта глубины
            depth_scale: масштаб глубины
            subsample: шаг субдискретизации (4 = каждый 4-й пиксель)
        
        Returns:
            points: Nx3 массив координат
            colors: Nx3 массив цветов
        """
        height, width = depth.shape[:2]
        
        # Субдискретизация для уменьшения количества точек
        depth_sub = depth[::subsample, ::subsample]
        rgb_sub = rgb[::subsample, ::subsample]
        h_sub, w_sub = depth_sub.shape[:2]
        
        cx = (self.cx if self.cx is not None else width / 2) / subsample
        cy = (self.cy if self.cy is not None else height / 2) / subsample
        fx = self.fx / subsample
        fy = self.fy / subsample
        
        # Нормализация глубины
        depth_norm = depth_sub.astype(np.float32)
        if depth_norm.max() > 0:
            depth_norm = depth_norm / depth_norm.max() * depth_scale
        
        # Создание сетки координат
        u, v = np.meshgrid(np.arange(w_sub), np.arange(h_sub))
        
        # Преобразование в 3D
        z = depth_norm
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Маска валидных точек
        valid = z > 0.01
        
        # Извлечение точек (используем float32 для экономии памяти)
        points = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)
        
        # Цвета
        if len(rgb_sub.shape) == 3:
            colors = (rgb_sub[valid] / 255.0).astype(np.float32)
        else:
            colors = np.stack([rgb_sub[valid]] * 3, axis=-1).astype(np.float32) / 255.0
        
        return points, colors


class MeshGenerator:
    """
    Генерация полигональной сетки из облака точек.
    """
    
    @staticmethod
    def filter_outliers(pcd: 'o3d.geometry.PointCloud', 
                        nb_neighbors: int = 20,
                        std_ratio: float = 2.0) -> 'o3d.geometry.PointCloud':
        """Удаление выбросов."""
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return pcd.select_by_index(ind)
    
    @staticmethod
    def estimate_normals(pcd: 'o3d.geometry.PointCloud',
                         radius: float = 0.1,
                         max_nn: int = 30) -> 'o3d.geometry.PointCloud':
        """Оценка нормалей."""
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        return pcd
    
    @staticmethod
    def create_mesh_poisson(pcd: 'o3d.geometry.PointCloud',
                            depth: int = 9) -> 'o3d.geometry.TriangleMesh':
        """
        Создание mesh методом Пуассона.
        
        Args:
            pcd: облако точек с нормалями
            depth: глубина октодерева (5-10, больше = детальнее)
        """
        print(f"  Реконструкция Пуассона (depth={depth})...")
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, n_threads=-1
        )
        
        # Удаление вершин с низкой плотностью
        densities = np.asarray(densities)
        threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        mesh.compute_vertex_normals()
        
        print(f"  ✓ Mesh: {len(mesh.vertices)} вершин, {len(mesh.triangles)} треугольников")
        
        return mesh
    
    @staticmethod
    def create_mesh_ball_pivoting(pcd: 'o3d.geometry.PointCloud') -> 'o3d.geometry.TriangleMesh':
        """Создание mesh методом Ball Pivoting."""
        print("  Реконструкция Ball Pivoting...")
        
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 0.5, avg_dist, avg_dist * 2, avg_dist * 4]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        mesh.compute_vertex_normals()
        
        print(f"  ✓ Mesh: {len(mesh.vertices)} вершин, {len(mesh.triangles)} треугольников")
        
        return mesh


def visualize_3d_interactive(points: np.ndarray, colors: np.ndarray = None, 
                              title: str = "3D Point Cloud", max_points: int = 100000):
    """
    Интерактивная 3D визуализация в браузере с Plotly.
    
    Управление:
    - Левая кнопка мыши: вращение
    - Колёсико: масштаб  
    - Shift + мышь: перемещение
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly не установлен! pip install plotly")
        return
    
    print(f"\nПодготовка визуализации ({len(points)} точек)...")
    
    # Субдискретизация для плавной работы
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
        print(f"  Субдискретизация: {max_points} точек")
    
    # Подготовка цветов
    if colors is not None and len(colors) == len(points):
        color_values = ['rgb({},{},{})'.format(
            int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
        marker_dict = dict(size=2, color=color_values, opacity=0.8)
    else:
        marker_dict = dict(
            size=2, 
            color=points[:, 2],
            colorscale='Viridis',
            colorbar=dict(title='Глубина (Z)'),
            opacity=0.8
        )
    
    # Создание графика
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=marker_dict,
        name='Point Cloud'
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (глубина)',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=-1, z=0),
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    print("Открытие в браузере...")
    fig.show()
    
    return fig


def visualize_mesh_interactive(mesh, title: str = "3D Mesh"):
    """
    Интерактивная визуализация mesh в браузере.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly не установлен!")
        return
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    print(f"\nВизуализация mesh: {len(vertices)} вершин, {len(triangles)} треугольников")
    
    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=vertices[:, 2],
        colorscale='Viridis',
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
        name='Mesh'
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1400,
        height=900
    )
    
    print("Открытие в браузере...")
    fig.show()
    
    return fig


def process_video_to_3d_model(video_path: str,
                               model_name: str = 'depth-anything-small',
                               frame_step: int = 15,
                               max_frames: int = 30,
                               mesh_depth: int = 8,
                               output_dir: str = None,
                               visualize: bool = True,
                               single_frame: bool = False):
    """
    Полный пайплайн: видео -> 3D модель.
    
    Args:
        video_path: путь к видео
        model_name: модель для оценки глубины
        frame_step: шаг между кадрами
        max_frames: максимум кадров для обработки
        mesh_depth: глубина для Пуассона (5-10)
        output_dir: директория для сохранения
        visualize: показывать 3D визуализацию после обработки
        single_frame: использовать только один кадр (лучше для статичных сцен)
    """
    print("\n" + "="*70)
    print("ПОСТРОЕНИЕ 3D МОДЕЛИ ИЗ ВИДЕО")
    print("="*70)
    
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 1. Оценка глубины
    print("\n[1/4] ОЦЕНКА ГЛУБИНЫ")
    print("-" * 40)
    
    estimator = MonocularDepthEstimator(model_name)
    
    # Режим одного кадра для статичных сцен
    if single_frame:
        print("  Режим: ОДИН КАДР (рекомендуется для статичных сцен)")
        # Берём кадр из середины видео
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Ошибка: не удалось прочитать кадр!")
            return None
        
        depth = estimator.estimate_depth(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_depth = [(rgb, depth)]
        print(f"  Использован кадр #{mid_frame} из {total}")
    else:
        frames_depth = estimator.estimate_depth_video(video_path, frame_step, max_frames)
    
    if len(frames_depth) == 0:
        print("Ошибка: не удалось обработать кадры!")
        return None
    
    # 2. Построение облаков точек
    print("\n[2/4] ПОСТРОЕНИЕ ОБЛАКОВ ТОЧЕК")
    print("-" * 40)
    
    # Если только 1 кадр - используем его напрямую
    if len(frames_depth) == 1:
        print("  Режим: один кадр (высокое качество)")
        rgb, depth = frames_depth[0]
        converter = DepthToPointCloud(fx=500, fy=500)
        # Меньше subsample для лучшего качества
        combined_points, combined_colors = converter.create_point_cloud_manual(
            rgb, depth, depth_scale=5.0, subsample=2
        )
    else:
        print(f"  Режим: {len(frames_depth)} кадров с ICP выравниванием")
        converter = DepthToPointCloud(fx=500, fy=500)
        
        # Первый кадр как базовый
        rgb_base, depth_base = frames_depth[0]
        base_points, base_colors = converter.create_point_cloud_manual(
            rgb_base, depth_base, depth_scale=5.0, subsample=3
        )
        
        all_points = [base_points]
        all_colors = [base_colors]
        
        # Создаём базовое облако для ICP
        pcd_base = o3d.geometry.PointCloud()
        pcd_base.points = o3d.utility.Vector3dVector(base_points)
        
        for i, (rgb, depth) in enumerate(frames_depth[1:], 1):
            points, colors = converter.create_point_cloud_manual(
                rgb, depth, depth_scale=5.0, subsample=3
            )
            
            # Создаём облако текущего кадра
            pcd_current = o3d.geometry.PointCloud()
            pcd_current.points = o3d.utility.Vector3dVector(points)
            
            # ICP для выравнивания с базовым облаком
            try:
                # Грубое выравнивание
                threshold = 0.5  # Порог расстояния
                trans_init = np.eye(4)  # Начальная трансформация
                
                reg = o3d.pipelines.registration.registration_icp(
                    pcd_current, pcd_base, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                
                # Применяем трансформацию
                if reg.fitness > 0.3:  # Если выравнивание успешное
                    pcd_current.transform(reg.transformation)
                    aligned_points = np.asarray(pcd_current.points).astype(np.float32)
                    print(f"  Кадр {i+1}: {len(points)} точек, ICP fitness={reg.fitness:.2f}")
                else:
                    # Плохое выравнивание - пропускаем кадр
                    print(f"  Кадр {i+1}: пропущен (плохое выравнивание)")
                    continue
                    
            except Exception as e:
                # Если ICP не работает - используем без выравнивания
                print(f"  Кадр {i+1}: {len(points)} точек (без ICP: {e})")
                aligned_points = points
            
            all_points.append(aligned_points)
            all_colors.append(colors)
        
        # Объединение
        print("  Объединение облаков...")
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # Освобождаем память
        del all_points, all_colors
    
    print(f"\n✓ Всего точек: {len(combined_points)}")
    
    # 3. Создание облака точек Open3D
    print("\n[3/4] ОБРАБОТКА ОБЛАКА ТОЧЕК")
    print("-" * 40)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Фильтрация
    print("  Фильтрация выбросов...")
    pcd = MeshGenerator.filter_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
    print(f"  После фильтрации: {len(pcd.points)} точек")
    
    # Downsampling для ускорения
    voxel_size = 0.02
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"  После downsampling: {len(pcd.points)} точек")
    
    # Сохранение облака точек
    pcd_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"  ✓ Облако сохранено: {pcd_path}")
    
    # 4. Генерация mesh
    print("\n[4/4] ГЕНЕРАЦИЯ MESH")
    print("-" * 40)
    
    # Оценка нормалей
    print("  Оценка нормалей...")
    pcd = MeshGenerator.estimate_normals(pcd)
    
    # Создание mesh
    mesh = MeshGenerator.create_mesh_poisson(pcd, depth=mesh_depth)
    
    # Поворот mesh (камера смотрела "вниз")
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    
    # Сохранение mesh
    mesh_path = os.path.join(output_dir, f"{base_name}_mesh.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"  ✓ Mesh сохранён: {mesh_path}")
    
    # Также сохраняем в PLY
    mesh_ply_path = os.path.join(output_dir, f"{base_name}_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_ply_path, mesh)
    
    print("\n" + "="*70)
    print("ГОТОВО!")
    print("="*70)
    print(f"\nФайлы:")
    print(f"  • Облако точек: {pcd_path}")
    print(f"  • Mesh (OBJ): {mesh_path}")
    print(f"  • Mesh (PLY): {mesh_ply_path}")
    
    # Интерактивная 3D визуализация
    if visualize and PLOTLY_AVAILABLE:
        print("\n" + "="*70)
        print("ИНТЕРАКТИВНАЯ 3D ВИЗУАЛИЗАЦИЯ")
        print("="*70)
        print("\nОткрытие в браузере...")
        print("  • Вращение: левая кнопка мыши")
        print("  • Масштаб: колёсико")
        print("  • Перемещение: Shift + мышь")
        
        # Визуализация облака точек
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        visualize_3d_interactive(points, colors, title=f"3D Модель: {base_name}")
    elif visualize and not PLOTLY_AVAILABLE:
        print("\n⚠ Для визуализации установите: pip install plotly")
    
    return pcd, mesh


def process_single_image(image_path: str, model_name: str = 'depth-anything-small'):
    """
    Обработка одного изображения для демонстрации.
    """
    print("\n" + "="*60)
    print("ОБРАБОТКА ИЗОБРАЖЕНИЯ")
    print("="*60)
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        return
    
    print(f"Изображение: {image_path}")
    print(f"Размер: {image.shape[1]}x{image.shape[0]}")
    
    # Оценка глубины
    print("\nОценка глубины...")
    estimator = MonocularDepthEstimator(model_name)
    depth = estimator.estimate_depth(image)
    
    print(f"Глубина: min={depth.min():.3f}, max={depth.max():.3f}")
    
    # Создание облака точек
    print("\nСоздание облака точек...")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    converter = DepthToPointCloud(fx=500, fy=500)
    pcd = converter.create_point_cloud(rgb, depth, depth_scale=5.0)
    
    print(f"Точек: {len(pcd.points)}")
    
    # Фильтрация
    pcd = MeshGenerator.filter_outliers(pcd)
    pcd = pcd.voxel_down_sample(0.02)
    
    # Сохранение
    base = os.path.splitext(image_path)[0]
    pcd_path = f"{base}_pointcloud.ply"
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"✓ Сохранено: {pcd_path}")
    
    # Генерация mesh
    print("\nГенерация mesh...")
    pcd = MeshGenerator.estimate_normals(pcd)
    mesh = MeshGenerator.create_mesh_poisson(pcd, depth=8)
    
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    
    mesh_path = f"{base}_mesh.obj"
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"✓ Mesh сохранён: {mesh_path}")
    
    # Визуализация
    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(rgb)
        axes[0].set_title("Исходное изображение")
        axes[0].axis('off')
        
        axes[1].imshow(depth, cmap='plasma')
        axes[1].set_title("Карта глубины")
        axes[1].axis('off')
        
        # Нормализованная глубина для цветовой шкалы
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        axes[2].imshow(depth_norm, cmap='viridis')
        axes[2].set_title("Глубина (нормализованная)")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        vis_path = f"{base}_depth_visualization.png"
        plt.savefig(vis_path, dpi=150)
        print(f"✓ Визуализация: {vis_path}")
        
        plt.show()
    
    return pcd, mesh


def main():
    """Главная функция."""
    print("\n" + "="*70)
    print("3D РЕКОНСТРУКЦИЯ С МОНОКУЛЯРНОЙ ОЦЕНКОЙ ГЛУБИНЫ")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch не установлен!")
        print("   pip install torch torchvision")
        return
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n❌ Transformers не установлен!")
        print("   pip install transformers")
        return
    
    if not OPEN3D_AVAILABLE:
        print("\n❌ Open3D не установлен!")
        print("   pip install open3d")
        return
    
    print("\nДоступные модели:")
    print("  1. depth-anything-small (быстрая, ~100MB)")
    print("  2. depth-anything (средняя, ~400MB)")
    print("  3. midas (Intel DPT-Hybrid)")
    print("  4. glpn (легкая)")
    
    print("\nВыберите действие:")
    print("  1. Обработать видео (много кадров)")
    print("  2. Обработать изображение")
    print("  3. Тест на примере")
    print("  4. Обработать видео (1 кадр - лучшее качество)")
    
    choice = input("\nВаш выбор (1/2/3/4): ").strip()
    
    if choice == "1":
        video_path = input("Путь к видео: ").strip()
        
        if not os.path.exists(video_path):
            print(f"Файл не найден: {video_path}")
            return
        
        model = input("Модель (Enter для midas): ").strip()
        if not model:
            model = 'midas'
        
        process_video_to_3d_model(
            video_path,
            model_name=model,
            frame_step=15,
            max_frames=10,  # Меньше кадров для лучшего качества
            mesh_depth=9
        )
    
    elif choice == "4":
        # Режим высокого качества - 1 кадр
        video_path = input("Путь к видео: ").strip()
        
        if not os.path.exists(video_path):
            print(f"Файл не найден: {video_path}")
            return
        
        model = input("Модель (Enter для midas): ").strip()
        if not model:
            model = 'midas'
        
        process_video_to_3d_model(
            video_path,
            model_name=model,
            mesh_depth=9,
            single_frame=True  # Один кадр для лучшего качества!
        )
    
    elif choice == "2":
        image_path = input("Путь к изображению: ").strip()
        
        if not os.path.exists(image_path):
            print(f"Файл не найден: {image_path}")
            return
        
        model = input("Модель (Enter для midas): ").strip()
        if not model:
            model = 'midas'
        
        process_single_image(image_path, model_name=model)
    
    elif choice == "3":
        print("\nСоздание тестового изображения...")
        
        # Создаём тестовое изображение
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (300, 300), (0, 0, 255), -1)
        cv2.rectangle(test_img, (350, 150), (550, 350), (0, 255, 0), -1)
        cv2.circle(test_img, (320, 400), 60, (255, 0, 0), -1)
        
        test_path = "test_image.jpg"
        cv2.imwrite(test_path, test_img)
        
        process_single_image(test_path)
    
    else:
        print("Неверный выбор")


if __name__ == "__main__":
    main()
