"""
Реконструкция рельефа местности по одной картинке.

Использует монокулярную оценку глубины (Depth Anything) для создания
3D модели рельефа из обычной фотографии.

Использование:
    python image_terrain_reconstruction.py <путь_к_изображению>
    python image_terrain_reconstruction.py Data/2.jpg
"""

import cv2
import numpy as np
import os
import sys
import argparse
from typing import Optional, Tuple

# Проверка зависимостей
print("=" * 60)
print("РЕКОНСТРУКЦИЯ РЕЛЬЕФА ПО ИЗОБРАЖЕНИЮ")
print("=" * 60)

try:
    import torch
    TORCH_AVAILABLE = True
    device_info = f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    print(f"✓ PyTorch ({device_info})")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch: pip install torch torchvision")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("✗ Transformers: pip install transformers")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("✓ Open3D")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("✗ Open3D: pip install open3d")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("✓ Matplotlib")
except ImportError:
    MATPLOTLIB_AVAILABLE = False

print("=" * 60)


class ImageTerrainReconstructor:
    """
    Реконструкция 3D рельефа из одного изображения.
    
    Пайплайн:
    1. Загрузка изображения
    2. Оценка глубины нейросетью (Depth Anything)
    3. Создание облака точек
    4. Построение mesh (поверхности)
    5. Визуализация и сохранение
    """
    
    DEPTH_MODELS = {
        'small': 'LiheYoung/depth-anything-small-hf',
        'base': 'LiheYoung/depth-anything-base-hf', 
        'large': 'LiheYoung/depth-anything-large-hf',
        'glpn': 'vinvino02/glpn-nyu',
        'midas': 'Intel/dpt-hybrid-midas',
    }
    
    def __init__(self, model_name: str = 'small'):
        """
        Args:
            model_name: модель глубины ('small', 'base', 'large', 'glpn', 'midas')
        """
        self.model_name = model_name
        self.model_id = self.DEPTH_MODELS.get(model_name, model_name)
        self.depth_pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Загружает модель оценки глубины."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("Требуются PyTorch и Transformers!")
        
        print(f"\nЗагрузка модели: {self.model_id}...")
        
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model=self.model_id,
            device=0 if self.device == "cuda" else -1
        )
        print("✓ Модель загружена")
        
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Оценка глубины изображения.
        
        Args:
            image: BGR изображение (OpenCV)
            
        Returns:
            depth_map: карта глубины (float32)
        """
        from PIL import Image
        
        if self.depth_pipe is None:
            self.load_model()
        
        # Конвертация BGR -> RGB -> PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Оценка глубины
        print("Оценка глубины...")
        result = self.depth_pipe(pil_image)
        depth = np.array(result["depth"]).astype(np.float32)
        
        # Нормализация
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        print(f"  Размер карты глубины: {depth.shape}")
        print(f"  Диапазон: {depth.min():.3f} - {depth.max():.3f}")
        
        return depth
    
    def depth_to_point_cloud(self, 
                              image: np.ndarray, 
                              depth: np.ndarray,
                              fx: float = None,
                              fy: float = None,
                              depth_scale: float = 10.0,
                              max_depth: float = 0.98,
                              high_quality: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Преобразует изображение и карту глубины в ПЛОТНОЕ облако точек.
        
        Args:
            image: BGR изображение
            depth: карта глубины (0-1)
            fx, fy: фокусное расстояние (пиксели)
            depth_scale: масштаб глубины для 3D
            max_depth: отсечка дальних точек
            high_quality: использовать улучшенный алгоритм
            
        Returns:
            points: координаты точек (N, 3)
            colors: цвета точек (N, 3) в диапазоне 0-1
        """
        h, w = depth.shape[:2]
        
        # Фокусное расстояние
        if fx is None:
            fx = max(w, h)  # Более реалистичное значение
        if fy is None:
            fy = max(w, h)
            
        cx, cy = w / 2, h / 2
        
        # RGB для цветов
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ===== Предобработка глубины =====
        depth_processed = depth.astype(np.float32).copy()
        
        # 1. Мягкое сглаживание для уменьшения шума (сохраняем края)
        depth_uint8 = (depth_processed * 255).astype(np.uint8)
        depth_smooth = cv2.bilateralFilter(depth_uint8, d=5, sigmaColor=50, sigmaSpace=50)
        depth_processed = depth_smooth.astype(np.float32) / 255.0
        
        # 2. Заполнение дыр медианным фильтром
        depth_median = cv2.medianBlur(depth_uint8, 5)
        # Заполняем только нулевые области
        zero_mask = depth_processed < 0.01
        depth_processed[zero_mask] = depth_median[zero_mask].astype(np.float32) / 255.0
        
        # ===== Создание плотного облака точек =====
        # Используем ВСЕ пиксели без пропусков
        
        # Сетка координат
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Глубина: инвертируем (в depth map ближе = светлее = больше значение)
        # Переворачиваем так, чтобы ближние объекты были ближе к камере
        z = (1.0 - depth_processed) * depth_scale
        
        # 3D координаты по модели пинхол-камеры
        x = (u_coords - cx) * z / fx
        y = (v_coords - cy) * z / fy
        
        # Формируем массивы
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
        
        # Минимальная фильтрация - только явно невалидные точки
        valid = (
            (depth_processed.flatten() > 0.005) &    # Почти нулевая глубина
            (depth_processed.flatten() < max_depth)   # Слишком далеко
        )
        
        points = points[valid].astype(np.float32)
        colors = colors[valid].astype(np.float32)
        
        print(f"  Создано точек: {len(points)} из {h*w} пикселей ({100*len(points)/(h*w):.1f}%)")
        
        return points, colors
    
    def create_mesh(self, 
                    points: np.ndarray, 
                    colors: np.ndarray,
                    method: str = 'poisson',
                    depth: int = 9,
                    high_quality: bool = True) -> 'o3d.geometry.TriangleMesh':
        """
        Создаёт mesh из облака точек.
        
        Args:
            points: координаты точек
            colors: цвета точек
            method: 'poisson' или 'ball_pivoting'
            depth: глубина для Poisson реконструкции
            high_quality: более строгая фильтрация и мелкий voxel
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D не установлен!")
            return None
            
        print(f"\nСоздание mesh ({method})...")
        
        # Создаём облако точек Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print(f"  Исходное облако: {len(pcd.points)} точек")
        
        # Даунсэмплинг - ОЧЕНЬ мягкий чтобы сохранить детали
        voxel_size = 0.02  # Фиксированный небольшой размер
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  После даунсэмплинга: {len(pcd.points)} точек")
        
        # Мягкое удаление только явных выбросов
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        print(f"  После фильтрации: {len(pcd.points)} точек")
        
        # Оценка нормалей
        print("  Оценка нормалей...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., -10.]))
        
        # Построение mesh
        if method == 'poisson':
            print(f"  Poisson реконструкция (depth={depth})...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
            
            # Мягкое удаление низкоплотных вершин (только 1%)
            densities = np.asarray(densities)
            threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        elif method == 'ball_pivoting':
            print("  Ball pivoting...")
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        # Финализация
        mesh.compute_vertex_normals()
        
        # Перенос цветов на mesh
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        print(f"✓ Mesh создан: {len(mesh.vertices)} вершин, {len(mesh.triangles)} треугольников")
        
        return mesh, pcd
    
    def visualize(self, 
                  image: np.ndarray, 
                  depth: np.ndarray,
                  pcd: 'o3d.geometry.PointCloud' = None,
                  mesh: 'o3d.geometry.TriangleMesh' = None):
        """Визуализация результатов."""
        
        # 2D визуализация (matplotlib)
        if MATPLOTLIB_AVAILABLE:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Исходное изображение
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Исходное изображение')
            axes[0].axis('off')
            
            # Карта глубины
            im = axes[1].imshow(depth, cmap='plasma')
            axes[1].set_title('Карта глубины')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.show()
        
        # 3D визуализация (Open3D)
        if OPEN3D_AVAILABLE and (pcd is not None or mesh is not None):
            print("\nЗапуск 3D визуализации...")
            print("Управление: мышь для вращения, колёсико для масштаба")
            
            geometries = []
            if pcd is not None:
                geometries.append(pcd)
            if mesh is not None:
                geometries.append(mesh)
            
            # Координатные оси
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            geometries.append(axes)
            
            o3d.visualization.draw_geometries(
                geometries,
                window_name="3D Реконструкция рельефа",
                width=1280,
                height=720
            )
    
    def save_results(self, 
                     output_dir: str,
                     base_name: str,
                     depth: np.ndarray,
                     pcd: 'o3d.geometry.PointCloud' = None,
                     mesh: 'o3d.geometry.TriangleMesh' = None):
        """Сохраняет результаты."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Карта глубины
        depth_vis = (depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
        depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
        cv2.imwrite(depth_path, depth_colored)
        print(f"✓ Карта глубины: {depth_path}")
        
        if OPEN3D_AVAILABLE:
            # Облако точек
            if pcd is not None:
                pcd_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                print(f"✓ Облако точек: {pcd_path}")
            
            # Mesh
            if mesh is not None:
                mesh_path = os.path.join(output_dir, f"{base_name}_mesh.obj")
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                print(f"✓ Mesh: {mesh_path}")
    
    def process(self, 
                image_path: str,
                output_dir: str = "Data",
                visualize: bool = True,
                depth_scale: float = 10.0,
                mesh_depth: int = 9,
                high_quality: bool = True) -> dict:
        """
        Полный пайплайн реконструкции.
        
        Args:
            image_path: путь к изображению
            output_dir: папка для сохранения
            visualize: показать визуализацию
            depth_scale: масштаб глубины
            mesh_depth: глубина Poisson реконструкции
            high_quality: использовать улучшенный алгоритм для чёткости
            
        Returns:
            dict с результатами
        """
        print(f"\n{'='*60}")
        print(f"Обработка: {image_path}")
        print(f"Режим: {'ВЫСОКОЕ КАЧЕСТВО' if high_quality else 'стандартный'}")
        print(f"{'='*60}")
        
        # 1. Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {image_path}")
        
        print(f"Размер изображения: {image.shape[1]}x{image.shape[0]}")
        
        # 2. Оценка глубины
        depth = self.estimate_depth(image)
        
        # 3. Создание облака точек (с улучшениями)
        points, colors = self.depth_to_point_cloud(
            image, depth, depth_scale=depth_scale, high_quality=high_quality
        )
        
        # 4. Создание mesh
        mesh, pcd = self.create_mesh(points, colors, depth=mesh_depth, high_quality=high_quality)
        
        # 5. Сохранение
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.save_results(output_dir, base_name, depth, pcd, mesh)
        
        # 6. Визуализация
        if visualize:
            self.visualize(image, depth, pcd, mesh)
        
        return {
            'image': image,
            'depth': depth,
            'points': points,
            'colors': colors,
            'pcd': pcd,
            'mesh': mesh
        }


def main():
    parser = argparse.ArgumentParser(
        description='Реконструкция 3D рельефа по изображению'
    )
    parser.add_argument(
        'image', 
        nargs='?',
        default='Data/2.jpg',
        help='Путь к изображению'
    )
    parser.add_argument(
        '--output', '-o',
        default='Data',
        help='Папка для сохранения результатов'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['small', 'base', 'large', 'glpn', 'midas'],
        default='small',
        help='Модель оценки глубины'
    )
    parser.add_argument(
        '--depth-scale', '-s',
        type=float,
        default=10.0,
        help='Масштаб глубины для 3D'
    )
    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='Отключить визуализацию'
    )
    parser.add_argument(
        '--high-quality', '--hq',
        action='store_true',
        default=True,
        help='Использовать улучшенный алгоритм для чёткости (по умолчанию)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Быстрый режим (меньше точек, быстрее обработка)'
    )
    
    args = parser.parse_args()
    
    # Проверка файла
    if not os.path.exists(args.image):
        print(f"Файл не найден: {args.image}")
        print("\nДоступные изображения в Data/:")
        if os.path.exists("Data"):
            for f in os.listdir("Data"):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    print(f"  - Data/{f}")
        sys.exit(1)
    
    # Проверка зависимостей
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        print("\nТребуемые зависимости не установлены!")
        print("Установите: pip install torch torchvision transformers")
        sys.exit(1)
    
    # Режим качества
    high_quality = not args.fast
    
    print(f"\nРежим: {'ВЫСОКОЕ КАЧЕСТВО' if high_quality else 'БЫСТРЫЙ'}")
    
    # Запуск реконструкции
    reconstructor = ImageTerrainReconstructor(model_name=args.model)
    
    result = reconstructor.process(
        image_path=args.image,
        output_dir=args.output,
        visualize=not args.no_vis,
        depth_scale=args.depth_scale,
        high_quality=high_quality
    )
    
    print(f"\n{'='*60}")
    print("✓ Реконструкция завершена!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
