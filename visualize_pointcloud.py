"""
Визуализатор облаков точек с несколькими бэкендами.
Решает проблему с OpenGL на некоторых системах Windows.
"""

import numpy as np
import os

# Проверка доступных библиотек
OPEN3D_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False
PYVISTA_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    pass


def load_point_cloud(filepath: str):
    """Загрузка облака точек из PLY файла."""
    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}")
        return None, None
    
    if OPEN3D_AVAILABLE:
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        print(f"✓ Загружено {len(points)} точек из {filepath}")
        return points, colors
    else:
        # Ручной парсинг PLY
        points = []
        colors = []
        
        with open(filepath, 'r') as f:
            in_header = True
            vertex_count = 0
            
            for line in f:
                line = line.strip()
                
                if in_header:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line == 'end_header':
                        in_header = False
                else:
                    parts = line.split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(parts) >= 6:
                            colors.append([int(parts[3])/255, int(parts[4])/255, int(parts[5])/255])
        
        points = np.array(points)
        colors = np.array(colors) if len(colors) > 0 else None
        print(f"✓ Загружено {len(points)} точек из {filepath}")
        return points, colors


def visualize_matplotlib(points: np.ndarray, colors: np.ndarray = None, 
                         title: str = "3D Point Cloud", 
                         point_size: float = 1.0,
                         save_path: str = None):
    """
    Визуализация с помощью Matplotlib.
    Работает на всех системах, но менее интерактивна.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib не установлен!")
        return
    
    print("\n" + "="*50)
    print("ВИЗУАЛИЗАЦИЯ (Matplotlib)")
    print("="*50)
    print("Управление:")
    print("  • Левая кнопка мыши - вращение")
    print("  • Правая кнопка - масштаб")
    print("  • Закройте окно для выхода")
    print("="*50 + "\n")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Подготовка цветов
    if colors is not None and len(colors) == len(points):
        c = colors
    else:
        # Цвет по высоте (Z)
        z = points[:, 2]
        c = (z - z.min()) / (z.max() - z.min() + 1e-6)
    
    # Уменьшение количества точек для производительности
    max_points = 50000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_plot = points[indices]
        if isinstance(c, np.ndarray) and len(c) == len(points):
            c = c[indices]
    else:
        points_plot = points
    
    # Отрисовка
    scatter = ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                        c=c, s=point_size, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\n({len(points)} точек)")
    
    # Равные пропорции осей
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.colorbar(scatter, ax=ax, label='Высота (Z)', shrink=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Изображение сохранено: {save_path}")
    
    plt.tight_layout()
    plt.show()


def visualize_open3d_offscreen(points: np.ndarray, colors: np.ndarray = None,
                                save_path: str = "point_cloud_render.png"):
    """
    Рендеринг Open3D в файл (без окна).
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D не установлен!")
        return
    
    print("\nРендеринг в файл...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Цвет по высоте
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        colors_height = np.zeros((len(points), 3))
        colors_height[:, 0] = z_norm  # Red
        colors_height[:, 2] = 1 - z_norm  # Blue
        pcd.colors = o3d.utility.Vector3dVector(colors_height)
    
    # Создание визуализатора
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    vis.add_geometry(pcd)
    
    # Добавляем координатные оси
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.1
    )
    vis.add_geometry(coord_frame)
    
    # Настройка камеры
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Сохранение
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    
    print(f"✓ Рендер сохранён: {save_path}")
    
    # Показываем изображение
    if MATPLOTLIB_AVAILABLE:
        img = plt.imread(save_path)
        plt.figure(figsize=(14, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"3D Point Cloud ({len(points)} точек)")
        plt.show()


def visualize_open3d_legacy(points: np.ndarray, colors: np.ndarray = None):
    """
    Попытка использовать legacy визуализатор Open3D.
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D не установлен!")
        return
    
    print("\n" + "="*50)
    print("ВИЗУАЛИЗАЦИЯ (Open3D)")
    print("="*50)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        colors_height = np.zeros((len(points), 3))
        colors_height[:, 0] = z_norm
        colors_height[:, 2] = 1 - z_norm
        pcd.colors = o3d.utility.Vector3dVector(colors_height)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.1
    )
    
    try:
        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name="3D Point Cloud",
            width=1280,
            height=720
        )
    except Exception as e:
        print(f"Ошибка Open3D визуализации: {e}")
        print("Попробуйте альтернативный метод визуализации.")


def visualize_pyvista(points: np.ndarray, colors: np.ndarray = None):
    """
    Визуализация с PyVista.
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista не установлен! pip install pyvista")
        return
    
    print("\n" + "="*50)
    print("ВИЗУАЛИЗАЦИЯ (PyVista)")
    print("="*50)
    
    # Создание облака точек
    cloud = pv.PolyData(points)
    
    if colors is not None and len(colors) == len(points):
        cloud['RGB'] = (colors * 255).astype(np.uint8)
        scalars = 'RGB'
    else:
        cloud['elevation'] = points[:, 2]
        scalars = 'elevation'
    
    # Визуализация
    plotter = pv.Plotter()
    plotter.add_points(cloud, scalars=scalars, point_size=3, 
                      render_points_as_spheres=True, cmap='viridis')
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


def create_mesh_from_points(points: np.ndarray, colors: np.ndarray = None):
    """
    Создание полигональной сетки из облака точек.
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D не установлен!")
        return None
    
    print("\nСоздание полигональной сетки...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Вычисление нормалей
    print("  Вычисление нормалей...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Ball Pivoting
    print("  Построение mesh (Ball Pivoting)...")
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist * 0.5, avg_dist, avg_dist * 2]
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    print(f"✓ Mesh создан: {len(mesh.vertices)} вершин, {len(mesh.triangles)} треугольников")
    
    return mesh


def main():
    """Главная функция визуализатора."""
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАТОР ОБЛАКОВ ТОЧЕК")
    print("="*60)
    
    # Поиск PLY файлов
    ply_files = []
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.ply'):
                ply_files.append(os.path.join(root, f))
    
    if not ply_files:
        print("\nНе найдено PLY файлов в текущей директории.")
        filepath = input("Введите путь к PLY файлу: ").strip()
    else:
        print("\nНайденные PLY файлы:")
        for i, f in enumerate(ply_files):
            print(f"  {i+1}. {f}")
        
        choice = input("\nВыберите номер файла или введите путь: ").strip()
        
        try:
            idx = int(choice) - 1
            filepath = ply_files[idx]
        except:
            filepath = choice
    
    # Загрузка
    points, colors = load_point_cloud(filepath)
    
    if points is None or len(points) == 0:
        print("Не удалось загрузить облако точек!")
        return
    
    # Статистика
    print(f"\nСтатистика облака точек:")
    print(f"  Количество точек: {len(points)}")
    print(f"  X: от {points[:,0].min():.3f} до {points[:,0].max():.3f}")
    print(f"  Y: от {points[:,1].min():.3f} до {points[:,1].max():.3f}")
    print(f"  Z: от {points[:,2].min():.3f} до {points[:,2].max():.3f}")
    
    # Выбор метода визуализации
    print("\nМетоды визуализации:")
    print("  1. Matplotlib (работает везде)")
    print("  2. Open3D (интерактивный)")
    print("  3. Open3D рендер в файл")
    if PYVISTA_AVAILABLE:
        print("  4. PyVista")
    print("  5. Создать mesh и визуализировать")
    
    method = input("\nВыберите метод (1-5): ").strip()
    
    if method == "1":
        save_img = input("Сохранить изображение? (y/n): ").strip().lower() == 'y'
        save_path = filepath.replace('.ply', '_view.png') if save_img else None
        visualize_matplotlib(points, colors, save_path=save_path)
    
    elif method == "2":
        visualize_open3d_legacy(points, colors)
    
    elif method == "3":
        save_path = filepath.replace('.ply', '_render.png')
        visualize_open3d_offscreen(points, colors, save_path)
    
    elif method == "4" and PYVISTA_AVAILABLE:
        visualize_pyvista(points, colors)
    
    elif method == "5":
        mesh = create_mesh_from_points(points, colors)
        if mesh is not None:
            # Сохранение mesh
            mesh_path = filepath.replace('.ply', '_mesh.ply')
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"✓ Mesh сохранён: {mesh_path}")
            
            # Визуализация
            try:
                o3d.visualization.draw_geometries([mesh], window_name="3D Mesh")
            except:
                print("Ошибка визуализации. Используйте Matplotlib.")
    
    else:
        print("Используем Matplotlib по умолчанию...")
        visualize_matplotlib(points, colors)


if __name__ == "__main__":
    main()
