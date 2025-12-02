"""
Интерактивная 3D визуализация облаков точек и mesh.
Открывается в браузере с возможностью вращения, масштабирования.
"""

import numpy as np
import os
import sys

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly не установлен: pip install plotly")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D не установлен: pip install open3d")


def visualize_point_cloud_plotly(filepath: str, max_points: int = 100000, point_size: float = 1.5):
    """
    Интерактивная визуализация облака точек в браузере.
    
    Управление:
    - Левая кнопка мыши: вращение
    - Правая кнопка / колёсико: масштаб
    - Shift + мышь: перемещение
    """
    if not PLOTLY_AVAILABLE or not OPEN3D_AVAILABLE:
        print("Требуются plotly и open3d!")
        return
    
    print(f"Загрузка: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    print(f"Загружено: {len(points)} точек")
    
    # Субдискретизация для плавной работы
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
        print(f"Визуализация: {max_points} точек (субдискретизация)")
    
    # Подготовка цветов
    if colors is not None:
        color_values = ['rgb({},{},{})'.format(
            int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
        marker_dict = dict(size=point_size, color=color_values, opacity=0.8)
    else:
        marker_dict = dict(
            size=point_size, 
            color=points[:, 2],
            colorscale='Viridis',
            colorbar=dict(title='Глубина'),
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
        title=f'3D Облако точек ({len(points)} точек)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (глубина)',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=-1, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    print("Открытие в браузере...")
    fig.show()
    
    return fig


def visualize_mesh_plotly(filepath: str):
    """
    Интерактивная визуализация mesh в браузере.
    """
    if not PLOTLY_AVAILABLE or not OPEN3D_AVAILABLE:
        print("Требуются plotly и open3d!")
        return
    
    print(f"Загрузка mesh: {filepath}")
    mesh = o3d.io.read_triangle_mesh(filepath)
    mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Цвета вершин (если есть)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        vertex_colors = ['rgb({},{},{})'.format(
            int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
    else:
        vertex_colors = vertices[:, 2]  # По высоте
    
    print(f"Mesh: {len(vertices)} вершин, {len(triangles)} треугольников")
    
    # Создание mesh
    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=vertices[:, 2] if not mesh.has_vertex_colors() else None,
        colorscale='Viridis' if not mesh.has_vertex_colors() else None,
        opacity=1.0,
        flatshading=True,
        lighting=dict(
            ambient=0.5,
            diffuse=0.8,
            specular=0.2,
            roughness=0.5
        ),
        name='Mesh'
    )])
    
    fig.update_layout(
        title=f'3D Mesh ({len(vertices)} вершин, {len(triangles)} треугольников)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=-1, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    print("Открытие в браузере...")
    fig.show()
    
    return fig


def visualize_both(pcd_path: str, mesh_path: str, max_points: int = 50000):
    """
    Визуализация облака точек и mesh рядом.
    """
    if not PLOTLY_AVAILABLE or not OPEN3D_AVAILABLE:
        return
    
    # Загрузка облака точек
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    # Загрузка mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Смещение mesh для отображения рядом
    offset = points[:, 0].max() - points[:, 0].min() + 2
    vertices_shifted = vertices.copy()
    vertices_shifted[:, 0] += offset
    
    # Создание фигуры
    fig = go.Figure()
    
    # Облако точек
    if colors is not None:
        color_values = ['rgb({},{},{})'.format(
            int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
    else:
        color_values = points[:, 2]
    
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=1, color=color_values if colors is not None else points[:, 2],
                   colorscale='Viridis' if colors is None else None),
        name='Облако точек'
    ))
    
    # Mesh
    fig.add_trace(go.Mesh3d(
        x=vertices_shifted[:, 0],
        y=vertices_shifted[:, 1],
        z=vertices_shifted[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=vertices[:, 2],
        colorscale='Plasma',
        opacity=0.9,
        name='Mesh'
    ))
    
    fig.update_layout(
        title='Сравнение: Облако точек vs Mesh',
        scene=dict(aspectmode='data'),
        width=1600,
        height=900
    )
    
    fig.show()
    return fig


def main():
    """Главная функция."""
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНАЯ 3D ВИЗУАЛИЗАЦИЯ")
    print("="*60)
    
    # Поиск файлов
    ply_files = []
    obj_files = []
    
    for root, dirs, files in os.walk('.'):
        for f in files:
            path = os.path.join(root, f)
            if f.endswith('.ply'):
                ply_files.append(path)
            elif f.endswith('.obj'):
                obj_files.append(path)
    
    print("\nНайденные файлы:")
    all_files = []
    
    if ply_files:
        print("\nОблака точек (.ply):")
        for i, f in enumerate(ply_files):
            print(f"  {len(all_files)+1}. {f}")
            all_files.append(('ply', f))
    
    if obj_files:
        print("\nMesh (.obj):")
        for i, f in enumerate(obj_files):
            print(f"  {len(all_files)+1}. {f}")
            all_files.append(('obj', f))
    
    if not all_files:
        print("Файлы не найдены!")
        return
    
    print("\nОпции:")
    print("  V. Визуализировать выбранный файл")
    print("  B. Показать облако точек и mesh вместе")
    
    choice = input("\nВыберите номер файла: ").strip()
    
    try:
        idx = int(choice) - 1
        file_type, filepath = all_files[idx]
        
        if file_type == 'ply':
            # Проверяем, mesh это или облако точек
            mesh = o3d.io.read_triangle_mesh(filepath)
            if len(mesh.triangles) > 0:
                print("Это mesh файл")
                visualize_mesh_plotly(filepath)
            else:
                print("Это облако точек")
                visualize_point_cloud_plotly(filepath)
        else:
            visualize_mesh_plotly(filepath)
            
    except ValueError:
        if choice.lower() == 'b':
            # Найти пару облако/mesh
            pcd_file = None
            mesh_file = None
            
            for ft, fp in all_files:
                if 'pointcloud' in fp.lower() and fp.endswith('.ply'):
                    pcd_file = fp
                elif 'mesh' in fp.lower():
                    mesh_file = fp
            
            if pcd_file and mesh_file:
                visualize_both(pcd_file, mesh_file)
            else:
                print("Не найдены файлы pointcloud и mesh")
        else:
            print("Неверный выбор")


if __name__ == "__main__":
    main()
