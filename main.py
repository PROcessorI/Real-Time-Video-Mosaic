import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from PIL import Image, ImageDraw, ImageFont
import argparse
import sys


class VideMosaic:
    # def __init__(self, first_image, output_height_times=2, output_width_times=4, detector_type="sift"):
    def __init__(self, first_image, output_height_times=3, output_width_times=1.2, detector_type="sift", show_intermediate=True, output_dir=None, visualize=True):
        """Этот класс обрабатывает каждый кадр и генерирует панораму.

        Args:
            first_image (изображение для первого кадра): первое изображение для инициализации размера вывода
            output_height_times (int, optional): определяет высоту вывода на основе высоты входного изображения. По умолчанию 3.
            output_width_times (int, optional): определяет ширину вывода на основе ширины входного изображения. По умолчанию 1.2.
            detector_type (str, optional): детектор для обнаружения особенностей. Может быть "sift" или "orb". По умолчанию "sift".
            show_intermediate (bool, optional): показывать ли промежуточные окна OpenCV во время обработки. По умолчанию True.
            output_dir (str, optional): каталог для сохранения выходных файлов. По умолчанию None.
            visualize (bool, optional): показывать ли визуализацию сопоставления ключевых точек. По умолчанию True.
        """
        self.detector_type = detector_type
        self.show_intermediate = show_intermediate
        self.output_dir = output_dir
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(700)
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(700)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.visualize = visualize

        # Инициализировать модель YOLO для обнаружения с большей моделью для лучшей точности
        try:
            self.model = YOLO('yolo11n.pt')  # YOLOv11 nano модель
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить модель YOLO: {e}")
            self.model = None

        # Инициализировать окно OpenCV для промежуточной визуализации, если включено
        # if self.show_intermediate:
        #     cv2.namedWindow('Mosaic Progress', cv2.WINDOW_NORMAL)
        #     cv2.namedWindow('Current Frame', cv2.WINDOW_AUTOSIZE)
        #     print("Окна OpenCV 'Mosaic Progress' и 'Current Frame' созданы")

        self.process_first_frame(first_image)

        self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), int(
            output_width_times*first_image.shape[1]), first_image.shape[2]))

        # смещение
        # self.w_offset = int(self.output_img.shape[0]/2 - first_image.shape[0]/2)
        # self.h_offset = int(self.output_img.shape[1]/2 - first_image.shape[1]/2)
        self.w_offset = int(self.output_img.shape[0]/1 - first_image.shape[0]/1)
        self.h_offset = int(self.output_img.shape[1]/2 - first_image.shape[1]/2)

        self.output_img[self.w_offset:self.w_offset+first_image.shape[0],
                        self.h_offset:self.h_offset+first_image.shape[1], :] = first_image

        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

    def process_first_frame(self, first_image):
        """обрабатывает первый кадр для обнаружения особенностей и описания

        Args:
            first_image (cv2 изображение/np массив): первое изображение для обнаружения особенностей
        """
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def detect_people(self, frame):
        if self.model is None:
            return []
        # Использовать более высокий порог уверенности и оптимизированные параметры для лучшего качества
        results = self.model.predict(
            frame, 
            classes=[0],  # класс 0 - 'person'
            conf=0.5,      # порог уверенности - только обнаружения с 50%+ уверенностью
            iou=0.45,      # порог IoU для NMS - уменьшает перекрывающиеся боксы
            imgsz=640,     # размер изображения для обнаружения - больше для лучшего качества
            verbose=False  # уменьшить вывод в консоль
        )
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    def detect_objects(self, frame):
        if self.model is None:
            return []
        # Изменить размер кадра до стандартного размера для лучшего обнаружения
        original_shape = frame.shape[:2]
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        # Использовать оптимизированные параметры для максимального качества обнаружения
        results = self.model.predict(
            resized,
            conf=0.4,      # порог уверенности - фильтровать низкоуверенные обнаружения
            iou=0.45,      # порог IoU для NMS - уменьшает дублирующиеся обнаружения
            imgsz=640,     # размер изображения для обнаружения - больше для лучшей точности
            verbose=False  # уменьшить вывод в консоль
        )
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Масштабировать обратно к оригинальному размеру
                scale_x = original_shape[1] / 640
                scale_y = original_shape[0] / 640
                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])  # Получить оценку уверенности
                class_name = self.model.names[class_id] if hasattr(self.model, 'names') else str(class_id)
                detections.append({
                    'class': class_name, 
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence
                })
        
        # Добавить детекцию огня и дыма по цвету
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(frame)
        
        # Детекция огня - комбинация HSV и BGR анализа
        lower_fire1 = np.array([0, 80, 100])
        upper_fire1 = np.array([15, 255, 255])
        mask_fire1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        lower_fire2 = np.array([165, 80, 100])
        upper_fire2 = np.array([180, 255, 255])
        mask_fire2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        mask_fire = cv2.bitwise_or(mask_fire1, mask_fire2)
        
        # Дополнительная проверка: красный канал > синий и зеленый
        fire_bgr_mask = ((r > g + 20) & (r > b + 30) & (r > 100)).astype(np.uint8) * 255
        mask_fire = cv2.bitwise_and(mask_fire, fire_bgr_mask)
        
        # Морфологическая обработка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_CLOSE, kernel)
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel)
        
        # Детекция дыма - низкая насыщенность, средняя яркость
        lower_smoke = np.array([0, 0, 80])
        upper_smoke = np.array([180, 50, 230])
        mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        # Дополнительная проверка: близкие значения RGB (серый цвет)
        smoke_bgr_mask = ((np.abs(r.astype(int) - g.astype(int)) < 30) & 
                         (np.abs(g.astype(int) - b.astype(int)) < 30) & 
                         (r > 60)).astype(np.uint8) * 255
        mask_smoke = cv2.bitwise_and(mask_smoke, smoke_bgr_mask)
        
        # Морфологическая обработка
        mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_CLOSE, kernel)
        mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_OPEN, kernel)
        
        # Найти контуры для огня
        fire_detections = []
        contours_fire, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_fire:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                # Проверка плотности заполнения контура
                density = area / (w * h) if w * h > 0 else 0
                if density > 0.3:
                    fire_detections.append({
                        'class': 'fire',
                        'box': (x, y, x + w, y + h),
                        'confidence': 0.8,
                        'area': area
                    })
        
        # Найти контуры для дыма
        smoke_detections = []
        contours_smoke, _ = cv2.findContours(mask_smoke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_smoke:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                # Проверка плотности и формы
                density = area / (w * h) if w * h > 0 else 0
                if density > 0.2:
                    smoke_detections.append({
                        'class': 'smoke',
                        'box': (x, y, x + w, y + h),
                        'confidence': 0.7,
                        'area': area
                    })
        
        # Применить NMS для удаления перекрывающихся боксов
        def apply_nms(dets, iou_threshold=0.5):
            if not dets:
                return []
            dets_sorted = sorted(dets, key=lambda x: x['area'], reverse=True)
            keep = []
            while dets_sorted:
                current = dets_sorted.pop(0)
                keep.append(current)
                x1, y1, x2, y2 = current['box']
                dets_sorted = [d for d in dets_sorted if 
                              calculate_iou((x1, y1, x2, y2), d['box']) < iou_threshold]
            return keep
        
        def calculate_iou(box1, box2):
            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2
            xi1, yi1 = max(x1, x3), max(y1, y3)
            xi2, yi2 = min(x2, x4), min(y2, y4)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x4 - x3) * (y4 - y3)
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0
        
        # Применить NMS и добавить к результатам
        for det in apply_nms(fire_detections, 0.4):
            detections.append({'class': det['class'], 'box': det['box'], 'confidence': det['confidence']})
        for det in apply_nms(smoke_detections, 0.4):
            detections.append({'class': det['class'], 'box': det['box'], 'confidence': det['confidence']})
        
        return detections

    def match(self, des_cur, des_prev):
        """сопоставляет дескрипторы

        Args:
            des_cur (np массив): дескрипторы текущего кадра
            des_prev (np массив): дескрипторы предыдущего кадра

        Returns:
            массив: массив соответствий между дескрипторами
        """
        # сопоставление
        if self.detector_type == "sift":
            pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
            matches = []
            for m, n in pair_matches:
                if m.distance < 0.7*n.distance:
                    matches.append(m)

        elif self.detector_type == "orb":
            matches = self.bf.match(des_cur, des_prev)

        # Сортировать их в порядке их расстояния.
        matches = sorted(matches, key=lambda x: x.distance)

        # получить максимум 20 лучших соответствий
        # matches = matches[:min(len(matches), 20)]
        # Нарисовать первые 10 соответствий.
        if self.visualize:
            match_img = cv2.drawMatches(self.frame_cur, self.kp_cur, self.frame_prev, self.kp_prev, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
            cv2.imshow('matches', match_img)
        return matches

    def process_frame(self, frame_cur, frame_count):
        """получает изображение и обрабатывает его для мозаики

        Args:
            frame_cur (np массив): вход текущего кадра для мозаики
        """
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        self.matches = self.match(self.des_cur, self.des_prev)

        if len(self.matches) < 4:
            return

        self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
        self.H = np.matmul(self.H_old, self.H)
        # TODO: проверить на плохую гомографию

        self.warp(self.frame_cur, self.H)

        # Обновить промежуточные окна, если включено
        # if self.show_intermediate:
        #     cv2.imshow('Mosaic Progress', self.output_img.astype(np.uint8))
        #     cv2.imshow('Current Frame', self.frame_cur)

        # подготовка цикла
        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur

    @ staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        """получает два соответствия и рассчитывает гомографию между двумя изображениями

        Args:
            image_1_kp (np массив): ключевые точки изображения 1
            image_2_kp (np массив): ключевые точки изображения 2
            matches (np массив): соответствия между ключевыми точками в изображении 1 и 2

        Returns:
            np массив формы [3,3]: матрица гомографии
        """
        # взято из https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        homography, mask = cv2.findHomography(
            image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

        return homography

    def warp(self, frame_cur, H):
        """деформирует текущий кадр на основе рассчитанной гомографии H

        Args:
            frame_cur (np массив): текущий кадр
            H (np массив формы [3,3]): матрица гомографии

        Returns:
            np массив: выходное изображение мозаики
        """
        warped_img = cv2.warpPerspective(frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR)

        transformed_corners = self.get_transformed_corners(frame_cur, H)
        # Убрано рисование черной границы: warped_img = self.draw_border(warped_img, transformed_corners)

        # Применить улучшенный блендинг с плавными переходами
        # Создать маску для нового кадра (одноканальную)
        mask_new = np.any(warped_img > 0, axis=2).astype(np.uint8) * 255
        mask_old = np.any(self.output_img > 0, axis=2).astype(np.uint8) * 255
        
        # Найти области перекрытия
        overlap = cv2.bitwise_and(mask_new, mask_old)
        
        # Создать весовую маску с расстоянием от краёв для плавного перехода
        if np.any(overlap):
            # Вычислить расстояние от краёв для обеих масок
            dist_new = cv2.distanceTransform(mask_new, cv2.DIST_L2, 5)
            dist_old = cv2.distanceTransform(mask_old, cv2.DIST_L2, 5)
            
            # Нормализовать расстояния для весов
            dist_sum = dist_new + dist_old + 1e-6
            weight_new = dist_new / dist_sum
            weight_old = dist_old / dist_sum
            
            # Применить сильное размытие для очень плавных переходов
            weight_new = cv2.GaussianBlur(weight_new.astype(np.float32), (51, 51), 0)
            weight_old = cv2.GaussianBlur(weight_old.astype(np.float32), (51, 51), 0)
            
            # Расширить до 3 каналов
            weight_new_3ch = np.stack([weight_new, weight_new, weight_new], axis=2)
            weight_old_3ch = np.stack([weight_old, weight_old, weight_old], axis=2)
            
            # Плавное смешивание с учётом расстояния от краёв
            blended = (self.output_img.astype(np.float32) * weight_old_3ch + 
                      warped_img.astype(np.float32) * weight_new_3ch)
            
            # Применить блендинг только в области перекрытия
            overlap_3ch = np.stack([overlap, overlap, overlap], axis=2) > 0
            self.output_img = np.where(overlap_3ch, blended.astype(np.uint8), self.output_img)
            
            # В неперекрывающихся областях просто заменить
            non_overlap_new = cv2.bitwise_and(mask_new, cv2.bitwise_not(overlap)) > 0
            non_overlap_3ch = np.stack([non_overlap_new, non_overlap_new, non_overlap_new], axis=2)
            self.output_img = np.where(non_overlap_3ch, warped_img, self.output_img)
        else:
            # Нет перекрытия, просто заменить
            self.output_img[warped_img > 0] = warped_img[warped_img > 0]
        
        output_temp = np.copy(self.output_img)
        output_temp = self.draw_border(output_temp, transformed_corners, color=(0, 0, 255))
        
        if self.visualize:
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output',  output_temp/255.)

        return self.output_img

    @ staticmethod
    def get_transformed_corners(frame_cur, H):
        """находит угол текущего кадра после деформации

        Args:
            frame_cur (np массив): текущий кадр
            H (np массив формы [3,3]): матрица гомографии

        Returns:
            [np массив]: список из 4 угловых точек после деформации
        """
        corner_0 = np.array([0, 0])
        corner_1 = np.array([frame_cur.shape[1], 0])
        corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner_3 = np.array([0, frame_cur.shape[0]])

        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = np.array(transformed_corners, dtype=np.int32)
        # mask = np.zeros(shape=(output.shape[0], output.shape[1], 1))
        # cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
        # cv2.imshow('mask', mask)

        return transformed_corners

    def draw_border(self, image, corners, color=(0, 0, 0)):
        """Эта функция рисует прямоугольную границу

        Args:
            image ([type]): текущий выход мозаики
            corners (np массив): список угловых точек
            color (tuple, optional): цвет линий границы. По умолчанию (0, 0, 0).

        Returns:
            np массив: выходное изображение с границей
        """
        for i in range(corners.shape[1]-1, -1, -1):
            cv2.line(image, tuple(corners[0, i, :]), tuple(corners[0, i-1, :]), thickness=5, color=color)
        return image


def crop_black_areas(image, threshold=15, margin=5):
    """Обрезать черные области и темные артефакты на краях.
    
    Args:
        image: Входное изображение
        threshold: Порог яркости (пиксели < threshold считаются черными)
        margin: Отступ от краев после обрезки
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Использовать порог для игнорирования очень темных пикселей
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    # Добавить отступ от краев для удаления тонких артефактов
    x = max(0, x + margin)
    y = max(0, y + margin)
    w = min(image.shape[1] - x, w - 2 * margin)
    h = min(image.shape[0] - y, h - 2 * margin)
    return image[y:y+h, x:x+w]


def scale_to_screen(image, target_w=None, target_h=None):
    """Масштабировать изображение до целевого размера (по умолчанию размер первичного экрана на Windows) с сохранением соотношения сторон.

    Возвращает масштабированное изображение (может быть больше оригинала).
    """
    ih, iw = image.shape[0], image.shape[1]
    # Попытаться получить размер первичного экрана Windows
    screen_w = target_w
    screen_h = target_h
    if screen_w is None or screen_h is None:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
        except Exception:
            # резерв
            screen_w, screen_h = 1920, 1080

    # Вычислить масштаб, сохраняя соотношение
    scale = min(max(1.0, screen_w / float(iw)), max(1.0, screen_h / float(ih)))
    # Использовать минимальный коэффициент увеличения, который помещает изображение на экран в хотя бы одном измерении
    # но сохранить соотношение сторон: выбрать масштаб, такой что изображение помещается на экран в хотя бы одном измерении
    scale_w = screen_w / float(iw)
    scale_h = screen_h / float(ih)
    scale = min(scale_w, scale_h)
    if scale <= 0:
        scale = 1.0

    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def draw_dotted_line(img, pt1, pt2, color, thickness):
    dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
    num_dots = max(1, int(dist / 10))
    for i in range(num_dots):
        t = i / num_dots
        px = int(pt1[0] + t * (pt2[0] - pt1[0]))
        py = int(pt1[1] + t * (pt2[1] - pt1[1]))
        cv2.circle(img, (px, py), thickness, color, -1)


def analyze_for_navigation(frame, detections, start_point=None, compute_paths=True):
    """Простой анализ для навигации: отметить препятствия и нарисовать пути к объектам на одном кадре.

    Args:
        frame (np массив): последний кадр
        detections (list): список обнаруженных объектов
        start_point: начальная точка для навигации
        compute_paths: вычислять ли пути к зданиям (медленно для изображений)

    Returns:
        np массив: кадр с отмеченными препятствиями и путями
    """
    labels_to_draw = []
    # Преобразовать в HSV для детекции огня и дыма по цветам
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создать маску препятствий на основе обнаруженных объектов
    obstacles = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Добавить буфер вокруг обнаруженных объектов для более безопасных препятствий
    # Расширенный список классов и увеличенные буферы
    for det in detections:
        if det['class'] in ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 
                           'dog', 'horse', 'cat', 'bird', 'cow', 'sheep',
                           'smoke', 'fire', 'building']:
            x1, y1, x2, y2 = det['box']
            # Увеличенный буфер безопасности в зависимости от типа объекта
            if det['class'] in ['fire', 'smoke']:
                buffer = 30  # Больший буфер для опасных объектов
            elif det['class'] in ['car', 'truck', 'bus']:
                buffer = 20  # Средний буфер для транспорта
            else:
                buffer = 15  # Стандартный буфер
            
            cv2.rectangle(obstacles, 
                         (max(0, x1-buffer), max(0, y1-buffer)), 
                         (min(obstacles.shape[1], x2+buffer), min(obstacles.shape[0], y2+buffer)), 
                         255, -1)

    # Добавить цветовую детекцию огня и дыма для препятствий (мягкие пороги для карты)
    # Эти маски нужны для визуализации красных контуров препятствий, не для боксов
    b_nav, g_nav, r_nav = cv2.split(frame)
    
    # Огонь - мягкие пороги для карты препятствий
    lower_fire1 = np.array([0, 100, 100])
    upper_fire1 = np.array([15, 255, 255])
    mask_fire1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    lower_fire2 = np.array([165, 100, 100])
    upper_fire2 = np.array([180, 255, 255])
    mask_fire2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    mask_fire = cv2.bitwise_or(mask_fire1, mask_fire2)
    
    # Простая BGR проверка для огня
    fire_bgr_mask = ((r_nav > g_nav + 20) & (r_nav > b_nav + 30) & (r_nav > 100)).astype(np.uint8) * 255
    mask_fire = cv2.bitwise_and(mask_fire, fire_bgr_mask)
    
    # Морфологическая обработка
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_CLOSE, kernel)
    mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel)
    obstacles = cv2.bitwise_or(obstacles, mask_fire)
    
    # Дым - мягкие пороги для карты препятствий
    lower_smoke = np.array([0, 0, 80])
    upper_smoke = np.array([180, 50, 230])
    mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)
    
    # Простая BGR проверка для дыма
    smoke_bgr_mask = ((np.abs(r_nav.astype(int) - g_nav.astype(int)) < 30) &
                     (np.abs(g_nav.astype(int) - b_nav.astype(int)) < 30) &
                     (r_nav > 70) & (g_nav > 70) & (b_nav > 70)).astype(np.uint8) * 255
    mask_smoke = cv2.bitwise_and(mask_smoke, smoke_bgr_mask)
    
    # Морфологическая обработка
    mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_CLOSE, kernel)
    mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_OPEN, kernel)
    obstacles = cv2.bitwise_or(obstacles, mask_smoke)

    # Расширить препятствия для большей безопасности
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    obstacles = cv2.dilate(obstacles, kernel, iterations=2)

    # Отметить препятствия как контуры на копии кадра
    nav_map = frame.copy()
    contours, _ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(nav_map, contours, -1, (0, 0, 255), 2)  # Красные контуры для препятствий

    # точка по умолчанию (нижний-центр) если не предоставлена GUI
    default_start = (frame.shape[1] // 2, frame.shape[0] - 50)
    if start_point is not None:
        start_x, start_y = start_point
    else:
        start_x, start_y = default_start

    # Отметить начальную позицию
    cv2.circle(nav_map, (start_x, start_y), 10, (255, 255, 255), -1)  # Белый круг для старта

    # Использовать PIL для добавления текста с поддержкой русского
    pil_img = Image.fromarray(cv2.cvtColor(nav_map, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    # Попытка загрузить TTF-шрифт с поддержкой кириллицы (Windows). Если не найден - fallback на встроенный.
    font = None
    font_paths = [
        "arial.ttf",  # PIL может найти системные шрифты по имени
        "verdana.ttf",
        "tahoma.ttf",
        "times.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\times.ttf",
    ]
    for p in font_paths:
        try:
            font = ImageFont.truetype(p, 20)
            break
        except Exception:
            continue

    if font is None:
        # последний резерв: шрифт по умолчанию (ограниченная кириллица возможна / может показывать '?')
        try:
            font = ImageFont.load_default()
            print("Предупреждение: TTF шрифт не найден; используется шрифт по умолчанию, который может не поддерживать кириллицу полностью.")
        except Exception:
            font = None

    # Если шрифт все еще None, PIL будет использовать fallback при рисовании, но логировать для пользователя.
    if font is None:
        print("Предупреждение: Шрифт недоступен для рисования текста. Русский текст может не отображаться.")

    # Нарисовать легенду и метку старта (PIL обрабатывает Unicode если шрифт поддерживает)
    try:
        draw.text((10, 30), "Красные контуры: препятствия", fill=(255, 0, 0), font=font)
        draw.text((10, 60), "Зелёные линии: пути к объектам", fill=(0, 255, 0), font=font)
        draw.text((10, 90), "Жёлтые прямоугольники: обнаруженные объекты", fill=(255, 255, 0), font=font)
    except Exception as e:
        # Как изящный fallback, попробовать текст OpenCV (может показывать '?') и напечатать ошибку
        print(f"Ошибка рисования с шрифтом PIL: {e}")
        cv2.putText(nav_map, "Старт", (start_x + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(nav_map, "Красные контуры: препятствия", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(nav_map, "Зелёные линии: пути к объектам", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(nav_map, "Жёлтые прямоугольники: обнаруженные объекты", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    nav_map = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Отметить объекты на карте без рисования маршрутов
    # Словарь для перевода названий классов на русский
    class_names_ru = {
        'person': 'Человек',
        'car': 'Машина',
        'truck': 'Грузовик',
        'bus': 'Автобус',
        'motorcycle': 'Мотоцикл',
        'bicycle': 'Велосипед',
        'dog': 'Собака',
        'horse': 'Лошадь',
        'cat': 'Кот',
        'bird': 'Птица',
        'cow': 'Корова',
        'sheep': 'Овца',
        'smoke': 'Дым',
        'fire': 'Огонь',
        'building': 'Здание'
    }
    
    for det in detections:
        if det['class'] in class_names_ru:  # Отображать все известные объекты
            x1, y1, x2, y2 = det['box']
            # Цвет в зависимости от опасности
            if det['class'] in ['fire', 'smoke']:
                color = (0, 0, 255)  # Красный для опасных объектов
            elif det['class'] in ['car', 'truck', 'bus', 'motorcycle']:
                color = (0, 165, 255)  # Оранжевый для транспорта
            else:
                color = (0, 255, 255)  # Желтый для остальных
            
            cv2.rectangle(nav_map, (x1, y1), (x2, y2), color, 2)
            # Добавить подпись с процентом уверенности
            label_text = class_names_ru.get(det['class'], det['class'])
            confidence_pct = det.get('confidence', 1.0) * 100
            label_with_conf = f"{label_text} {confidence_pct:.0f}%"
            labels_to_draw.append((label_with_conf, (x1, max(5, y1 - 18)), color))

    # --- Новое: обнаружить цели типа зданий на мозаике (прямоугольные, большие контуры)
    def detect_buildings(img, min_area=30):
        # Попытаться более надежный многошаговый подход для поиска больших искусственных прямоугольных форм
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Использовать простой порог для лучшего обнаружения
        _, th = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY_INV)

        # Морфологическое закрытие для объединения областей крыш и удаления маленьких дырок
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Обнаружение краев на закрытом изображении
        edges = cv2.Canny(closed, 30, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        buildings = []
        print(f"Найдено {len(contours)} контуров")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # Более снисходительное приближение
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            if rect_area <= 0:
                continue
            fill_ratio = area / float(rect_area)
            # Ослабленные критерии для зданий
            if fill_ratio > 0.05 and w > 5 and h > 5 and len(approx) >= 3:  # По крайней мере треугольник
                # Вычислить уверенность на основе fill_ratio, размера и прямоугольности
                # fill_ratio близко к 1.0 = более прямоугольное = выше уверенность
                # Большая площадь = более вероятно здание
                size_confidence = min(1.0, area / 1000.0)  # Нормализовать по размеру
                shape_confidence = fill_ratio  # 0.05-1.0 диапазон
                # Комбинированная уверенность (60% форма, 40% размер)
                confidence = (shape_confidence * 0.6 + size_confidence * 0.4)
                confidence = min(0.95, max(0.50, confidence))  # Ограничить 50-95%
                buildings.append((x, y, x + w, y + h, confidence))
        print(f"Обнаружено {len(buildings)} зданий")
        return buildings

    # Построить downsampled occupancy grid и использовать A* для маршрутизации.
    def find_path_astar(start_px, goal_px, obstacle_mask, scale=8):
        # Создать матрицу, где 0 = проходимо, 1 = заблокировано
        h, w = obstacle_mask.shape
        gh = max(1, h // scale)
        gw = max(1, w // scale)
        matrix = [[0 for _ in range(gw)] for _ in range(gh)]
        for gy in range(gh):
            for gx in range(gw):
                y0 = gy * scale
                x0 = gx * scale
                y1 = min(h, y0 + scale)
                x1 = min(w, x0 + scale)
                block = obstacle_mask[y0:y1, x0:x1]
                # Если присутствует хоть один пиксель препятствия, отметить заблокированным
                if np.any(block > 0):
                    matrix[gy][gx] = 1
        grid = Grid(matrix=matrix)
        start_node = grid.node(max(0, min(gw - 1, start_px[0] // scale)), max(0, min(gh - 1, start_px[1] // scale)))
        end_node = grid.node(max(0, min(gw - 1, goal_px[0] // scale)), max(0, min(gh - 1, goal_px[1] // scale)))
        finder = AStarFinder(diagonal_movement=True)
        path, runs = finder.find_path(start_node, end_node, grid)
        if not path:
            return None
        # Преобразовать путь обратно в координаты пикселей (центр ячейки)
        pixel_path = []
        for gx, gy in path:
            px = int(gx * scale + scale // 2)
            py = int(gy * scale + scale // 2)
            pixel_path.append((px, py))
        return pixel_path

    def smooth_path(path, window=3):
        """Простое сглаживание скользящим средним по координатам пути."""
        if not path or len(path) < 3:
            return path
        smoothed = []
        n = len(path)
        half = window // 2
        for i in range(n):
            sx = 0
            sy = 0
            cnt = 0
            for j in range(i - half, i + half + 1):
                if 0 <= j < n:
                    sx += path[j][0]
                    sy += path[j][1]
                    cnt += 1
            smoothed.append((int(sx / cnt), int(sy / cnt)))
        return smoothed

    # Вычислить пути и обнаружения зданий синхронно
    def worker_compute_paths(result_dict):
        try:
            print("Вычисление путей...")
            buildings = detect_buildings(frame)
            
            result_dict['buildings'] = buildings
            overlays = []
            labels = []
            worker_nav = nav_map.copy()
            print(f"Обнаружено {len(buildings)} зданий")
            
            # Пропустить построение путей если compute_paths=False
            if not compute_paths:
                for building in buildings:
                    bx1, by1, bx2, by2, confidence = building
                    cv2.rectangle(worker_nav, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    label_with_conf = f"Здание {confidence*100:.0f}%"
                    labels.append((label_with_conf, (bx1, max(5, by1 - 18)), (0, 255, 255)))
                result_dict['nav_overlay'] = worker_nav
                result_dict['labels'] = labels
                return
            
            for building in buildings:
                bx1, by1, bx2, by2, confidence = building
                center_x = (bx1 + bx2) // 2
                center_y = (by1 + by2) // 2
                astar_scale = 8  # Увеличенный масштаб для более быстрого вычисления
                path = find_path_astar((start_x, start_y), (center_x, center_y), obstacles, scale=astar_scale)
                if path:
                    path = smooth_path(path, window=3)
                    try:
                        pts = np.array(path, dtype=np.int32)
                        cv2.polylines(worker_nav, [pts], False, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                    except Exception:
                        for i in range(len(path) - 1):
                            cv2.line(worker_nav, path[i], path[i + 1], (0, 255, 0), 3)
                else:
                    if is_path_clear(start_x, start_y, center_x, center_y, obstacles):
                        cv2.line(worker_nav, (start_x, start_y), (center_x, center_y), (0, 255, 0), 2)
                    else:
                        mid_x = (start_x + center_x) // 2 + 50
                        mid_y = (start_y + center_y) // 2
                        if is_path_clear(start_x, start_y, mid_x, mid_y, obstacles) and is_path_clear(mid_x, mid_y, center_x, center_y, obstacles):
                            cv2.line(worker_nav, (start_x, start_y), (mid_x, mid_y), (0, 255, 0), 2)
                            cv2.line(worker_nav, (mid_x, mid_y), (center_x, center_y), (0, 255, 0), 2)
                        else:
                            draw_dotted_line(worker_nav, (start_x, start_y), (center_x, center_y), (0, 255, 0), 2)
                cv2.rectangle(worker_nav, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                label_with_conf = f"Здание {confidence*100:.0f}%"
                labels.append((label_with_conf, (bx1, max(5, by1 - 18)), (0, 255, 255)))
            result_dict['nav_overlay'] = worker_nav
            result_dict['labels'] = labels
        except Exception as e:
            print(f"Ошибка worker: {e}")
            result_dict['nav_overlay'] = nav_map.copy()
            result_dict['labels'] = []

    result = {}
    worker_compute_paths(result)

    # Если worker завершился, скопировать overlay и labels
    nav_map = result.get('nav_overlay', nav_map)
    for lab in result.get('labels', []):
        labels_to_draw.append(lab)

    # Финальный шаг: нарисовать все поставленные labels с PIL (дает правильное отображение Unicode)
    try:
        pil_final = Image.fromarray(cv2.cvtColor(nav_map, cv2.COLOR_BGR2RGB))
        draw_final = ImageDraw.Draw(pil_final)
        if 'font' in locals() and font is not None:
            use_font = font
        else:
            try:
                use_font = ImageFont.load_default()
            except Exception:
                use_font = None

        img_height, img_width = nav_map.shape[:2]
        for text, (tx, ty), color in labels_to_draw:
            # Ограничить координаты границами изображения
            tx = max(5, min(tx, img_width - 100))  # 100px запас справа для текста
            ty = max(20, min(ty, img_height - 5))  # 20px сверху, 5px снизу
            
            # нарисовать тонкий черный контур для читаемости
            if use_font is not None:
                outline_color = (0, 0, 0)
                # смещения для псевдо-контура
                for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw_final.text((tx + ox, ty + oy), text, font=use_font, fill=outline_color)
                draw_final.text((tx, ty), text, font=use_font, fill=tuple(color))
            else:
                # резерв: ничего не делать
                pass

        nav_map = cv2.cvtColor(np.array(pil_final), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Предупреждение: не удалось нарисовать labels с PIL: {e}")

    return nav_map


def is_path_clear(x1, y1, x2, y2, obstacles):
    """Проверить, избегает ли линия между двумя точками препятствий."""
    # Простая проверка: сэмплировать точки вдоль линии
    num_samples = 20
    for i in range(num_samples + 1):
        t = i / num_samples
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        if 0 <= px < obstacles.shape[1] and 0 <= py < obstacles.shape[0]:
            if obstacles[py, px] > 0:
                return False
    return True


def main(video_path=None, images_dir=None, update_callback=None, show_intermediate=True, output_dir=None):

    if images_dir:
        import glob
        image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))
        if not image_files:
            print("Нет изображений в указанной папке")
            return
        
        # Создать папку Detections
        detections_dir = os.path.join(output_dir, 'Detections') if output_dir else 'Detections'
        os.makedirs(detections_dir, exist_ok=True)
        
        # Загрузить модель YOLO
        try:
            model = YOLO('yolo11n.pt')
        except Exception as e:
            print(f"Ошибка загрузки модели YOLO: {e}")
            return
        
        def detect_objects_static(frame, model):
            original_shape = frame.shape[:2]
            resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            results = model.predict(resized, conf=0.4, iou=0.45, imgsz=640, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    scale_x = original_shape[1] / 640
                    scale_y = original_shape[0] / 640
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id] if hasattr(model, 'names') else str(class_id)
                    detections.append({
                        'class': class_name, 
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence
                    })
            
            # Добавить детекцию огня и дыма по цвету
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            b, g, r = cv2.split(frame)
            
            # Детекция огня - комбинация HSV и BGR анализа (очень строгие пороги)
            lower_fire1 = np.array([0, 130, 170])
            upper_fire1 = np.array([10, 255, 255])
            mask_fire1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
            lower_fire2 = np.array([170, 130, 170])
            upper_fire2 = np.array([180, 255, 255])
            mask_fire2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
            mask_fire = cv2.bitwise_or(mask_fire1, mask_fire2)
            
            # Дополнительная проверка: очень яркий красный, сильно доминирует над G и B
            fire_bgr_mask = ((r > g + 40) & (r > b + 50) & (r > 180)).astype(np.uint8) * 255
            mask_fire = cv2.bitwise_and(mask_fire, fire_bgr_mask)
            
            # Морфологическая обработка
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_CLOSE, kernel)
            mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel)

            # Детекция дыма - низкая насыщенность, высокая яркость
            lower_smoke = np.array([0, 0, 130])
            upper_smoke = np.array([180, 35, 230])
            mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)
            
            # Дополнительная проверка: очень близкие значения RGB (серый цвет) и высокая яркость
            smoke_bgr_mask = ((np.abs(r.astype(int) - g.astype(int)) < 15) & 
                             (np.abs(g.astype(int) - b.astype(int)) < 15) & 
                             (r > 130) & (g > 130) & (b > 130)).astype(np.uint8) * 255
            mask_smoke = cv2.bitwise_and(mask_smoke, smoke_bgr_mask)
            
            # Исключить тени по яркости
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright_mask = (gray > 130).astype(np.uint8) * 255
            mask_smoke = cv2.bitwise_and(mask_smoke, bright_mask)
            
            # Морфологическая обработка
            mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_CLOSE, kernel)
            mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_OPEN, kernel)
            
            # Найти контуры для огня
            fire_detections = []
            contours_fire, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_fire:
                area = cv2.contourArea(cnt)
                if area > 800:  # Увеличена минимальная площадь
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Проверка плотности заполнения контура
                    density = area / (w * h) if w * h > 0 else 0
                    if density > 0.35:  # Более строгая проверка
                        # Дополнительная проверка контраста в области
                        roi = frame[y:y+h, x:x+w]
                        if roi.size > 0:
                            roi_r = roi[:, :, 2]
                            max_r = np.max(roi_r)
                            mean_r = np.mean(roi_r)
                            # Огонь имеет высокий контраст в красном канале
                            if max_r > 200 and (max_r - mean_r) > 50:
                                fire_detections.append({
                                    'class': 'fire',
                                    'box': (x, y, x + w, y + h),
                                    'confidence': 0.9,
                                    'area': area
                                })
            
            # Найти контуры для дыма
            smoke_detections = []
            contours_smoke, _ = cv2.findContours(mask_smoke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_smoke:
                area = cv2.contourArea(cnt)
                if area < 2000:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                # Проверка плотности и формы
                density = area / (w * h) if w * h > 0 else 0
                if density < 0.35:
                    continue
                roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                if roi.size == 0:
                    continue
                std_dev = np.std(roi)
                mean_val = np.mean(roi)
                if not (mean_val > 135 and std_dev < 35):
                    continue
                smoke_detections.append({
                    'class': 'smoke',
                    'box': (x, y, x + w, y + h),
                    'confidence': 0.9,
                    'area': area
                })
            
            # Применить NMS для удаления перекрывающихся боксов
            def apply_nms(dets, iou_threshold=0.5):
                if not dets:
                    return []
                dets_sorted = sorted(dets, key=lambda x: x['area'], reverse=True)
                keep = []
                while dets_sorted:
                    current = dets_sorted.pop(0)
                    keep.append(current)
                    x1, y1, x2, y2 = current['box']
                    dets_sorted = [d for d in dets_sorted if 
                                  calculate_iou((x1, y1, x2, y2), d['box']) < iou_threshold]
                return keep
            
            def calculate_iou(box1, box2):
                x1, y1, x2, y2 = box1
                x3, y3, x4, y4 = box2
                xi1, yi1 = max(x1, x3), max(y1, y3)
                xi2, yi2 = min(x2, x4), min(y2, y4)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (x4 - x3) * (y4 - y3)
                union_area = box1_area + box2_area - inter_area
                return inter_area / union_area if union_area > 0 else 0
            
            # Применить NMS и добавить к результатам
            for det in apply_nms(fire_detections, 0.4):
                detections.append({'class': det['class'], 'box': det['box'], 'confidence': det['confidence']})
            for det in apply_nms(smoke_detections, 0.4):
                detections.append({'class': det['class'], 'box': det['box'], 'confidence': det['confidence']})
            
            return detections
        
        print(f"Найдено {len(image_files)} изображений для обработки")
        for image_path in image_files:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Не удалось загрузить {image_path}")
                continue
            
            detections = detect_objects_static(frame, model)
            
            # Создать навигационную карту с детекциями (без построения путей для скорости)
            nav_map = analyze_for_navigation(frame, detections, compute_paths=False)
            
            # Нарисовать bounding boxes на оригинальном изображении
            detected_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(detected_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Сохранить изображение с detections
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            out_path = os.path.join(detections_dir, f"{name}_detected{ext}")
            cv2.imwrite(out_path, detected_frame)
            
            # Сохранить навигационную карту
            nav_path = os.path.join(detections_dir, f"{name}_navigation{ext}")
            cv2.imwrite(nav_path, nav_map)
            
            print(f"Обработано {base_name}: {len(detections)} объектов")
        
        print("Обработка изображений завершена.")
        return
        
    else:
        if video_path is None:
            video_path = 'Data/поиски квадрокоптера 2 (360p) 03.mp4'
        print(f"Открытие видеофайла: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть видеофайл")
            return

        # Создать папку Detections
        detections_dir = os.path.join(output_dir, 'Detections') if output_dir else 'Detections'
        os.makedirs(detections_dir, exist_ok=True)

        # Получить общее количество кадров для расчета прогресса
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Всего кадров в видео: {total_frames}")
        frame_count = 0
        is_first_frame = True
        first_frame_shape = None
        print("Запуск обработки видео и формирования мозаики...")
        
        while cap.isOpened():
            ret, frame_cur = cap.read()
            if not ret:
                break

            if is_first_frame:
                first_frame_shape = frame_cur.shape[:2]
                video_mosaic = VideMosaic(frame_cur, detector_type="sift", show_intermediate=show_intermediate, output_dir=output_dir, visualize=show_intermediate)
                is_first_frame = False
                # Рассчитать начальную точку как нижний-центр первого кадра в координатах мозаики
                start_x = video_mosaic.h_offset + first_frame_shape[1] // 2
                start_y = video_mosaic.w_offset + first_frame_shape[0]
                start_point = (start_x, start_y)
                continue

            frame_count += 1
            # обработать каждый кадр
            video_mosaic.process_frame(frame_cur, frame_count)
            
            # Обновить окна OpenCV
            cv2.waitKey(1)
            
            # Печатать прогресс каждые 50 кадров
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Обработан кадр {frame_count}/{total_frames} ({progress:.1f}%)")
                sys.stdout.flush()
            
            # Проверить, запросил ли пользователь прервать во время обработки
            if hasattr(video_mosaic, 'quit_requested') and video_mosaic.quit_requested:
                print("Обработка прервана пользователем.")
                break
            
            # Обновить прогресс, если предоставлен callback
            if update_callback:
                progress = (frame_count / total_frames) * 100
                update_callback(frame_count, video_mosaic.output_img.copy(), progress)
                
        cap.release()
        print("Обработка видео завершена. Мозаика сформирована.")
        sys.stdout.flush()
    
    # Держать финальную мозаику отображаемой до закрытия пользователем
    if show_intermediate:
        print("Финальная мозаика завершена. Нажмите любую клавишу для закрытия окна.")
        cv2.imshow('Mosaic Progress', video_mosaic.output_img)
        # cv2.waitKey(0)  # Ждать любого нажатия клавиши
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        
    print("Обрезка черных областей из мозаики...")
    try:
        cropped = crop_black_areas(video_mosaic.output_img, threshold=80, margin=30)
        print(f"Размер обрезанной мозаики: {cropped.shape[1]}x{cropped.shape[0]}")
    except Exception as e:
        print(f"Предупреждение: обрезка не удалась, используется полная мозаика: {e}")
        cropped = video_mosaic.output_img

    print("Масштабирование мозаики к экрану (с сохранением соотношения)...")
    try:
        scaled_mosaic = scale_to_screen(cropped)
        print(f"Размер масштабированной мозаики: {scaled_mosaic.shape[1]}x{scaled_mosaic.shape[0]}")
    except Exception as e:
        print(f"Предупреждение: масштабирование не удалось, сохраняется оригинал: {e}")
        scaled_mosaic = cropped

    print("Сохранение изображения мозаики...")
    mosaic_path = os.path.join(output_dir, 'mosaic.jpg') if output_dir else 'mosaic.jpg'
    cv2.imwrite(mosaic_path, scaled_mosaic)
    print(f"Мозаика сохранена как '{mosaic_path}'")

    # Обнаружить объекты на мозаике
    print("Обнаружение объектов на мозаике...")
    detections = video_mosaic.detect_objects(video_mosaic.output_img.astype(np.uint8))
    # fire_smoke_detections = video_mosaic.detect_fire_smoke(video_mosaic.output_img.astype(np.uint8))
    # all_detections = detections + fire_smoke_detections
    all_detections = detections
    print(f"Обнаружено {len(all_detections)} объектов на мозаике.")

    # Опционально проанализировать мозаику для навигации: отметить препятствия и нарисовать пути к объектам
    # Закомментировано создание карты навигации по запросу пользователя
    print("Анализ мозаики для навигации...")
    navigation_map = analyze_for_navigation(scaled_mosaic.astype(np.uint8), all_detections, start_point=start_point)
    print("Масштабирование карты навигации к экрану...")
    try:
        scaled_nav = scale_to_screen(navigation_map)
    except Exception:
        scaled_nav = navigation_map
    print("Сохранение карты навигации...")
    nav_path = os.path.join(output_dir, 'navigation_map.jpg') if output_dir else 'navigation_map.jpg'
    cv2.imwrite(nav_path, scaled_nav)
    print(f"Карта навигации сохранена как '{nav_path}'")
    sys.stdout.flush()
    if show_intermediate:
        cv2.imshow('Navigation Map', scaled_nav)
        cv2.waitKey(1)
    if show_intermediate:
        cv2.imshow('Navigation Map', navigation_map)
        print("Карта навигации отображена. Нажмите любую клавишу в окне для продолжения.")
    
    # Закрыть все окна OpenCV в конце программы
    cv2.destroyAllWindows()
    
    # Проверить, была ли обработка прервана пользователем
    if hasattr(video_mosaic, 'quit_requested') and video_mosaic.quit_requested:
        print("Генерация мозаики видео была прервана пользователем.")
        return
    
    # Финальное обновление с завершением
    if update_callback:
        update_callback(frame_count, video_mosaic.output_img.copy(), 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Процессор Мозаики Видео')
    parser.add_argument('video_path', nargs='?', default=None, help='Путь к видеофайлу')
    parser.add_argument('--images-dir', help='Папка с изображениями для обработки вместо видео')
    parser.add_argument('--output-dir', default=None, help='Каталог вывода для результатов')
    parser.add_argument('--hide', action='store_true', help='Отключить окна GUI')
    
    args = parser.parse_args()
    
    show_intermediate = not args.hide
    if args.images_dir:
        main(images_dir=args.images_dir, show_intermediate=show_intermediate, output_dir=args.output_dir)
    else:
        main(video_path=args.video_path, show_intermediate=show_intermediate, output_dir=args.output_dir)