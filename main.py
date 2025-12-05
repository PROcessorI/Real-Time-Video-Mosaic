import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder
from PIL import Image, ImageDraw, ImageFont
import argparse
import sys


import gc  # Для сборки мусора

class VideMosaic:
    # def __init__(self, first_image, output_height_times=2, output_width_times=4, detector_type="sift"):
    def __init__(self, first_image, output_height_times=2, output_width_times=1.2, detector_type="sift", show_intermediate=True, output_dir=None, visualize=True):
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
            # Использовать nano модель для быстрой работы, кастомная детекция дополняет для аэроснимков
            self.model = YOLO('yolo11n.pt')  # YOLOv11 nano модель
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить модель YOLO: {e}")
            self.model = None
        
        # Инициализировать YOLO-World для детекции машин с вида сверху (аэроснимки)
        try:
            self.model_world = YOLO('yolov8x-worldv2.pt')
            # Оптимизированные классы для аэроснимков (меньше классов = точнее детекция)
            self.detection_classes = [
                # Транспорт (основные)
                'car', 'truck', 'bus', 'van',
                # Люди
                'person',
                # Животные (основные)
                'dog', 'cat',
                # Строения - фокус на крышах и зданиях с вида сверху
                'building', 'house', 'roof', 'shed', 'barn', 'garage', 
                'greenhouse', 'warehouse',
                # Прочее важное
                'pool', 'boat'
            ]
            self.model_world.set_classes(self.detection_classes)
            print("YOLO-World модель загружена для универсальной детекции объектов")
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить YOLO-World: {e}")
            self.model_world = None

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
        
        # Параметры стабилизации для защиты от тряски
        self.stabilization_enabled = True
        self.homography_history = []  # История гомографий для сглаживания
        self.history_size = 5  # Количество кадров для усреднения
        self.translation_threshold = 50  # Максимальное смещение (пиксели)
        self.scale_threshold = 0.3  # Максимальное изменение масштаба
        self.last_valid_H = np.eye(3)

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

    def _enhance_for_detection(self, frame):
        """Улучшает контраст изображения для лучшей детекции белых объектов на светлом фоне."""
        # Конвертируем в LAB для работы с яркостью
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Собираем обратно
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def detect_objects(self, frame):
        """Универсальная детекция объектов с помощью YOLO-World.
        
        YOLO-World может детектировать любые объекты по текстовому описанию,
        что идеально подходит для аэроснимков и нестандартных ракурсов.
        
        Args:
            frame: входное изображение (BGR)
            
        Returns:
            list: список детекций в формате {'class': str, 'box': tuple, 'confidence': float}
        """
        detections = []
        
        # Используем YOLO-World как основной детектор
        if hasattr(self, 'model_world') and self.model_world is not None:
            try:
                # === Мультимасштабная детекция для лучшего обнаружения зданий ===
                
                # 1. Детекция на полном изображении (высокое разрешение)
                # Используем более высокий порог для уменьшения ложных срабатываний
                results = self.model_world.predict(
                    frame,
                    conf=0.02,      # Повышенный порог для уменьшения FP
                    imgsz=1280,     # Высокое разрешение
                    verbose=False,
                    augment=True,   # Аугментация для лучшей детекции
                    iou=0.5         # Повышенный IoU для лучшего NMS
                )
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = self.model_world.names[cls_id]
                        normalized_class = self._normalize_class_name(class_name)
                        
                        detections.append({
                            'class': normalized_class,
                            'box': (x1, y1, x2, y2),
                            'confidence': conf
                        })
                
                # 2. Детекция на улучшенном изображении
                enhanced_frame = self._enhance_for_detection(frame)
                results2 = self.model_world.predict(
                    enhanced_frame,
                    conf=0.02,      # Повышенный порог
                    imgsz=1280,
                    verbose=False,
                    iou=0.5
                )
                
                for r in results2:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = self.model_world.names[cls_id]
                        normalized_class = self._normalize_class_name(class_name)
                        
                        if not self._is_duplicate(detections, x1, y1, x2, y2):
                            detections.append({
                                'class': normalized_class,
                                'box': (x1, y1, x2, y2),
                                'confidence': conf
                            })
                
                # 3. Sliding window детекция для поиска зданий на больших изображениях
                h, w = frame.shape[:2]
                if w > 800 or h > 800:
                    # Разбиваем на перекрывающиеся окна
                    window_size = 640
                    stride = 400
                    
                    for y_start in range(0, h - window_size // 2, stride):
                        for x_start in range(0, w - window_size // 2, stride):
                            x_end = min(x_start + window_size, w)
                            y_end = min(y_start + window_size, h)
                            
                            if x_end - x_start < 200 or y_end - y_start < 200:
                                continue
                            
                            window = frame[y_start:y_end, x_start:x_end]
                            
                            results_window = self.model_world.predict(
                                window,
                                conf=0.03,      # Повышенный порог для окон
                                imgsz=640,
                                verbose=False,
                                iou=0.5
                            )
                            
                            for r in results_window:
                                for box in r.boxes:
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])
                                    
                                    # Преобразуем координаты обратно в глобальные
                                    gx1 = x_start + wx1
                                    gy1 = y_start + wy1
                                    gx2 = x_start + wx2
                                    gy2 = y_start + wy2
                                    
                                    class_name = self.model_world.names[cls_id]
                                    normalized_class = self._normalize_class_name(class_name)
                                    
                                    if not self._is_duplicate(detections, gx1, gy1, gx2, gy2):
                                        detections.append({
                                            'class': normalized_class,
                                            'box': (gx1, gy1, gx2, gy2),
                                            'confidence': conf * 0.9  # Немного понижаем уверенность для sliding window
                                        })
                        
            except Exception as e:
                print(f"YOLO-World ошибка: {e}")
                import traceback
                traceback.print_exc()
                # Fallback на стандартную YOLO если YOLO-World не работает
                detections = self._detect_with_standard_yolo(frame)
        else:
            # Fallback на стандартную YOLO
            detections = self._detect_with_standard_yolo(frame)
        
        # Фильтруем слишком большие и слишком маленькие детекции
        frame_area = frame.shape[0] * frame.shape[1]
        max_det_area = frame_area * 0.15  # Макс 15% кадра
        filtered_detections = []
        for d in detections:
            w = d['box'][2] - d['box'][0]
            h = d['box'][3] - d['box'][1]
            area = w * h
            
            # Минимальная площадь зависит от класса
            min_area = 200 if d['class'] == 'building' else 80
            
            # Убираем слишком большие и слишком маленькие
            if min_area < area < max_det_area:
                # Дополнительно фильтруем здания по размерам
                if d['class'] == 'building':
                    # Здания должны быть достаточно большими
                    if min(w, h) >= 25 and max(w, h) >= 40:
                        filtered_detections.append(d)
                else:
                    filtered_detections.append(d)
        detections = filtered_detections
        
        # CV2 детекция зданий - для аэроснимков
        cv2_buildings = self._detect_buildings_cv2(frame)
        print(f"CV2 детектировал {len(cv2_buildings)} потенциальных зданий")
        
        # Добавляем только те, которые не перекрываются с существующими детекциями
        for cv2_det in cv2_buildings:
            is_duplicate = False
            for existing in detections:
                # Проверяем перекрытие со всеми детекциями (не только зданиями)
                x1, y1, x2, y2 = cv2_det['box']
                ex1, ey1, ex2, ey2 = existing['box']
                
                # Вычисляем пересечение
                ix1 = max(x1, ex1)
                iy1 = max(y1, ey1)
                ix2 = min(x2, ex2)
                iy2 = min(y2, ey2)
                
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (ex2 - ex1) * (ey2 - ey1)
                    iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0
                    if iou > 0.3:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                detections.append(cv2_det)
        
        # CV2 детекция машин (светлые прямоугольные объекты)
        cv2_vehicles = self._detect_vehicles_cv2(frame)
        print(f"CV2 детектировал {len(cv2_vehicles)} потенциальных машин")
        
        for cv2_det in cv2_vehicles:
            is_duplicate = False
            for existing in detections:
                x1, y1, x2, y2 = cv2_det['box']
                ex1, ey1, ex2, ey2 = existing['box']
                
                # Проверяем перекрытие центров
                c1 = ((x1+x2)/2, (y1+y2)/2)
                c2 = ((ex1+ex2)/2, (ey1+ey2)/2)
                dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
                if dist < 25:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                detections.append(cv2_det)
        
        return detections
    
    def _normalize_class_name(self, class_name):
        """Нормализует название класса к стандартному виду."""
        # Транспорт -> car
        if class_name in ['car', 'vehicle', 'automobile', 'van']:
            return 'car'
        elif class_name in ['truck']:
            return 'truck'
        elif class_name in ['bus']:
            return 'bus'
        elif class_name in ['motorcycle']:
            return 'motorcycle'
        elif class_name in ['bicycle']:
            return 'bicycle'
        # Люди -> person
        elif class_name in ['person', 'people', 'human', 'pedestrian']:
            return 'person'
        # Опасности
        elif class_name in ['fire', 'flame']:
            return 'fire'
        elif class_name in ['smoke']:
            return 'smoke'
        elif class_name in ['explosion']:
            return 'explosion'
        # Животные
        elif class_name in ['dog']:
            return 'dog'
        elif class_name in ['cat']:
            return 'cat'
        elif class_name in ['bird']:
            return 'bird'
        elif class_name in ['animal']:
            return 'animal'
        # Строения - расширенный список
        elif class_name in ['building', 'house', 'roof', 'structure', 'shed', 'barn', 
                           'garage', 'greenhouse', 'warehouse', 'cottage', 'cabin', 
                           'hut', 'shelter', 'rooftop', 'construction', 'facility',
                           'residential building', 'metal roof', 'wooden building',
                           'container', 'storage', 'outbuilding', 'farmhouse',
                           'pavilion', 'canopy', 'carport', 'shack']:
            return 'building'
        # Воздушный/водный транспорт
        elif class_name in ['boat', 'ship']:
            return 'boat'
        elif class_name in ['airplane']:
            return 'airplane'
        elif class_name in ['helicopter']:
            return 'helicopter'
        elif class_name in ['drone']:
            return 'drone'
        elif class_name in ['pool']:
            return 'pool'
        elif class_name in ['tent']:
            return 'tent'
        elif class_name in ['solar panel']:
            return 'solar_panel'
        elif class_name in ['fence']:
            return 'fence'
        elif class_name in ['garden bed']:
            return 'garden_bed'
        else:
            return class_name
    
    def _is_duplicate(self, detections, x1, y1, x2, y2, threshold=40):
        """Проверяет, является ли детекция дубликатом существующей."""
        for d in detections:
            dx1, dy1, dx2, dy2 = d['box']
            # Если центры близко - дубликат
            c1 = ((x1+x2)/2, (y1+y2)/2)
            c2 = ((dx1+dx2)/2, (dy1+dy2)/2)
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            if dist < threshold:
                return True
            
            # Также проверяем IoU
            ix1 = max(x1, dx1)
            iy1 = max(y1, dy1)
            ix2 = min(x2, dx2)
            iy2 = min(y2, dy2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (dx2 - dx1) * (dy2 - dy1)
                iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0
                if iou > 0.5:
                    return True
        return False
    
    def _detect_with_standard_yolo(self, frame):
        """Fallback детекция со стандартной YOLO моделью."""
        if self.model is None:
            return []
            
        detections = []
        results = self.model.predict(
            frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id] if hasattr(self.model, 'names') else str(class_id)
                detections.append({
                    'class': class_name,
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence
                })
        
        return detections

    def _detect_buildings_cv2(self, frame):
        """Детекция зданий с watershed для разделения слипшихся областей."""
        detections = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Убираем чёрные границы
        _, valid_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        frame_h, frame_w = gray.shape
        frame_area = frame_h * frame_w
        min_area = 400
        max_area = frame_area * 0.08  # 8% после разделения
        
        # === Маска серых областей (здания) ===
        gray_mask = cv2.inRange(s, 0, 50)
        gray_mask = cv2.bitwise_and(gray_mask, cv2.inRange(v, 60, 220))
        gray_mask = cv2.bitwise_and(gray_mask, valid_mask)
        
        # === Границы для разделения ===
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120)
        
        # Толстые границы
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_thick = cv2.dilate(edges, kernel_edge, iterations=3)
        
        # === WATERSHED для разделения ===
        # Уверенный фон (то что точно НЕ здание)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        sure_bg = cv2.dilate(gray_mask, kernel_bg, iterations=3)
        
        # Уверенный передний план (центры зданий)
        dist_transform = cv2.distanceTransform(gray_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Неизвестная область
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Маркеры для watershed
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed
        frame_bgr = frame.copy()
        markers = cv2.watershed(frame_bgr, markers)
        
        # Создаём маску разделённых зданий
        buildings_separated = np.zeros_like(gray)
        buildings_separated[markers > 1] = 255
        
        # Вычитаем границы watershed
        buildings_separated[markers == -1] = 0
        
        # Дополнительно вычитаем Canny edges
        buildings_separated = cv2.subtract(buildings_separated, edges_thick)
        
        # Морфология
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        buildings_separated = cv2.morphologyEx(buildings_separated, cv2.MORPH_OPEN, kernel_clean, iterations=2)
        
        cv2.imwrite('debug_watershed.jpg', buildings_separated)
        
        # Находим контуры
        contours, _ = cv2.findContours(buildings_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            if min(w, h) < 15:
                continue
            
            rect = cv2.minAreaRect(cnt)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area == 0:
                continue
            
            rectangularity = area / rect_area
            if rectangularity < 0.35:
                continue
            
            aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if aspect > 5:
                continue
            
            eps = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) < 4:
                continue
            
            if self._is_duplicate(detections, x, y, x+w, y+h, threshold=20):
                continue
            
            confidence = min(0.75, rectangularity * 0.5 + 0.20)
            
            detections.append({
                'class': 'building',
                'box': (x, y, x + w, y + h),
                'confidence': confidence
            })
        
        print(f"CV2 (watershed) детектировал {len(detections)} зданий")
        return detections
        
        # Обрабатываем каждую маску
        for mask_idx, mask in enumerate(all_building_masks):
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                if area < min_building_area or area > max_building_area:
                    continue
                
                # Получаем bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Минимальный размер (снижен для небольших построек)
                if min(w, h) < 25:
                    continue
                
                # Aspect ratio - здания обычно не сильно вытянутые
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect > 5.0:  # Увеличено для длинных зданий
                    continue
                
                # Прямоугольность
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                
                # Здания заполняют bbox хотя бы на 30% (снижено)
                if extent < 0.30:
                    continue
                
                # Аппроксимируем контур
                epsilon = 0.04 * cv2.arcLength(cnt, True)  # Увеличено для упрощения контуров
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                # Здания имеют 4-16 углов (более гибко для сложных форм)
                if len(approx) < 4 or len(approx) > 20:
                    continue
                
                # Проверяем что это не дубликат
                if self._is_duplicate(detections, x, y, x+w, y+h, threshold=40):
                    continue
                
                # Вычисляем уверенность на основе размера и прямоугольности
                size_factor = min(1.0, area / 5000)  # Больше здание - выше уверенность
                confidence = min(0.75, extent * 0.4 + 0.25 + size_factor * 0.1)
                
                detections.append({
                    'class': 'building',
                    'box': (x, y, x + w, y + h),
                    'confidence': confidence
                })
        
        print(f"CV2 детектировал {len(detections)} зданий (серые/светлые/тёмные крыши)")
        return detections

    def _detect_vehicles_cv2(self, frame):
        """Детекция машин через CV2 - ищем прямоугольные объекты характерного размера."""
        detections = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Убираем чёрные границы мозаики
        _, valid_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Ищем светлые объекты (белые/серые машины)
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        bright = cv2.bitwise_and(bright, valid_mask)
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Машины с вида сверху имеют площадь примерно 200-5000 пикселей
            if 150 < area < 8000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / h if h > 0 else 0
                
                # Машины обычно прямоугольные 1.5-3.5 : 1
                if 0.3 < aspect < 4.0 and min(w, h) > 8 and max(w, h) < 150:
                    rect_area = w * h
                    extent = area / rect_area if rect_area > 0 else 0
                    
                    # Машины заполняют bbox на 50%+
                    if extent > 0.5:
                        detections.append({
                            'class': 'car',
                            'box': (x, y, x+w, y+h),
                            'confidence': 0.3 + extent * 0.3
                        })
        
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
            print(f"Предупреждение: Недостаточно совпадений ({len(self.matches)}), пропуск кадра")
            return

        # Вычислить гомографию
        H_relative = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
        
        if H_relative is None:
            print("Предупреждение: Не удалось вычислить гомографию, пропуск кадра")
            return
        
        # Проверка на валидность гомографии (защита от тряски)
        if not self.validate_homography(H_relative):
            print("Предупреждение: Невалидная гомография (тряска/размытие), использую последнюю валидную")
            # Использовать последнюю валидную относительную гомографию (минимальное движение)
            H_relative = np.eye(3)
        else:
            # Сохранить как последнюю валидную
            self.last_valid_H = H_relative.copy()
        
        # Применить сглаживание для уменьшения дрожания
        H_relative_smoothed = self.smooth_homography(H_relative)
        
        # Вычислить абсолютную гомографию
        self.H = np.matmul(self.H_old, H_relative_smoothed)

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

    def validate_homography(self, H):
        """Проверяет гомографию на адекватность и фильтрует резкие движения. Удаляет эффект тряски камеры.
        
        Args:
            H: матрица гомографии для проверки
            
        Returns:
            bool: True если гомография валидна, False если подозрительна
        """
        if H is None:
            return False
        
        # Проверка на NaN и Inf
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            return False
        
        # Извлечь параметры трансформации
        # Смещение по X и Y
        translation_x = H[0, 2]
        translation_y = H[1, 2]
        translation = np.sqrt(translation_x**2 + translation_y**2)
        
        # Масштаб (определитель верхней левой 2x2 матрицы)
        scale = np.sqrt(np.linalg.det(H[:2, :2]))
        
        # Проверка на слишком большое смещение (тряска)
        if translation > self.translation_threshold:
            print(f"Предупреждение: Обнаружено большое смещение ({translation:.1f}px), возможна тряска")
            return False
        
        # Проверка на неадекватное изменение масштаба
        if abs(scale - 1.0) > self.scale_threshold:
            print(f"Предупреждение: Обнаружено большое изменение масштаба ({scale:.2f}), возможна тряска")
            return False
        
        # Проверка на перспективные искажения (элементы H[2,0] и H[2,1] должны быть малы)
        if abs(H[2, 0]) > 0.001 or abs(H[2, 1]) > 0.001:
            print(f"Предупреждение: Обнаружены сильные перспективные искажения")
            return False
        
        return True
    
    def smooth_homography(self, H):
        """Сглаживает гомографию используя историю предыдущих кадров.
        
        Args:
            H: текущая матрица гомографии
            
        Returns:
            np.array: сглаженная гомография
        """
        if not self.stabilization_enabled:
            return H
        
        # Добавить текущую гомографию в историю
        self.homography_history.append(H.copy())
        
        # Ограничить размер истории
        if len(self.homography_history) > self.history_size:
            self.homography_history.pop(0)
        
        # Если история недостаточна, вернуть текущую гомографию
        if len(self.homography_history) < 2:
            return H
        
        # Усреднить гомографии с весами (более свежие кадры имеют больший вес)
        weights = np.linspace(0.5, 1.0, len(self.homography_history))
        weights = weights / np.sum(weights)
        
        smoothed_H = np.zeros_like(H)
        for i, h in enumerate(self.homography_history):
            smoothed_H += weights[i] * h
        
        return smoothed_H
    
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
            try:
                # Вычислить расстояние от краёв для обеих масок
                dist_new = cv2.distanceTransform(mask_new, cv2.DIST_L2, 3)  # Уменьшен размер ядра
                dist_old = cv2.distanceTransform(mask_old, cv2.DIST_L2, 3)
                
                # Нормализовать расстояния для весов
                dist_sum = dist_new + dist_old + 1e-6
                weight_new = dist_new / dist_sum
                weight_old = dist_old / dist_sum
                
                # Применить размытие для плавных переходов (уменьшен размер ядра)
                weight_new = cv2.GaussianBlur(weight_new.astype(np.float32), (31, 31), 0)
                weight_old = cv2.GaussianBlur(weight_old.astype(np.float32), (31, 31), 0)
                
                # Расширить до 3 каналов
                weight_new_3ch = np.stack([weight_new, weight_new, weight_new], axis=2)
                weight_old_3ch = np.stack([weight_old, weight_old, weight_old], axis=2)
                
                # Плавное смешивание с учётом расстояния от краёв
                blended = (self.output_img.astype(np.float32) * weight_old_3ch + 
                          warped_img.astype(np.float32) * weight_new_3ch)
                
                # Применить блендинг только в области перекрытия
                overlap_3ch = np.stack([overlap, overlap, overlap], axis=2) > 0
                self.output_img = np.where(overlap_3ch, blended.astype(np.uint8), self.output_img)
                
                # Освобождаем память
                del dist_new, dist_old, weight_new, weight_old, weight_new_3ch, weight_old_3ch, blended
                gc.collect()
                
            except cv2.error as e:
                # При нехватке памяти используем простое наложение
                print(f"Предупреждение: упрощённое смешивание из-за нехватки памяти")
                self.output_img[warped_img > 0] = warped_img[warped_img > 0]
            
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
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
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
    # Создать взвешенную маску для приоритезации опасных препятствий
    obstacles_weighted = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    # Добавить буфер вокруг обнаруженных объектов с адаптивными параметрами
    # Расширенный список классов с приоритезацией
    danger_classes = ['fire', 'smoke']
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    living_classes = ['person', 'dog', 'horse', 'cat', 'bird', 'cow', 'sheep']
    static_classes = ['bicycle', 'building']
    
    for det in detections:
        det_class = det['class']
        if det_class in danger_classes + vehicle_classes + living_classes + static_classes:
            x1, y1, x2, y2 = det['box']
            
            # Адаптивный буфер в зависимости от типа и размера объекта
            obj_area = (x2 - x1) * (y2 - y1)
            size_factor = min(1.5, max(1.0, obj_area / 10000.0))  # Масштабирование по размеру
            
            if det_class in danger_classes:
                buffer = int(40 * size_factor)  # Максимальный буфер для огня/дыма
                weight = 1.0
            elif det_class in vehicle_classes:
                buffer = int(25 * size_factor)  # Средний буфер для транспорта
                weight = 0.9
            elif det_class in living_classes:
                buffer = int(20 * size_factor)  # Буфер для живых существ
                weight = 0.85
            else:
                buffer = int(15 * size_factor)  # Минимальный буфер для статичных объектов
                weight = 0.7
            
            x1_buf = max(0, x1 - buffer)
            y1_buf = max(0, y1 - buffer)
            x2_buf = min(obstacles.shape[1], x2 + buffer)
            y2_buf = min(obstacles.shape[0], y2 + buffer)
            
            cv2.rectangle(obstacles, (x1_buf, y1_buf), (x2_buf, y2_buf), 255, -1)
            cv2.rectangle(obstacles_weighted, (x1_buf, y1_buf), (x2_buf, y2_buf), weight, -1)

    # Добавить цветовую детекцию огня и дыма для препятствий
    # Использовать улучшенный подход с edge detection и текстурным анализом
    b_nav, g_nav, r_nav = cv2.split(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Огонь - многомасштабный подход для лучшего обнаружения
    # Основные маски - расширенный диапазон для навигационной карты
    lower_fire1 = np.array([0, 80, 120])
    upper_fire1 = np.array([15, 255, 255])
    mask_fire1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    
    lower_fire2 = np.array([160, 80, 120])
    upper_fire2 = np.array([180, 255, 255])
    mask_fire2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    
    # Желто-оранжевый огонь
    lower_fire3 = np.array([15, 70, 130])
    upper_fire3 = np.array([35, 255, 255])
    mask_fire3 = cv2.inRange(hsv, lower_fire3, upper_fire3)
    
    mask_fire = cv2.bitwise_or(cv2.bitwise_or(mask_fire1, mask_fire2), mask_fire3)
    
    # BGR проверки - несколько вариантов
    fire_bgr_mask1 = ((r_nav > g_nav + 25) & (r_nav > b_nav + 35) & (r_nav > 130)).astype(np.uint8) * 255
    # Желтый огонь
    fire_bgr_mask2 = ((r_nav > 130) & (g_nav > 110) & (r_nav > b_nav + 25) & 
                     (g_nav > b_nav + 15) & (np.abs(r_nav.astype(int) - g_nav.astype(int)) < 60)).astype(np.uint8) * 255
    
    fire_bgr_combined = cv2.bitwise_or(fire_bgr_mask1, fire_bgr_mask2)
    mask_fire = cv2.bitwise_and(mask_fire, fire_bgr_combined)
    
    # Добавить edge detection для контуров огня
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    # Использовать края как дополнительную подсказку
    mask_fire = cv2.bitwise_or(mask_fire, cv2.bitwise_and(edges_dilated, mask_fire))
    
    # Адаптивная морфологическая обработка
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel_small)
    obstacles = cv2.bitwise_or(obstacles, mask_fire)
    
    # Дым - улучшенная детекция с текстурным анализом
    # Несколько слоев для разных типов дыма
    lower_smoke1 = np.array([0, 0, 90])
    upper_smoke1 = np.array([180, 45, 245])
    mask_smoke1 = cv2.inRange(hsv, lower_smoke1, upper_smoke1)
    
    # Темный дым
    lower_smoke2 = np.array([0, 0, 60])
    upper_smoke2 = np.array([180, 55, 140])
    mask_smoke2 = cv2.inRange(hsv, lower_smoke2, upper_smoke2)
    
    mask_smoke = cv2.bitwise_or(mask_smoke1, mask_smoke2)
    
    # BGR проверка - близкие значения RGB (серый цвет)
    smoke_bgr_mask = ((np.abs(r_nav.astype(int) - g_nav.astype(int)) < 25) &
                     (np.abs(g_nav.astype(int) - b_nav.astype(int)) < 25) &
                     (r_nav > 60) & (g_nav > 60) & (b_nav > 60)).astype(np.uint8) * 255
    mask_smoke = cv2.bitwise_and(mask_smoke, smoke_bgr_mask)
    
    # Исключить очень темные области (тени)
    bright_enough = (gray > 70).astype(np.uint8) * 255
    mask_smoke = cv2.bitwise_and(mask_smoke, bright_enough)
    
    # Текстурный анализ - дым имеет низкую вариативность
    # Использовать локальное стандартное отклонение
    gray_float = gray.astype(np.float32)
    kernel_texture = np.ones((11, 11), np.float32) / 121
    local_mean = cv2.filter2D(gray_float, -1, kernel_texture)
    local_sq_mean = cv2.filter2D(gray_float**2, -1, kernel_texture)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    # Дым имеет низкую текстуру
    low_texture_mask = (local_std < 40).astype(np.uint8) * 255
    mask_smoke = cv2.bitwise_and(mask_smoke, low_texture_mask)
    
    # Агрессивная морфологическая обработка для объединения облаков дыма
    kernel_smoke_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_CLOSE, kernel_smoke_large, iterations=3)
    mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_OPEN, kernel_small, iterations=1)
    obstacles = cv2.bitwise_or(obstacles, mask_smoke)

    # ===== ДЕТЕКЦИЯ ПРЕПЯТСТВИЙ ПО ТЕКСТУРАМ =====
    
    # 1. Создать рабочую область (убрать черную рамку)
    _, valid_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    valid_mask = cv2.erode(valid_mask, kernel_erode, iterations=1)
    
    # 2. Текстурный анализ - находит деревья, кусты, здания
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    texture_diff = cv2.absdiff(gray, blurred)
    _, texture_mask = cv2.threshold(texture_diff, 6, 255, cv2.THRESH_BINARY)
    texture_mask = cv2.bitwise_and(texture_mask, valid_mask)
    
    # 3. Морфология - меньше обработки чтобы сохранить больше контуров
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # DEBUG: Сохранить
    cv2.imwrite('debug_texture_mask.jpg', texture_mask)
    
    # 4. Найти контуры текстур
    all_contours, hierarchy = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтруем контуры по площади
    obstacle_contours_to_draw = []
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 500000:  # Минимум 20 пикселей для большего количества контуров
            obstacle_contours_to_draw.append(cnt)
    
    print(f"DEBUG: Найдено {len(obstacle_contours_to_draw)} контуров препятствий")
    
    # Для навигации добавляем текстурную маску к существующим препятствиям
    obstacles = cv2.bitwise_or(obstacles, texture_mask)
    
    # Расширить препятствия для навигации
    kernel_nav_buffer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    obstacles_for_navigation = cv2.dilate(obstacles, kernel_nav_buffer, iterations=1)

    # Отметить препятствия как контуры на копии кадра
    nav_map = frame.copy()
    
    # Рисуем только сохраненные контуры препятствий красным цветом (без заливки и без краевых)
    if obstacle_contours_to_draw:
        cv2.drawContours(nav_map, obstacle_contours_to_draw, -1, (0, 0, 255), 2)  # Красные контуры

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
        "times.ttf"
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
            # Цвет в зависимости от типа объекта
            if det['class'] in ['fire', 'smoke']:
                color = (0, 0, 255)  # Красный для опасных объектов
            elif det['class'] in ['car', 'truck', 'bus', 'motorcycle']:
                color = (0, 165, 255)  # Оранжевый для транспорта
            elif det['class'] == 'building':
                color = (0, 255, 255)  # Жёлтый для зданий
            else:
                color = (255, 255, 0)  # Голубой для остальных (люди, животные)
            
            cv2.rectangle(nav_map, (x1, y1), (x2, y2), color, 2)
            # Добавить подпись с процентом уверенности
            label_text = class_names_ru.get(det['class'], det['class'])
            confidence_pct = det.get('confidence', 1.0) * 100
            label_with_conf = f"{label_text} {confidence_pct:.0f}%"
            labels_to_draw.append((label_with_conf, (x1, max(5, y1 - 18)), color))

    # Извлечь здания из детекций YOLO-World (вместо cv2-подхода)
    buildings_from_yolo = []
    for det in detections:
        if det['class'] == 'building':
            x1, y1, x2, y2 = det['box']
            confidence = det.get('confidence', 0.5)
            buildings_from_yolo.append((x1, y1, x2, y2, confidence))

    # Построить downsampled occupancy grid и использовать A* для маршрутизации.
    def find_path_astar(start_px, goal_px, obstacle_mask, scale=4):
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
                # Если больше 30% пикселей - препятствие, отметить заблокированным
                if np.sum(block > 0) > (scale * scale * 0.3):
                    matrix[gy][gx] = 1
        grid = Grid(matrix=matrix)
        start_node = grid.node(max(0, min(gw - 1, start_px[0] // scale)), max(0, min(gh - 1, start_px[1] // scale)))
        end_node = grid.node(max(0, min(gw - 1, goal_px[0] // scale)), max(0, min(gh - 1, goal_px[1] // scale)))
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
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
            # Используем здания из YOLO-World детекций (вместо cv2-подхода)
            buildings = buildings_from_yolo
            
            result_dict['buildings'] = buildings
            overlays = []
            labels = []
            worker_nav = nav_map.copy()
            print(f"YOLO-World обнаружил {len(buildings)} зданий")
            
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
                astar_scale = 4  # Более точный масштаб
                # Использовать расширенную маску для обхода с запасом
                path = find_path_astar((start_x, start_y), (center_x, center_y), obstacles_for_navigation, scale=astar_scale)
                if path:
                    path = smooth_path(path, window=5)  # Больше сглаживания
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
                cv2.rectangle(worker_nav, (bx1, by1), (bx2, by2), (0, 255, 255), 2)  # Желтые прямоугольники для зданий
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
        
        # Загрузить модель YOLO и создать временный объект для детекции
        try:
            # Создать временное изображение для инициализации
            temp_frame = cv2.imread(image_files[0])
            if temp_frame is None:
                print("Ошибка: не удалось загрузить первое изображение")
                return
            
            # Создать объект VideMosaic для использования его методов детекции
            temp_mosaic = VideMosaic(temp_frame, detector_type="sift", show_intermediate=False, output_dir=output_dir, visualize=False)
        except Exception as e:
            print(f"Ошибка инициализации детектора: {e}")
            return
        
        
        print(f"Найдено {len(image_files)} изображений для обработки")
        for image_path in image_files:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Не удалось загрузить {image_path}")
                continue
            
            detections = temp_mosaic.detect_objects(frame)
            
            # Создать навигационную карту с детекциями (С построением путей к зданиям)
            nav_map = analyze_for_navigation(frame, detections, compute_paths=True)
            
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
            video_path = 'Data/поиски квадрокоптера 2 (360p) 02.mp4'
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
    
    # Вывод статистики по классам
    class_counts = {}
    for det in all_detections:
        cls = det['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print("Статистика по классам:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cls}: {count}")

    # Опционально проанализировать мозаику для навигации: отметить препятствия и нарисовать пути к объектам
    # Закомментировано создание карты навигации по запросу пользователя
    print("Анализ мозаики для навигации...")
    # start_point=None заставит функцию использовать центр снизу текущего изображения (scaled_mosaic)
    navigation_map = analyze_for_navigation(scaled_mosaic.astype(np.uint8), all_detections, start_point=None)
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