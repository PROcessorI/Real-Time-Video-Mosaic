"""
Комбинированный детектор зданий: YOLO-World + OpenCV
Оптимизирован для аэроснимков и вида сверху
"""

import cv2
import numpy as np
from ultralytics import YOLO


class BuildingDetector:
    """Детектор зданий с комбинированным подходом YOLO + OpenCV."""
    
    def __init__(self, yolo_model_path='yolov8x-worldv2.pt'):
        """
        Args:
            yolo_model_path: путь к модели YOLO-World
        """
        # Загружаем YOLO-World
        try:
            self.model = YOLO(yolo_model_path)
            # Расширенный список классов для зданий
            self.building_classes = [
                # Основные постройки
                'building', 'house', 'home', 'roof', 'rooftop',
                'structure', 'construction', 'shed', 'barn', 
                'garage', 'warehouse', 'cabin', 'cottage',
                'hut', 'shack', 'shelter', 'tent', 'hangar',
                # Дополнительно
                'car', 'truck', 'vehicle', 'tractor',
                'road', 'path', 'parking lot',
                'tree', 'forest', 'field',
                'person', 'pool', 'fence'
            ]
            self.model.set_classes(self.building_classes)
            print(f"BuildingDetector: загружено {len(self.building_classes)} классов")
        except Exception as e:
            print(f"Ошибка загрузки YOLO: {e}")
            self.model = None
    
    def detect(self, frame, conf_threshold=0.01):
        """
        Комбинированная детекция зданий.
        
        Args:
            frame: входное изображение BGR
            conf_threshold: минимальный порог уверенности
            
        Returns:
            list: детекции в формате [{'class': str, 'box': tuple, 'confidence': float, 'source': str}]
        """
        all_detections = []
        
        # 1. YOLO-World детекция
        yolo_dets = self._detect_yolo(frame, conf_threshold)
        for d in yolo_dets:
            d['source'] = 'yolo'
        all_detections.extend(yolo_dets)
        
        # 2. OpenCV детекция зданий (серые крыши)
        cv_buildings = self._detect_buildings_opencv(frame)
        for d in cv_buildings:
            d['source'] = 'opencv'
            if not self._is_duplicate(all_detections, d['box']):
                all_detections.append(d)
        
        # 3. OpenCV детекция контуров (геометрические структуры)
        cv_contours = self._detect_structures_contours(frame)
        for d in cv_contours:
            d['source'] = 'contours'
            if not self._is_duplicate(all_detections, d['box']):
                all_detections.append(d)
        
        # 4. Edge-based detection (границы зданий)
        edge_dets = self._detect_by_edges(frame)
        for d in edge_dets:
            d['source'] = 'edges'
            if not self._is_duplicate(all_detections, d['box']):
                all_detections.append(d)
        
        # Фильтруем по уверенности > 10%
        filtered_detections = [d for d in all_detections if d.get('confidence', 0) > 0.1]
        
        return filtered_detections
    
    def _detect_yolo(self, frame, conf_threshold):
        """YOLO-World детекция."""
        detections = []
        if self.model is None:
            return detections
        
        try:
            # Основная детекция
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                imgsz=1280,
                verbose=False,
                augment=True
            )
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = self.model.names[cls_id]
                    
                    # Нормализуем класс
                    norm_class = self._normalize_class(class_name)
                    
                    detections.append({
                        'class': norm_class,
                        'box': (x1, y1, x2, y2),
                        'confidence': conf
                    })
            
            # Детекция на улучшенном изображении
            enhanced = self._enhance_contrast(frame)
            results2 = self.model.predict(
                enhanced,
                conf=conf_threshold,
                imgsz=1280,
                verbose=False
            )
            
            for r in results2:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = self.model.names[cls_id]
                    norm_class = self._normalize_class(class_name)
                    
                    if not self._is_duplicate(detections, (x1, y1, x2, y2)):
                        detections.append({
                            'class': norm_class,
                            'box': (x1, y1, x2, y2),
                            'confidence': conf * 0.95
                        })
                        
        except Exception as e:
            print(f"YOLO error: {e}")
        
        return detections
    
    def _detect_buildings_opencv(self, frame):
        """OpenCV детекция зданий по цвету и форме крыш."""
        detections = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Маска валидных областей (убираем чёрные границы)
        _, valid_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        frame_h, frame_w = gray.shape
        frame_area = frame_h * frame_w
        min_area = 500
        max_area = frame_area * 0.1
        
        # === 1. Серые крыши (низкая насыщенность) ===
        gray_roof_mask = cv2.inRange(s, 0, 60)  # Низкая насыщенность
        gray_roof_mask = cv2.bitwise_and(gray_roof_mask, cv2.inRange(v, 50, 220))
        gray_roof_mask = cv2.bitwise_and(gray_roof_mask, valid_mask)
        
        # === 2. Тёмные крыши ===
        dark_roof_mask = cv2.inRange(v, 30, 80)
        dark_roof_mask = cv2.bitwise_and(dark_roof_mask, valid_mask)
        
        # === 3. Светлые крыши (белые/бежевые) ===
        light_roof_mask = cv2.inRange(v, 180, 255)
        light_roof_mask = cv2.bitwise_and(light_roof_mask, cv2.inRange(s, 0, 80))
        light_roof_mask = cv2.bitwise_and(light_roof_mask, valid_mask)
        
        # === 4. Коричневые/красные крыши ===
        brown_mask1 = cv2.inRange(hsv, (0, 30, 50), (20, 150, 180))
        brown_mask2 = cv2.inRange(hsv, (160, 30, 50), (180, 150, 180))
        brown_roof_mask = cv2.bitwise_or(brown_mask1, brown_mask2)
        brown_roof_mask = cv2.bitwise_and(brown_roof_mask, valid_mask)
        
        # Объединяем маски
        all_masks = [
            ('gray', gray_roof_mask),
            ('dark', dark_roof_mask),
            ('light', light_roof_mask),
            ('brown', brown_roof_mask)
        ]
        
        # Морфологическое ядро
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        for mask_name, mask in all_masks:
            # Очистка маски
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Watershed для разделения слипшихся зданий
            mask_separated = self._apply_watershed(frame, mask_clean)
            
            # Находим контуры
            contours, _ = cv2.findContours(mask_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Минимальный размер
                if min(w, h) < 20:
                    continue
                
                # Aspect ratio
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect > 5:
                    continue
                
                # Прямоугольность
                rect = cv2.minAreaRect(cnt)
                rect_area = rect[1][0] * rect[1][1]
                if rect_area == 0:
                    continue
                rectangularity = area / rect_area
                if rectangularity < 0.4:
                    continue
                
                # Аппроксимация контура
                eps = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                if len(approx) < 4:
                    continue
                
                # Вычисляем уверенность
                confidence = min(0.6, rectangularity * 0.4 + 0.15)
                
                detections.append({
                    'class': 'building',
                    'box': (x, y, x + w, y + h),
                    'confidence': confidence
                })
        
        return detections
    
    def _detect_structures_contours(self, frame):
        """Детекция структур через анализ контуров."""
        detections = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Убираем чёрные границы
        _, valid_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        # Адаптивная бинаризация
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 2
        )
        adaptive = cv2.bitwise_and(adaptive, valid_mask)
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_area = gray.shape[0] * gray.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800 or area > frame_area * 0.08:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            if min(w, h) < 25:
                continue
            
            # Только прямоугольные объекты
            rect = cv2.minAreaRect(cnt)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area == 0:
                continue
            
            rectangularity = area / rect_area
            if rectangularity < 0.6:  # Только очень прямоугольные
                continue
            
            aspect = max(w, h) / min(w, h)
            if aspect > 3.5:
                continue
            
            detections.append({
                'class': 'building',
                'box': (x, y, x + w, y + h),
                'confidence': rectangularity * 0.4
            })
        
        return detections
    
    def _detect_by_edges(self, frame):
        """Детекция зданий через границы (Canny + Hough)."""
        detections = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edges
        edges = cv2.Canny(blurred, 30, 100)
        
        # Находим прямоугольные контуры
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Морфология для соединения близких границ
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours_closed, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_area = gray.shape[0] * gray.shape[1]
        
        for cnt in contours_closed:
            area = cv2.contourArea(cnt)
            if area < 1000 or area > frame_area * 0.1:
                continue
            
            # Аппроксимируем до многоугольника
            eps = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            
            # Здания имеют 4-8 углов
            if 4 <= len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(cnt)
                
                if min(w, h) < 30:
                    continue
                
                aspect = max(w, h) / min(w, h)
                if aspect > 4:
                    continue
                
                # Прямоугольность
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                
                if extent > 0.5:
                    detections.append({
                        'class': 'building',
                        'box': (x, y, x + w, y + h),
                        'confidence': extent * 0.35
                    })
        
        return detections
    
    def _apply_watershed(self, frame, mask):
        """Применяет Watershed для разделения слипшихся объектов."""
        if cv2.countNonZero(mask) == 0:
            return mask
        
        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Нормализуем и находим пики (центры объектов)
        _, sure_fg = cv2.threshold(dist, 0.25 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Уверенный фон
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        
        # Неизвестная область
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Маркеры
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed
        frame_bgr = frame.copy()
        markers = cv2.watershed(frame_bgr, markers)
        
        # Результат
        result = np.zeros_like(mask)
        result[markers > 1] = 255
        result[markers == -1] = 0  # Границы watershed
        
        return result
    
    def _enhance_contrast(self, frame):
        """Улучшение контраста для лучшей детекции."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _normalize_class(self, class_name):
        """Нормализует название класса."""
        building_names = ['building', 'house', 'home', 'roof', 'rooftop', 
                         'structure', 'shed', 'barn', 'garage', 'warehouse',
                         'cabin', 'cottage', 'hut', 'shack', 'shelter', 'tent',
                         'hangar', 'construction']
        if class_name.lower() in building_names:
            return 'building'
        
        vehicle_names = ['car', 'vehicle', 'automobile', 'van']
        if class_name.lower() in vehicle_names:
            return 'car'
        
        return class_name.lower()
    
    def _is_duplicate(self, detections, box, iou_threshold=0.4):
        """Проверяет дубликат по IoU."""
        x1, y1, x2, y2 = box
        
        for d in detections:
            dx1, dy1, dx2, dy2 = d['box']
            
            # IoU
            ix1 = max(x1, dx1)
            iy1 = max(y1, dy1)
            ix2 = min(x2, dx2)
            iy2 = min(y2, dy2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (dx2 - dx1) * (dy2 - dy1)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                
                if iou > iou_threshold:
                    return True
        
        return False
    
    def visualize(self, frame, detections, output_path=None):
        """Визуализирует детекции на изображении."""
        result = frame.copy()
        
        # Цвета для разных источников
        colors = {
            'yolo': (0, 255, 0),      # Зелёный
            'opencv': (255, 0, 0),     # Синий
            'contours': (0, 255, 255), # Жёлтый
            'edges': (255, 0, 255)     # Фиолетовый
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls = det['class']
            conf = det['confidence']
            source = det.get('source', 'unknown')
            
            color = colors.get(source, (128, 128, 128))
            
            # Рисуем рамку
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Подпись
            label = f"{cls} {conf:.0%} [{source}]"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Сохранено: {output_path}")
        
        return result


def test_building_detector():
    """Тестирует детектор на mosaic.jpg"""
    import os
    
    print("="*60)
    print("🏢 КОМБИНИРОВАННЫЙ ДЕТЕКТОР ЗДАНИЙ")
    print("="*60)
    
    detector = BuildingDetector()
    
    # Тестируем на mosaic.jpg
    image_path = 'mosaic.jpg'
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден!")
        return
    
    frame = cv2.imread(image_path)
    print(f"Изображение: {frame.shape}")
    
    # Детекция
    detections = detector.detect(frame, conf_threshold=0.01)
    
    # Статистика
    print(f"\nВсего детекций: {len(detections)}")
    
    by_class = {}
    by_source = {}
    for d in detections:
        cls = d['class']
        src = d.get('source', 'unknown')
        by_class[cls] = by_class.get(cls, 0) + 1
        by_source[src] = by_source.get(src, 0) + 1
    
    print("\nПо классам:")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    
    print("\nПо источникам:")
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")
    
    # Только здания
    buildings = [d for d in detections if d['class'] == 'building']
    print(f"\n🏠 Всего зданий: {len(buildings)}")
    
    # Визуализация
    os.makedirs('test_output', exist_ok=True)
    output_path = 'test_output/buildings_combined.jpg'
    detector.visualize(frame, detections, output_path)
    
    print(f"\n✅ Результат: {output_path}")


if __name__ == "__main__":
    test_building_detector()
