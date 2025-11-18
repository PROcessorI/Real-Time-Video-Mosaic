import cv2
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from PIL import Image, ImageDraw, ImageFont
import os
import sys


class VideMosaic:
    """This class processes every frame and generates the panorama

    Args:
        first_image (image for the first frame): first image to initialize the output size
        output_height_times (int, optional): determines the output height based on input image height. Defaults to 2.
        output_width_times (int, optional): determines the output width based on input image width. Defaults to 4.
        detector_type (str, optional): the detector for feature detection. It can be "sift" or "orb". Defaults to "sift".
    """
    def __init__(self, first_image, output_height_times=1, output_width_times=4, detector_type="sift", update_callback=None, show_intermediate=True):
        self.update_callback = update_callback
        self.detector_type = detector_type
        self.show_intermediate = show_intermediate
        self.quit_requested = False
        if detector_type == "sift":
            # Keep feature count moderate to avoid excessive memory use
            self.detector = cv2.SIFT_create(300)
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(700)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # enable visualization for intermediate results
        self.visualize = True

        # Initialize YOLO model for detection with larger model for better accuracy
        try:
            self.model = YOLO('yolov8l.pt')  # Using large model for maximum detection quality
        except Exception as e:
            print(f"Warning: failed to load YOLO model: {e}")
            self.model = None

        self.process_first_frame(first_image)

        self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), int(output_width_times * first_image.shape[1]), first_image.shape[2]), dtype=np.uint8)

        # offsets to place first image at left
        self.w_offset = 0
        self.h_offset = 0

        self.output_img[self.w_offset:self.w_offset+first_image.shape[0],
                        self.h_offset:self.h_offset+first_image.shape[1], :] = first_image

        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset
        
        # Initialize OpenCV window for intermediate visualization if enabled
        if self.show_intermediate:
            cv2.namedWindow('Mosaic Progress', cv2.WINDOW_FULLSCREEN)
            cv2.namedWindow('Current Frame', cv2.WINDOW_AUTOSIZE)
            print("OpenCV windows 'Mosaic Progress' and 'Current Frame' created")

    def process_first_frame(self, first_image):
        """processes the first frame for feature detection and description

        Args:
            first_image (cv2 image/np array): first image for feature detection
        """
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def detect_people(self, frame):
        if self.model is None:
            return []
        # Use higher confidence threshold and optimized parameters for better quality
        results = self.model.predict(
            frame, 
            classes=[0],  # class 0 is 'person'
            conf=0.5,      # confidence threshold - only detections with 50%+ confidence
            iou=0.45,      # IoU threshold for NMS - reduces overlapping boxes
            imgsz=640,     # image size for detection - larger for better quality
            verbose=False  # reduce console output
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
        # Use optimized parameters for maximum detection quality
        results = self.model.predict(
            frame,
            conf=0.4,      # confidence threshold - filter out low-confidence detections
            iou=0.45,      # IoU threshold for NMS - reduces duplicate detections
            imgsz=640,     # image size for detection - larger for better accuracy
            verbose=False  # reduce console output
        )
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])  # Get confidence score
                class_name = self.model.names[class_id] if hasattr(self.model, 'names') else str(class_id)
                detections.append({
                    'class': class_name, 
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence
                })
        return detections

    def match(self, des_cur, des_prev):
        if des_cur is None or des_prev is None:
            return []
        if self.detector_type == "sift":
            pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
            matches = []
            for m, n in pair_matches:
                if m.distance < 0.7*n.distance:
                    matches.append(m)
        elif self.detector_type == "orb":
            matches = self.bf.match(des_cur, des_prev)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:min(len(matches), 20)]
        return matches

    def process_frame(self, frame_cur, frame_count):
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)
        self.matches = self.match(self.des_cur, self.des_prev)
        print(f"Frame {frame_count}: matches = {len(self.matches)}")
        if len(self.matches) < 4:
            # Display current mosaic state even if frame registration fails
            if self.show_intermediate:
                cv2.imshow('Mosaic Progress', self.output_img)
                cv2.imshow('Current Frame', self.frame_cur)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit_requested = True
                    return
            return
        self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
        self.H = np.matmul(self.H_old, self.H)
        self.warp(self.frame_cur, self.H)
        people_boxes = self.detect_people(self.frame_cur)
        for box in people_boxes:
            x1, y1, x2, y2 = box
            corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
            transformed_corners = cv2.perspectiveTransform(corners, self.H)
            try:
                cv2.rectangle(self.output_img, tuple(transformed_corners[0][0].astype(int)), tuple(transformed_corners[0][2].astype(int)), (0, 255, 0), 2)
            except Exception:
                pass
        detections = self.detect_objects(self.frame_cur)
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(self.frame_cur, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Display class name and confidence score
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(self.frame_cur, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if detections:
            os.makedirs('Detections', exist_ok=True)
            cv2.imwrite(os.path.join('Detections', f'frame_{frame_count}.jpg'), self.frame_cur)
        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur
        self.last_frame = self.frame_cur.copy()
        
        # Display current mosaic state if visualization is enabled
        if self.show_intermediate:
            cv2.imshow('Mosaic Progress', self.output_img)
            cv2.imshow('Current Frame', self.frame_cur)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.quit_requested = True
                return
            elif key == 27:  # ESC key
                self.quit_requested = True
                return

    @ staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt
        homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)
        return homography

    def warp(self, frame_cur, H):
        warped_img = cv2.warpPerspective(frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR)
        # transformed_corners = self.get_transformed_corners(frame_cur, H)
        # warped_img = self.draw_border(warped_img, transformed_corners)
        self.output_img[warped_img > 0] = warped_img[warped_img > 0]
        return self.output_img

    @ staticmethod
    def get_transformed_corners(frame_cur, H):
        corner_0 = np.array([0, 0])
        corner_1 = np.array([frame_cur.shape[1], 0])
        corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner_3 = np.array([0, frame_cur.shape[0]])
        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        transformed_corners = np.array(transformed_corners, dtype=np.int32)
        return transformed_corners

    def draw_border(self, image, corners, color=(0, 0, 255)):
        for i in range(corners.shape[1]-1, -1, -1):
            cv2.line(image, tuple(corners[0, i, :]), tuple(corners[0, i-1, :]), thickness=5, color=color)
        return image





def is_path_clear(x1, y1, x2, y2, obstacles):
    num_samples = 20
    for i in range(num_samples + 1):
        t = i / num_samples
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        if 0 <= px < obstacles.shape[1] and 0 <= py < obstacles.shape[0]:
            if obstacles[py, px] > 0:
                return False
    return True


def crop_black_areas(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]


def draw_dotted_line(img, pt1, pt2, color, thickness):
    dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
    num_dots = max(1, int(dist / 10))
    for i in range(num_dots):
        t = i / num_dots
        px = int(pt1[0] + t * (pt2[0] - pt1[0]))
        py = int(pt1[1] + t * (pt2[1] - pt1[1]))
        cv2.circle(img, (px, py), thickness, color, -1)


def main(video_path=None, update_callback=None, show_intermediate=True):

    if video_path is None:
        video_path = 'Data/поиски квадрокоптера 2 (360p) 03.mp4'
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл")
        return

    # Create Detections folder
    os.makedirs('Detections', exist_ok=True)

    # Get total frame count for progress calculation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    frame_count = 0
    is_first_frame = True
    print("Starting video processing and mosaic formation...")
    
    while cap.isOpened():
        ret, frame_cur = cap.read()
        if not ret:
            if is_first_frame:
                continue
            break

        if is_first_frame:
            video_mosaic = VideMosaic(frame_cur, detector_type="sift", update_callback=update_callback, show_intermediate=show_intermediate)
            is_first_frame = False
            continue

        frame_count += 1
        # process each frame
        video_mosaic.process_frame(frame_cur, frame_count)
        
        # Print progress every 50 frames
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed frame {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # Check if user requested to quit during processing
        if hasattr(video_mosaic, 'quit_requested') and video_mosaic.quit_requested:
            print("Processing interrupted by user.")
            break
        
        # Update progress if callback is provided
        if update_callback:
            progress = (frame_count / total_frames) * 100
            update_callback(frame_count, video_mosaic.output_img.copy(), progress)
            
    cap.release()
    print("Video processing completed. Mosaic formed.")
    
    # Keep the final mosaic displayed until user closes it
    if show_intermediate:
        print("Final mosaic completed. Press any key to close the window.")
        cv2.imshow('Mosaic Progress', video_mosaic.output_img)
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        
    print("Saving mosaic image...")
    cv2.imwrite('mosaic.jpg', video_mosaic.output_img)
    print("Mosaic saved as 'mosaic.jpg'")

    # Detect objects on the mosaic
    print("Detecting objects on the mosaic...")
    detections = video_mosaic.detect_objects(video_mosaic.output_img.astype(np.uint8))
    print(f"Detected {len(detections)} objects on the mosaic.")

    # Optionally analyze mosaic for navigation: mark obstacles and draw paths to objects
    # Commenting out the navigation map creation as per user request
    print("Analyzing mosaic for navigation...")
    navigation_map = analyze_for_navigation(video_mosaic.output_img.astype(np.uint8), detections)
    print("Saving navigation map...")
    cv2.imwrite('navigation_map.jpg', navigation_map)
    print("Navigation map saved as 'navigation_map.jpg'")
    if show_intermediate:
        cv2.imshow('Navigation Map', navigation_map)
        print("Navigation map displayed. Press any key in the window to continue.")
    
    # Close all OpenCV windows at the end of the program
    cv2.destroyAllWindows()
    
    # Check if processing was interrupted by user
    if hasattr(video_mosaic, 'quit_requested') and video_mosaic.quit_requested:
        print("Video mosaic generation was interrupted by user.")
        return
    
    # Final update with completion
    if update_callback:
        update_callback(frame_count, video_mosaic.output_img.copy(), 100)


def analyze_for_navigation(frame, detections, start_point=None):
    """Simple analysis for navigation: mark obstacles and draw paths to objects on a single frame.

    Args:
        frame (np array): the last frame
        detections (list): list of detected objects

    Returns:
        np array: frame with marked obstacles and paths
    """
    labels_to_draw = []
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for obstacles (simplified)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    obstacles = cv2.bitwise_or(mask_green, mask_blue)

    # Mark obstacles as contours on frame copy
    nav_map = frame.copy()
    contours, _ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(nav_map, contours, -1, (0, 0, 255), 2)  # Red contours for obstacles

    # interactive start selection with separate OpenCV window was removed
    # Use GUIApp.toggle_set_start and click on the navigation canvas to set a start point in-app.

    # default center point (bottom-center) unless provided by GUI
    default_start = (frame.shape[1] // 2, frame.shape[0] - 50)
    if start_point is not None:
        start_x, start_y = start_point
    else:
        start_x, start_y = default_start

    # Mark start position
    cv2.circle(nav_map, (start_x, start_y), 10, (255, 255, 255), -1)  # White circle for start

    # Use PIL to add text with Russian support
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
        # last resort: default font (ограниченная кириллица возможна / может показывать '?')
        try:
            font = ImageFont.load_default()
            print("Warning: TTF font not found; using default font which may not support Cyrillic fully.")
        except Exception:
            font = None

    # If font is still None, PIL will use a fallback when drawing, but log for the user.
    if font is None:
        print("Warning: No font available for drawing text. Русский текст может не отображаться.")

    # Draw legend and start label (PIL handles Unicode if font supports it)
    try:
        draw.text((10, 30), "Красные контуры: препятствия", fill=(255, 0, 0), font=font)
        draw.text((10, 60), "Зелёные линии: пути к объектам", fill=(0, 255, 0), font=font)
        draw.text((10, 90), "Жёлтые прямоугольники: обнаруженные объекты", fill=(255, 255, 0), font=font)
    except Exception as e:
        # As a graceful fallback, try OpenCV text (may show '?') and print the error
        print(f"Error drawing with PIL font: {e}")
        cv2.putText(nav_map, "Старт", (start_x + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(nav_map, "Красные контуры: препятствия", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(nav_map, "Зелёные линии: пути к объектам", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(nav_map, "Жёлтые прямоугольники: обнаруженные объекты", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    nav_map = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Mark objects on the map without drawing routes
    for det in detections:
        if det['class'] in ['person', 'car', 'truck', 'dog', 'horse', 'cat']:  # Key objects
            x1, y1, x2, y2 = det['box']
            # Mark object
            cv2.rectangle(nav_map, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for objects

    # --- New: detect building-like targets on the mosaic (rectangular, large contours)
    def detect_buildings(img, min_area=200):
        # Try a more robust multi-step approach to find large man-made rectangular shapes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive threshold to separate objects with different lighting
        try:
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)
        except Exception:
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological closing to merge roof regions and remove small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Edge detection on closed image
        edges = cv2.Canny(closed, 30, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        buildings = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            if rect_area <= 0:
                continue
            fill_ratio = area / float(rect_area)
            # Accept roughly rectangular-ish large contours OR tall/wide boxes
            if fill_ratio > 0.1 and w > 10 and h > 10 and len(approx) >= 4 and len(approx) <= 8:
                buildings.append((x, y, x + w, y + h))
        return buildings

    # Build a downsampled occupancy grid and use A* to route.
    def find_path_astar(start_px, goal_px, obstacle_mask, scale=8):
        # Create matrix where 0 = walkable, 1 = blocked
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
                # If any obstacle pixel present, mark blocked
                if np.any(block > 0):
                    matrix[gy][gx] = 1
        grid = Grid(matrix=matrix)
        start_node = grid.node(max(0, min(gw - 1, start_px[0] // scale)), max(0, min(gh - 1, start_px[1] // scale)))
        end_node = grid.node(max(0, min(gw - 1, goal_px[0] // scale)), max(0, min(gh - 1, goal_px[1] // scale)))
        finder = AStarFinder(diagonal_movement=True)
        path, runs = finder.find_path(start_node, end_node, grid)
        if not path:
            return None
        # Convert path back to pixel coordinates (center of cell)
        pixel_path = []
        for gx, gy in path:
            px = int(gx * scale + scale // 2)
            py = int(gy * scale + scale // 2)
            pixel_path.append((px, py))
        return pixel_path

    def smooth_path(path, window=3):
        """Simple moving-average smoothing over the path coordinates."""
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

    # Compute paths and building detections synchronously
    def worker_compute_paths(result_dict):
        try:
            print("Computing paths...")
            buildings = detect_buildings(frame)
            # Limit to first 5 buildings for faster processing
            buildings = buildings[:5]
            result_dict['buildings'] = buildings
            overlays = []
            labels = []
            worker_nav = nav_map.copy()
            print(f"Detected {len(buildings)} building(s)")
            for (bx1, by1, bx2, by2) in buildings:
                center_x = (bx1 + bx2) // 2
                center_y = (by1 + by2) // 2
                astar_scale = 8  # Increased scale for faster computation
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
                labels.append(("Здание", (bx1, max(5, by1 - 18)), (0, 255, 255)))
            result_dict['nav_overlay'] = worker_nav
            result_dict['labels'] = labels
        except Exception as e:
            print(f"Worker error: {e}")
            result_dict['nav_overlay'] = nav_map.copy()
            result_dict['labels'] = []

    result = {}
    worker_compute_paths(result)

    # If worker finished copy overlay and labels
    nav_map = result.get('nav_overlay', nav_map)
    for lab in result.get('labels', []):
        labels_to_draw.append(lab)

    # Final step: draw all queued labels with PIL (gives correct Unicode rendering)
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

        for text, (tx, ty), color in labels_to_draw:
            # draw a thin black outline for readability
            if use_font is not None:
                outline_color = (0, 0, 0)
                # offsets for pseudo-outline
                for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw_final.text((tx + ox, ty + oy), text, font=use_font, fill=outline_color)
                draw_final.text((tx, ty), text, font=use_font, fill=tuple(color))
            else:
                # fallback: do nothing
                pass

        nav_map = cv2.cvtColor(np.array(pil_final), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Warning: failed to draw labels with PIL: {e}")

    return nav_map

def is_path_clear(x1, y1, x2, y2, obstacles):
    """Check if the line between two points avoids obstacles."""
    # Simple check: sample points along the line
    num_samples = 20
    for i in range(num_samples + 1):
        t = i / num_samples
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        if 0 <= px < obstacles.shape[1] and 0 <= py < obstacles.shape[0]:
            if obstacles[py, px] > 0:
                return False
    return True

def crop_black_areas(image):
    """Crop black areas from the image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find non-zero pixels
    coords = cv2.findNonZero(gray)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def draw_dotted_line(img, pt1, pt2, color, thickness):
    """Draw a dotted line between two points."""
    dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
    num_dots = int(dist / 10)
    for i in range(num_dots):
        t = i / num_dots
        px = int(pt1[0] + t * (pt2[0] - pt1[0]))
        py = int(pt1[1] + t * (pt2[1] - pt1[1]))
        cv2.circle(img, (px, py), thickness, color, -1)

if __name__ == "__main__":
    vid = None
    if len(sys.argv) > 1:
        vid = sys.argv[1]
    main(vid, show_intermediate=True)
