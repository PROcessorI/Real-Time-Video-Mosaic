# Real-Time Video Mosaic

A system for creating panoramas from videos in real-time with object detection and navigation map generation.

## Description

Real-Time Video Mosaic is a project for creating panoramic images (mosaics) from video files using computer vision. The system automatically stitches sequential video frames into a single panorama, detects objects (people, vehicles, etc.) using YOLOv8, and creates navigation maps with pathfinding to detected objects.

## Features

- ✅ Real-time panorama creation from videos
- ✅ Support for SIFT and ORB feature detectors
- ✅ Object detection using YOLOv8 (people, vehicles, animals, etc.)
- ✅ Automatic navigation map generation
- ✅ A* algorithm for pathfinding with obstacle avoidance
- ✅ Graphical User Interface (GUI) built with CustomTkinter
- ✅ Command-line interface (CLI) for automation
- ✅ Real-time mosaic formation visualization
- ✅ Save results as images

## System Requirements

### Minimum Requirements:
- **OS:** Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python:** 3.8 or newer
- **RAM:** 8 GB (16 GB recommended)
- **Disk Space:** 2 GB

### Recommended Requirements:
- **RAM:** 16 GB or more
- **GPU:** NVIDIA GPU with CUDA support (for YOLOv8 acceleration)
- **CPU:** Intel Core i5 or AMD Ryzen 5 and above

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/PROcessorI/Real-Time-Video-Mosaic.git
cd Real-Time-Video-Mosaic
```

### 2. Create a virtual environment (recommended)

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### List of dependencies:
- `opencv-python` - computer vision library
- `numpy` - mathematical operations with arrays
- `ultralytics` - YOLOv8 for object detection
- `pathfinding` - pathfinding algorithms (A*)
- `Pillow` - image processing
- `customtkinter` - modern Python GUI

### 4. Download YOLOv8 models

YOLOv8 models are downloaded automatically on first run. The project uses `yolov8n.pt` (nano model) by default. The repository already includes models:
- `yolov8n.pt` - fastest, used by default
- `yolov8s.pt` - small model
- `yolov8l.pt` - large model (more accurate but slower)

## Usage

### Graphical User Interface (GUI)

Launch the application with GUI:

```bash
python gui.py
```

**Instructions:**
1. Click **"Выбрать видео"** (Select Video) button and choose a video file
2. Click **"Запустить обработку"** (Start Processing)
3. Watch the real-time mosaic formation process
4. After processing completes, you'll see:
   - Final panorama
   - Navigation map with routes
   - List of frames with detected objects

### Command-Line Interface (CLI)

Run video processing via command line:

```bash
python main.py
```

By default, the program will process the video `Data/поиски квадрокоптера 2 (360p) 03.mp4`.

#### Change input video

Edit `main.py` file, changing the video path in the `main()` function:

```python
if __name__ == "__main__":
    main(video_path='path/to/your/video.mp4')
```

Or modify line 639 in `main.py`:
```python
video_path = 'Data/поиски квадрокоптера 2 (360p) 03.mp4'
```

### Results

After processing completes, the following files will appear in the root directory:

- **`mosaic.jpg`** - final panorama
- **`navigation_map.jpg`** - navigation map with routes
- **`Detections/`** - folder with frames containing detected objects

## Project Structure

```
Real-Time-Video-Mosaic/
├── main.py                 # Main video processing module
├── gui.py                  # Graphical user interface
├── requirements.txt        # Dependencies list
├── Data/                   # Folder with video files
│   ├── поиски квадрокоптера 2 (360p) 01.mp4
│   ├── поиски квадрокоптера 2 (360p) 02.mp4
│   └── поиски квадрокоптера 2 (360p) 03.mp4
├── Detections/             # Auto-created - frames with detections
├── yolov8n.pt              # YOLOv8 nano model
├── yolov8s.pt              # YOLOv8 small model
├── yolov8l.pt              # YOLOv8 large model
├── mosaic.jpg              # Result - panorama (created after processing)
└── navigation_map.jpg      # Result - navigation map
```

## Main Modules Description

### `main.py`

Main module containing:

- **`VideMosaic`** - class for creating video panoramas
  - `__init__()` - initialize feature detector (SIFT/ORB) and YOLOv8 model
  - `process_frame()` - process each video frame
  - `detect_people()` - detect people in frame
  - `detect_objects()` - detect various objects
  - `findHomography()` - compute homography between frames
  - `warp()` - transform frame and add to mosaic

- **`analyze_for_navigation()`** - analyze mosaic to build navigation map
  - Obstacle detection (by color)
  - Building and large object detection
  - Pathfinding using A*
  - Route visualization

- **`main()`** - main video processing function

### `gui.py`

Graphical interface built with CustomTkinter:

- Video file selection
- Processing progress display
- Real-time current mosaic visualization
- Final results display
- Detected objects viewer

## Configuration

### Change feature detector

In `main.py`, line 666:
```python
video_mosaic = VideMosaic(frame_cur, detector_type="sift")
```

Options: `"sift"` (recommended) or `"orb"` (faster but less accurate).

### Change output mosaic size

In `main.py`, line 12:
```python
def __init__(self, first_image, output_height_times=3, output_width_times=1.2, ...):
```

- `output_height_times` - height multiplier (default 3)
- `output_width_times` - width multiplier (default 1.2)

### Change YOLOv8 model

In `main.py`, line 35:
```python
self.model = YOLO('yolov8n.pt')  # Can change to yolov8s.pt or yolov8l.pt
```

Available models:
- `yolov8n.pt` - nano (fast, ~3MB)
- `yolov8s.pt` - small (~11MB)
- `yolov8m.pt` - medium (~25MB) - needs separate download
- `yolov8l.pt` - large (~43MB, more accurate)
- `yolov8x.pt` - extra large (~68MB) - needs separate download

### Configure detection thresholds

In `main.py`, `detect_objects()` method, line 102:
```python
results = self.model.predict(
    resized,
    conf=0.4,      # confidence threshold (0.0-1.0)
    iou=0.45,      # IoU threshold for NMS
    imgsz=640,     # image size for detection
    verbose=False
)
```

## Usage Examples

### Example 1: Process your own video via CLI

1. Place video file in `Data/` folder
2. Open `main.py`
3. Change line 639:
   ```python
   video_path = 'Data/your_file.mp4'
   ```
4. Run:
   ```bash
   python main.py
   ```

### Example 2: Using GUI

```bash
python gui.py
```
Select video through interface and click "Start Processing".

### Example 3: Programmatic usage

```python
from main import VideMosaic, main
import cv2

# Method 1: Using main function
main(video_path='Data/your_video.mp4', show_intermediate=False)

# Method 2: Programmatic use of VideMosaic class
cap = cv2.VideoCapture('Data/your_video.mp4')
ret, first_frame = cap.read()

mosaic = VideMosaic(first_frame, detector_type="sift")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    mosaic.process_frame(frame, frame_count=0)

cap.release()
cv2.imwrite('result.jpg', mosaic.output_img)
```

## Troubleshooting

### Issue: "Error: Could not open video file"

**Solution:**
- Check that the video file path is correct
- Ensure the file exists
- Check video format (supported: mp4, avi, mov)

### Issue: "Warning: failed to load YOLO model"

**Solution:**
1. Ensure `ultralytics` package is installed:
   ```bash
   pip install ultralytics
   ```
2. Check for model file `yolov8n.pt` in root directory
3. On first run, model downloads automatically - internet connection required

### Issue: Slow video processing

**Solution:**
- Use lighter YOLO model (`yolov8n.pt` instead of `yolov8l.pt`)
- Reduce input video resolution
- Use GPU (install `torch` with CUDA support):
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### Issue: Errors with Cyrillic in paths

**Solution:**
- Use English file and path names
- Or ensure system locale supports UTF-8

### Issue: GUI doesn't start

**Solution:**
1. Check that `customtkinter` is installed:
   ```bash
   pip install customtkinter
   ```
2. On Linux, tkinter installation may be required:
   ```bash
   sudo apt-get install python3-tk
   ```

### Issue: Black areas in mosaic

This is normal behavior - black areas appear where there was no video data. The `crop_black_areas()` function automatically crops them in the final result.

### Issue: Insufficient memory

**Solution:**
- Reduce `output_height_times` and `output_width_times` parameters
- Process lower resolution video
- Close other applications

## Additional Information

### Algorithms Used

- **SIFT/ORB** - for keypoint extraction and matching
- **RANSAC** - for robust homography computation
- **YOLOv8** - for real-time object detection
- **A*** - for optimal pathfinding on navigation map
- **Perspective Transform** - for frame transformation and stitching

### Video Formats

All formats supported by OpenCV are supported:
- MP4 (H.264, HEVC)
- AVI
- MOV
- MKV
- WEBM

### Performance

Typical processing speed on Intel Core i5 + 16GB RAM:
- 360p video: ~10-15 FPS
- 720p video: ~5-8 FPS
- 1080p video: ~2-4 FPS

With GPU (CUDA), speed can be 3-5x faster.

## License

This project is open source. You are free to use, modify, and distribute it.

## Contact and Support

If you have questions or issues:
- Create an Issue on GitHub: https://github.com/PROcessorI/Real-Time-Video-Mosaic/issues
- Submit a Pull Request with improvements

## Acknowledgments

This project uses the following libraries:
- OpenCV - computer vision
- Ultralytics YOLOv8 - object detection
- CustomTkinter - modern GUI
- pathfinding - pathfinding algorithms

---

**Version:** 1.0  
**Last Updated:** 2024
