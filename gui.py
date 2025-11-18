import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import threading
import cv2
import main  # Import the main script

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Real-Time Video Mosaic")
        self.geometry("1200x800")

        # Variables
        self.video_path = None
        self.mosaic_image = None
        self.nav_image = None

        # UI Elements
        self.select_button = ctk.CTkButton(self, text="Выбрать видео", command=self.select_video)
        self.select_button.pack(pady=10)

        self.run_button = ctk.CTkButton(self, text="Запустить обработку", command=self.run_processing)
        self.run_button.pack(pady=10)

        # Progress bar
        self.progress_label = ctk.CTkLabel(self, text="Прогресс: 0%")
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(pady=5, padx=20, fill="x")
        self.progress_bar.set(0)

        # Frames for images
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Current mosaic display
        self.current_mosaic_frame = ctk.CTkFrame(self.image_frame)
        self.current_mosaic_frame.pack(side="left", padx=10, pady=10)
        
        self.current_mosaic_label = ctk.CTkLabel(self.current_mosaic_frame, text="Текущая мозаика")
        self.current_mosaic_label.pack()
        
        self.current_mosaic_display = ctk.CTkLabel(self.current_mosaic_frame, text="Ожидание обработки...")
        self.current_mosaic_display.pack()

        # Final mosaic and navigation map
        self.results_frame = ctk.CTkFrame(self.image_frame)
        self.results_frame.pack(side="right", padx=10, pady=10)
        
        self.mosaic_label = ctk.CTkLabel(self.results_frame, text="Финальная мозаика")
        self.mosaic_label.pack()
        
        self.nav_label = ctk.CTkLabel(self.results_frame, text="Навигационная карта")
        self.nav_label.pack()

        # Detections list
        self.detections_frame = ctk.CTkScrollableFrame(self, width=300, height=400)
        self.detections_frame.pack(side="right", fill="y", padx=10, pady=10)

        self.detections_label = ctk.CTkLabel(self.detections_frame, text="Обнаруженные кадры")
        self.detections_label.pack()

        self.detections_list = ctk.CTkScrollableFrame(self.detections_frame)
        self.detections_list.pack(fill="both", expand=True)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.select_button.configure(text=f"Выбрано: {os.path.basename(self.video_path)}")

    def run_processing(self):
        if not self.video_path:
            tk.messagebox.showerror("Ошибка", "Сначала выберите видео!")
            return

        # Run in thread to avoid freezing UI
        threading.Thread(target=self.process_video).start()

    def process_video(self):
        # Modify main.py to use self.video_path
        # For simplicity, assume main.main() is modified to take video_path
        # Here, we can call main.main() but need to set the path
        # Since main is hardcoded, perhaps modify main to accept argument

        # For now, assume we run main.main() and then load results
        main.main(self.video_path, update_callback=self.update_progress)  # Pass the callback

        # Load results
        self.load_results()

    def load_results(self):
        # Load mosaic.jpg
        if os.path.exists("mosaic.jpg"):
            img = Image.open("mosaic.jpg")
            img = img.resize((400, 300))
            self.mosaic_image = ctk.CTkImage(img, size=(400, 300))
            self.mosaic_label.configure(image=self.mosaic_image, text="")

        # Load navigation_map.jpg
        if os.path.exists("navigation_map.jpg"):
            img = Image.open("navigation_map.jpg")
            img = img.resize((400, 300))
            self.nav_image = ctk.CTkImage(img, size=(400, 300))
            self.nav_label.configure(image=self.nav_image, text="")

        # Load detections
        detections_dir = "Detections"
        if os.path.exists(detections_dir):
            for widget in self.detections_list.winfo_children():
                widget.destroy()
            for file in os.listdir(detections_dir):
                if file.endswith(".jpg"):
                    btn = ctk.CTkButton(self.detections_list, text=file, command=lambda f=file: self.show_detection(f))
                    btn.pack(pady=5)
                    
    def update_progress(self, frame_count, current_mosaic, progress):
        # Update progress bar in the main thread
        self.after(0, self._update_progress_ui, frame_count, current_mosaic, progress)
        
    def _update_progress_ui(self, frame_count, current_mosaic, progress):
        # Update the progress bar and label
        self.progress_bar.set(progress / 100)
        self.progress_label.configure(text=f"Прогресс: {progress:.1f}% (Кадр {frame_count})")
        
        # Update the current mosaic display
        # Convert the mosaic image from OpenCV to PIL
        mosaic_rgb = cv2.cvtColor(current_mosaic, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(mosaic_rgb)
        img_pil = img_pil.resize((400, 300))  # Resize for display
        ctk_image = ctk.CTkImage(img_pil, size=(400, 300))
        
        self.current_mosaic_display.configure(image=ctk_image, text="")
        self.current_mosaic_display._image = ctk_image  # Keep a reference to avoid garbage collection

    def show_detection(self, filename):
        img = Image.open(os.path.join("Detections", filename))
        img = img.resize((300, 200))
        det_image = ctk.CTkImage(img, size=(300, 200))
        # Show in a new window or update a label
        det_window = ctk.CTkToplevel(self)
        det_window.title(filename)
        label = ctk.CTkLabel(det_window, image=det_image, text="")
        label.pack()

if __name__ == "__main__":
    app = App()
    app.mainloop()