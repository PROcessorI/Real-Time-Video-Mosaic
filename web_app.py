from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import threading
import subprocess
import json
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables for progress tracking
progress_status = {"status": "idle", "progress": 0, "message": ""}

def run_main(video_path, output_dir):
    global progress_status
    try:
        progress_status = {"status": "running", "progress": 0, "message": "Запуск обработки..." }
        # Run main.py as subprocess, capturing output
        cmd = ['python', 'main.py']
        if video_path:
            cmd.append(video_path)
        cmd.extend(['--output-dir', output_dir or '.', '--hide'])
        print(f"Running command: {' '.join(cmd)}")  # Debug
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=os.getcwd())
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"Subprocess output: {output.strip()}")  # Debug
                # Parse progress from output (assuming main.py prints progress)
                if "Processed frame" in output:
                    # Extract progress percentage
                    try:
                        parts = output.split()
                        frame_info = parts[2]  # "50/100"
                        current, total = map(int, frame_info.split('/'))
                        progress = int((current / total) * 100)
                        progress_status = {"status": "running", "progress": progress, "message": output.strip()}
                    except:
                        progress_status = {"status": "running", "progress": progress_status["progress"], "message": output.strip()}
                else:
                    progress_status = {"status": "running", "progress": progress_status["progress"], "message": output.strip()}
        
        if process.returncode == 0:
            progress_status = {"status": "completed", "progress": 100, "message": "Обработка завершена успешно!"}
        else:
            progress_status = {"status": "error", "progress": 0, "message": f"Обработка не удалась, код возврата: {process.returncode}"}
    except Exception as e:
        progress_status = {"status": "error", "progress": 0, "message": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({"error": "Видео файл не предоставлен"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({"message": "Файл загружен успешно", "filename": filename})

@app.route('/start', methods=['POST'])
def start_processing():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "Имя файла не предоставлено"}), 400
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_dir = app.config['RESULTS_FOLDER']
    
    if progress_status["status"] == "running":
        return jsonify({"error": "Обработка уже запущена"}), 400
    
    thread = threading.Thread(target=run_main, args=(video_path, output_dir))
    thread.start()
    
    return jsonify({"message": "Обработка запущена"})

@app.route('/progress')
def get_progress():
    return jsonify(progress_status)

@app.route('/results')
def get_results():
    results = {}
    results_dir = Path(app.config['RESULTS_FOLDER'])
    if results_dir.exists():
        for file in results_dir.rglob('*'):
            if file.is_file() and file.suffix in ['.jpg', '.png']:
                relative_path = file.relative_to(results_dir)
                results[str(relative_path)] = f'/results/{relative_path}'
    print(f"Found results: {results}")  # Debug
    return jsonify(results)

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)