# Краткая инструкция по установке / Quick Installation Guide

## Русский

### Быстрая установка (3 шага)

1. **Клонировать репозиторий**
   ```bash
   git clone https://github.com/PROcessorI/Real-Time-Video-Mosaic.git
   cd Real-Time-Video-Mosaic
   ```

2. **Создать виртуальное окружение и установить зависимости**
   
   **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
   **Linux/macOS:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   sudo apt-get install python3-tk  # Только для Linux
   pip install -r requirements.txt
   ```

3. **Запустить приложение**
   
   **GUI (Графический интерфейс):**
   ```bash
   python gui.py
   ```
   
   **CLI (Командная строка):**
   ```bash
   python main.py
   ```

### Требования
- Python 3.8+
- 8 GB RAM (рекомендуется 16 GB)
- 2 GB свободного места на диске

### Полная документация
- [README.md](README.md) - Полная документация на русском
- [README_EN.md](README_EN.md) - Full documentation in English

---

## English

### Quick Installation (3 steps)

1. **Clone the repository**
   ```bash
   git clone https://github.com/PROcessorI/Real-Time-Video-Mosaic.git
   cd Real-Time-Video-Mosaic
   ```

2. **Create virtual environment and install dependencies**
   
   **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
   **Linux/macOS:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   sudo apt-get install python3-tk  # Linux only
   pip install -r requirements.txt
   ```

3. **Run the application**
   
   **GUI (Graphical Interface):**
   ```bash
   python gui.py
   ```
   
   **CLI (Command Line):**
   ```bash
   python main.py
   ```

### Requirements
- Python 3.8+
- 8 GB RAM (16 GB recommended)
- 2 GB free disk space

### Full Documentation
- [README.md](README.md) - Полная документация на русском
- [README_EN.md](README_EN.md) - Full documentation in English
