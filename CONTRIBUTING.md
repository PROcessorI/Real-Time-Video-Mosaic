# Руководство для разработчиков / Contributing Guide

## Русский

### Как внести вклад в проект

Мы рады любым улучшениям проекта! Вот как вы можете помочь:

#### 1. Сообщить об ошибке (Bug Report)
- Откройте [Issue](https://github.com/PROcessorI/Real-Time-Video-Mosaic/issues)
- Опишите проблему детально
- Укажите версию Python и ОС
- Приложите скриншоты или логи, если возможно

#### 2. Предложить улучшение
- Откройте [Issue](https://github.com/PROcessorI/Real-Time-Video-Mosaic/issues)
- Опишите ваше предложение
- Объясните, как это улучшит проект

#### 3. Отправить Pull Request

**Шаги:**
1. Сделайте Fork репозитория
2. Создайте новую ветку:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Внесите изменения
4. Убедитесь, что код работает:
   ```bash
   python main.py  # Проверьте основной функционал
   python gui.py   # Проверьте GUI
   ```
5. Закоммитьте изменения:
   ```bash
   git add .
   git commit -m "Описание изменений"
   ```
6. Отправьте в свой Fork:
   ```bash
   git push origin feature/my-new-feature
   ```
7. Откройте Pull Request на GitHub

### Стандарты кода
- Следуйте PEP 8 для Python кода
- Добавляйте комментарии для сложной логики
- Документируйте новые функции в README

### Области для улучшения
- ⚡ Оптимизация производительности
- 🎨 Улучшение GUI
- 📊 Новые алгоритмы обнаружения
- 🗺️ Улучшение навигационной карты
- 📝 Документация и примеры
- 🐛 Исправление ошибок
- 🧪 Добавление тестов

---

## English

### How to Contribute

We welcome any improvements to the project! Here's how you can help:

#### 1. Report a Bug
- Open an [Issue](https://github.com/PROcessorI/Real-Time-Video-Mosaic/issues)
- Describe the problem in detail
- Specify Python version and OS
- Attach screenshots or logs if possible

#### 2. Suggest an Enhancement
- Open an [Issue](https://github.com/PROcessorI/Real-Time-Video-Mosaic/issues)
- Describe your suggestion
- Explain how it would improve the project

#### 3. Submit a Pull Request

**Steps:**
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Make your changes
4. Ensure the code works:
   ```bash
   python main.py  # Test main functionality
   python gui.py   # Test GUI
   ```
5. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```
6. Push to your Fork:
   ```bash
   git push origin feature/my-new-feature
   ```
7. Open a Pull Request on GitHub

### Code Standards
- Follow PEP 8 for Python code
- Add comments for complex logic
- Document new features in README

### Areas for Improvement
- ⚡ Performance optimization
- 🎨 GUI improvements
- 📊 New detection algorithms
- 🗺️ Navigation map enhancements
- 📝 Documentation and examples
- 🐛 Bug fixes
- 🧪 Adding tests

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Real-Time-Video-Mosaic.git
cd Real-Time-Video-Mosaic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests (if any)
# python -m pytest

# Start development
python main.py  # or python gui.py
```

### Questions?
Feel free to open an Issue if you have any questions about contributing!
