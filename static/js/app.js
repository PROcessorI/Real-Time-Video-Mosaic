document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadStatus = document.getElementById('upload-status');
    const startBtn = document.getElementById('start-btn');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    const resultsContainer = document.getElementById('results-container');
    const themeToggle = document.getElementById('theme-toggle');

    let uploadedFilename = null;

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.classList.add(savedTheme + '-theme');
    updateThemeButton(savedTheme);

    function updateThemeButton(theme) {
        const icon = themeToggle.querySelector('i');
        if (theme === 'light') {
            icon.className = 'fas fa-moon';
            themeToggle.innerHTML = '<i class="fas fa-moon"></i> Тёмная Тема';
        } else {
            icon.className = 'fas fa-sun';
            themeToggle.innerHTML = '<i class="fas fa-sun"></i> Светлая Тема';
        }
    }

    // Theme toggle
    themeToggle.addEventListener('click', function() {
        const body = document.body;
        let newTheme;
        if (body.classList.contains('light-theme')) {
            body.classList.remove('light-theme');
            body.classList.add('dark-theme');
            newTheme = 'dark';
        } else {
            body.classList.remove('dark-theme');
            body.classList.add('light-theme');
            newTheme = 'light';
        }
        localStorage.setItem('theme', newTheme);
        updateThemeButton(newTheme);
    });

    // Upload form
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData();
        const fileInput = document.getElementById('video-file');
        formData.append('video', fileInput.files[0]);

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Загрузка...';

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                uploadStatus.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            } else {
                uploadStatus.innerHTML = `<div class="alert alert-success">Файл загружен успешно</div>`;
                uploadedFilename = data.filename;
                startBtn.disabled = false;
            }
        })
        .catch(error => {
            uploadStatus.innerHTML = `<div class="alert alert-danger">Upload failed: ${error.message}</div>`;
        })
        .finally(() => {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Загрузить';
        });
    });

    // Start processing
    startBtn.addEventListener('click', function() {
        if (!uploadedFilename) return;

        startBtn.disabled = true;
        startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Запуск...';

        fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename: uploadedFilename })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                statusMessage.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                startBtn.disabled = false;
                startBtn.innerHTML = '<i class="fas fa-play"></i> Начать Обработку';
            } else {
                statusMessage.innerHTML = `<div class="alert alert-info">${data.message}</div>`;
                progressContainer.style.display = 'block';
                pollProgress();
            }
        })
        .catch(error => {
            statusMessage.innerHTML = `<div class="alert alert-danger">Не удалось запустить: ${error.message}</div>`;
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-play"></i> Начать Обработку';
        });
    });

    // Poll progress
    function pollProgress() {
        fetch('/progress')
        .then(response => response.json())
        .then(data => {
            progressBar.style.width = `${data.progress}%`;
            progressBar.setAttribute('aria-valuenow', data.progress);
            statusMessage.innerHTML = `<div class="alert alert-info">${data.message}</div>`;

            if (data.status === 'completed' || data.status === 'error') {
                startBtn.disabled = false;
                startBtn.innerHTML = '<i class="fas fa-play"></i> Начать Обработку';
                if (data.status === 'completed') {
                    loadResults();
                }
            } else {
                setTimeout(pollProgress, 1000);
            }
        })
        .catch(error => {
            console.error('Progress poll failed:', error);
            setTimeout(pollProgress, 1000);
        });
    }

    // Load results
    function loadResults() {
        fetch('/results')
        .then(response => response.json())
        .then(data => {
            console.log('Results data:', data);  // Debug
            if (Object.keys(data).length === 0) {
                resultsContainer.innerHTML = '<p class="text-muted">Результаты недоступны.</p>';
                return;
            }

            let html = '<div class="row">';
            for (const [name, url] of Object.entries(data)) {
                const displayName = name.replace(/_/g, ' ').replace(/\//g, ' / ').toUpperCase();
                html += `
                    <div class="col-md-6 mb-3">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title">${displayName}</h6>
                                <img src="${url}" class="result-image img-fluid" alt="${name}">
                                <div class="mt-2">
                                    <a href="${url}" download class="btn btn-sm btn-primary">Скачать</a>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
            html += '</div>';
            resultsContainer.innerHTML = html;
        })
        .catch(error => {
            resultsContainer.innerHTML = `<div class="alert alert-danger">Не удалось загрузить результаты: ${error.message}</div>`;
        });
    }  // ИСПРАВЛЕНО: убрана лишняя закрывающая скобка и точка с запятой

    // Image zoom modal (перенесён внутрь DOMContentLoaded)
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('result-image')) {
            console.log('Image clicked:', e.target.src);
            const modalImage = document.getElementById('modalImage');
            modalImage.src = e.target.src;
            modalImage.alt = e.target.alt;
            // Show modal manually if needed
            const modal = new bootstrap.Modal(document.getElementById('imageModal'));
            modal.show();
        }
    });
});
