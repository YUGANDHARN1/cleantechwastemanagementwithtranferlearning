// JavaScript for Waste Management Classification System

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('imageInput');
    const uploadZone = document.querySelector('.upload-zone');
    const previewContainer = document.getElementById('imagePreview');
    const resultsContainer = document.getElementById('results');
    const loadingSpinner = document.getElementById('loading');

    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // Form submission handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }

    // Drag and drop handlers
    if (uploadZone) {
        setupDragAndDrop(uploadZone, fileInput);
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        if (validateFile(file)) {
            previewImage(file);
        } else {
            showError('Please select a valid image file (JPG, JPEG, PNG) under 10MB.');
        }
    }
}

function validateFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!validTypes.includes(file.type)) {
        return false;
    }

    if (file.size > maxSize) {
        return false;
    }

    return true;
}

function previewImage(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const previewContainer = document.getElementById('imagePreview');
        const previewImg = document.getElementById('preview');
        
        if (previewImg && previewContainer) {
            previewImg.src = e.target.result;
            previewContainer.style.display = 'block';
            
            // Hide previous results
            const resultsContainer = document.getElementById('results');
            if (resultsContainer) {
                resultsContainer.style.display = 'none';
            }
        }
    };
    
    reader.readAsDataURL(file);
}

function setupDragAndDrop(uploadZone, fileInput) {
    uploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                fileInput.files = files;
                previewImage(file);
            } else {
                showError('Please drop a valid image file (JPG, JPEG, PNG) under 10MB.');
            }
        }
    });

    uploadZone.addEventListener('click', function() {
        fileInput.click();
    });
}

function handleFormSubmit(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select an image first!');
        return;
    }

    if (!validateFile(file)) {
        showError('Please select a valid image file (JPG, JPEG, PNG) under 10MB.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    showLoading();
    hideResults();

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showError('Error: ' + data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        hideLoading();
        showError('Network error: ' + error.message);
    });
}

function showLoading() {
    const loadingSpinner = document.getElementById('loading');
    if (loadingSpinner) {
        loadingSpinner.style.display = 'block';
    }
}

function hideLoading() {
    const loadingSpinner = document.getElementById('loading');
    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
    }
}

function hideResults() {
    const resultsContainer = document.getElementById('results');
    if (resultsContainer) {
        resultsContainer.style.display = 'none';
    }
}

function displayResults(data) {
    const resultsContainer = document.getElementById('results');
    const topPredictionContainer = document.getElementById('topPrediction');
    const allPredictionsContainer = document.getElementById('allPredictions');

    if (!resultsContainer || !topPredictionContainer || !allPredictionsContainer) {
        return;
    }

    const predictions = data.predictions;
    const topPrediction = predictions[0];

    // Category colors
    const categoryColors = {
        'Recyclable': '#28a745',
        'Organic': '#8bc34a',
        'Hazardous': '#dc3545',
        'Non-Recyclable': '#fd7e14'
    };

    // Display top prediction
    const topPredHTML = `
        <div class="text-center">
            <h3 style="color: ${categoryColors[topPrediction.category]}" class="mb-3">
                ${getCategoryIcon(topPrediction.category)} ${topPrediction.category}
            </h3>
            <div class="confidence-bar mb-3">
                <div class="confidence-fill" 
                     style="width: ${topPrediction.confidence}%; background-color: ${categoryColors[topPrediction.category]}">
                </div>
            </div>
            <h4>Confidence: ${topPrediction.confidence.toFixed(1)}%</h4>
            <p class="text-muted">${getCategoryDescription(topPrediction.category)}</p>
        </div>
    `;

    topPredictionContainer.innerHTML = topPredHTML;

    // Display all predictions
    let allPredHTML = '';
    predictions.forEach((pred, index) => {
        const isTop = index === 0;
        allPredHTML += `
            <div class="mb-3 p-3 border rounded ${isTop ? 'border-primary' : ''}">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="fw-bold">${getCategoryIcon(pred.category)} ${pred.category}</span>
                    <span class="badge bg-primary">${pred.confidence.toFixed(1)}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" 
                         style="width: ${pred.confidence}%; background-color: ${categoryColors[pred.category]}">
                    </div>
                </div>
            </div>
        `;
    });

    allPredictionsContainer.innerHTML = allPredHTML;

    // Animate confidence bars
    setTimeout(() => {
        const bars = document.querySelectorAll('.confidence-fill');
        bars.forEach(bar => {
            bar.style.width = bar.style.width;
        });
    }, 100);

    resultsContainer.style.display = 'block';
}

function getCategoryIcon(category) {
    const icons = {
        'Recyclable': '<i class="fas fa-recycle"></i>',
        'Organic': '<i class="fas fa-leaf"></i>',
        'Hazardous': '<i class="fas fa-exclamation-triangle"></i>',
        'Non-Recyclable': '<i class="fas fa-trash"></i>'
    };
    return icons[category] || '<i class="fas fa-question"></i>';
}

function getCategoryDescription(category) {
    const descriptions = {
        'Recyclable': 'Materials that can be processed and reused',
        'Organic': 'Biodegradable waste from living organisms',
        'Hazardous': 'Dangerous materials requiring special handling',
        'Non-Recyclable': 'Materials that cannot be easily recycled'
    };
    return descriptions[category] || 'Unknown category';
}

function showError(message) {
    // Create or update error alert
    let errorAlert = document.getElementById('errorAlert');
    
    if (!errorAlert) {
        errorAlert = document.createElement('div');
        errorAlert.id = 'errorAlert';
        errorAlert.className = 'alert alert-danger alert-dismissible fade show mt-3';
        
        const container = document.querySelector('.container .card-body');
        if (container) {
            container.insertBefore(errorAlert, container.firstChild);
        }
    }

    errorAlert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (errorAlert) {
            errorAlert.remove();
        }
    }, 5000);
}

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}