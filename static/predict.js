/**
 * Predict Page - Handle predictions and results display
 */

// State
let currentSchema = null;
let predictionHistory = [];

// DOM Elements
const schemaLoadingEl = document.getElementById('schema-loading');
const predictFormEl = document.getElementById('predict-form');
const featureFieldsEl = document.getElementById('feature-fields');
const predictButtonEl = document.getElementById('predict-button');
const predictErrorEl = document.getElementById('predict-error');

const resultsEmptyEl = document.getElementById('results-empty');
const resultsPanelEl = document.getElementById('results-panel');

const historyEmptyEl = document.getElementById('history-empty');
const historyPanelEl = document.getElementById('history-panel');
const historyListEl = document.getElementById('history-list');
const clearHistoryBtn = document.getElementById('clear-history');

const modelStatusEl = document.getElementById('model-status');
const modelVersionEl = document.getElementById('model-version');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadSchema();
    await checkModelHealth();
    loadHistoryFromStorage();
    
    predictFormEl.addEventListener('submit', handlePrediction);
    document.getElementById('copy-result').addEventListener('click', copyResultAsJson);
    clearHistoryBtn.addEventListener('click', clearHistory);
});

/**
 * Load schema from backend
 */
async function loadSchema() {
    try {
        const response = await fetch('/api/predict-schema');
        if (!response.ok) {
            throw new Error(`Failed to load schema: ${response.statusText}`);
        }
        
        currentSchema = await response.json();
        buildForm();
        schemaLoadingEl.classList.add('hidden');
        predictFormEl.classList.remove('hidden');
    } catch (error) {
        console.error('Schema load error:', error);
        schemaLoadingEl.innerHTML = `<p>❌ Error loading schema: ${error.message}</p>`;
    }
}

/**
 * Build form fields from schema
 */
function buildForm() {
    if (!currentSchema || !currentSchema.columns) {
        featureFieldsEl.innerHTML = '<p>No schema available</p>';
        return;
    }

    const columns = currentSchema.columns;
    const targetColumn = currentSchema.target_column;
    
    featureFieldsEl.innerHTML = '';
    
    Object.entries(columns).forEach(([name, dtype]) => {
        if (name === targetColumn) return; // Skip target column
        
        const label = document.createElement('label');
        label.className = 'field';
        
        const labelText = document.createElement('span');
        labelText.textContent = `${name} (${dtype})`;
        label.appendChild(labelText);
        
        let input;
        if (dtype === 'float64' || dtype === 'int64') {
            input = document.createElement('input');
            input.type = 'number';
            input.name = name;
            input.placeholder = `Enter ${dtype === 'float64' ? 'decimal' : 'integer'} value`;
            input.step = dtype === 'float64' ? '0.01' : '1';
            input.required = true;
        } else if (dtype === 'str' || dtype === 'object') {
            input = document.createElement('input');
            input.type = 'text';
            input.name = name;
            input.placeholder = `Enter text value`;
            input.required = true;
        } else {
            input = document.createElement('input');
            input.type = 'text';
            input.name = name;
            input.required = true;
        }
        
        label.appendChild(input);
        featureFieldsEl.appendChild(label);
    });
}

/**
 * Check model health
 */
async function checkModelHealth() {
    try {
        const response = await fetch('/api/health/predict');
        if (!response.ok) {
            throw new Error('Health check failed');
        }
        
        const health = await response.json();
        
        if (health.model_loaded) {
            modelStatusEl.textContent = '✅ Ready';
            modelStatusEl.parentElement.parentElement.style.color = 'var(--success)';
            modelVersionEl.textContent = health.model_version || '-';
        } else {
            modelStatusEl.textContent = '⚠️ No Model';
            modelStatusEl.parentElement.parentElement.style.color = 'var(--alert)';
            modelVersionEl.textContent = '-';
        }
    } catch (error) {
        console.error('Health check error:', error);
        modelStatusEl.textContent = '❌ Error';
        modelStatusEl.parentElement.parentElement.style.color = 'var(--alert)';
    }
}

/**
 * Handle form submission
 */
async function handlePrediction(e) {
    e.preventDefault();
    
    // Collect form data
    const formData = new FormData(predictFormEl);
    const features = {};
    
    for (const [name, value] of formData.entries()) {
        // Convert to appropriate type
        const columnType = currentSchema.columns[name];
        if (columnType === 'float64') {
            features[name] = parseFloat(value);
        } else if (columnType === 'int64') {
            features[name] = parseInt(value);
        } else {
            features[name] = value;
        }
    }
    
    // Clear previous errors
    predictErrorEl.classList.add('hidden');
    
    // Make prediction
    predictButtonEl.disabled = true;
    predictButtonEl.textContent = 'Making prediction...';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || response.statusText);
        }
        
        const result = await response.json();
        displayResults(result, features);
        addToHistory(result, features);
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message);
    } finally {
        predictButtonEl.disabled = false;
        predictButtonEl.textContent = 'Get Prediction';
    }
}

/**
 * Display prediction results
 */
function displayResults(result, features) {
    // Show results panel
    resultsEmptyEl.classList.add('hidden');
    resultsPanelEl.classList.remove('hidden');
    
    // Prediction value
    const predictionValue = result.prediction;
    document.getElementById('result-prediction').textContent = 
        typeof predictionValue === 'number' ? predictionValue.toFixed(2) : predictionValue;
    
    // Confidence score
    const confidence = result.confidence_score * 100;
    document.getElementById('result-confidence-text').textContent = confidence.toFixed(1) + '%';
    document.getElementById('result-confidence-fill').style.width = confidence + '%';
    
    // Set confidence color based on value
    const confidenceBar = document.getElementById('result-confidence-bar');
    if (confidence >= 80) {
        confidenceBar.style.backgroundColor = 'var(--success-soft)';
        document.getElementById('result-confidence-fill').style.backgroundColor = 'var(--success)';
    } else if (confidence >= 60) {
        confidenceBar.style.backgroundColor = '#fef3c7';
        document.getElementById('result-confidence-fill').style.backgroundColor = '#f59e0b';
    } else {
        confidenceBar.style.backgroundColor = 'var(--alert-soft)';
        document.getElementById('result-confidence-fill').style.backgroundColor = 'var(--alert)';
    }
    
    // Model info
    document.getElementById('result-model-name').textContent = result.model_name;
    document.getElementById('result-model-version').textContent = result.model_version;
    document.getElementById('result-task-type').textContent = result.task_type;
    
    // Timestamp
    const timestamp = new Date(result.timestamp).toLocaleString();
    document.getElementById('result-timestamp').textContent = timestamp;
    
    // Store current result for copying
    window.currentPredictionResult = result;
}

/**
 * Copy result as JSON
 */
function copyResultAsJson() {
    if (!window.currentPredictionResult) return;
    
    const json = JSON.stringify(window.currentPredictionResult, null, 2);
    navigator.clipboard.writeText(json).then(() => {
        const btn = document.getElementById('copy-result');
        const originalText = btn.textContent;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    });
}

/**
 * Show error message
 */
function showError(message) {
    predictErrorEl.textContent = message;
    predictErrorEl.classList.remove('hidden');
}

/**
 * Add prediction to history
 */
function addToHistory(result, features) {
    const historyItem = {
        timestamp: new Date(),
        prediction: result.prediction,
        confidence: result.confidence_score,
        modelVersion: result.model_version,
        features: features,
    };
    
    predictionHistory.unshift(historyItem);
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }
    
    saveHistoryToStorage();
    displayHistory();
}

/**
 * Display prediction history
 */
function displayHistory() {
    if (predictionHistory.length === 0) {
        historyEmptyEl.classList.remove('hidden');
        historyPanelEl.classList.add('hidden');
        return;
    }
    
    historyEmptyEl.classList.add('hidden');
    historyPanelEl.classList.remove('hidden');
    
    historyListEl.innerHTML = predictionHistory.map((item, idx) => {
        const timestamp = new Date(item.timestamp).toLocaleTimeString();
        const confidence = (item.confidence * 100).toFixed(1);
        
        return `
            <div class="history-item">
                <div class="history-time">${timestamp}</div>
                <div class="history-main">
                    <div class="history-prediction">
                        <span class="history-label">Prediction:</span>
                        <span class="history-value">${typeof item.prediction === 'number' ? item.prediction.toFixed(2) : item.prediction}</span>
                    </div>
                    <div class="history-confidence">
                        <span class="history-label">Confidence:</span>
                        <span class="history-value">${confidence}%</span>
                    </div>
                </div>
                <div class="history-version">v${item.modelVersion}</div>
                <button class="history-details-btn" onclick="toggleHistoryDetails(${idx})">Details</button>
            </div>
            <div class="history-details hidden" id="history-details-${idx}">
                <pre>${JSON.stringify(item.features, null, 2)}</pre>
            </div>
        `;
    }).join('');
}

/**
 * Toggle history item details
 */
window.toggleHistoryDetails = function(idx) {
    const details = document.getElementById(`history-details-${idx}`);
    details.classList.toggle('hidden');
};

/**
 * Clear history
 */
function clearHistory() {
    if (confirm('Clear all prediction history?')) {
        predictionHistory = [];
        saveHistoryToStorage();
        displayHistory();
    }
}

/**
 * Save history to localStorage
 */
function saveHistoryToStorage() {
    try {
        const serializable = predictionHistory.map(item => ({
            ...item,
            timestamp: item.timestamp.toISOString(),
        }));
        localStorage.setItem('predictionHistory', JSON.stringify(serializable));
    } catch (error) {
        console.warn('Could not save history to localStorage:', error);
    }
}

/**
 * Load history from localStorage
 */
function loadHistoryFromStorage() {
    try {
        const stored = localStorage.getItem('predictionHistory');
        if (stored) {
            predictionHistory = JSON.parse(stored).map(item => ({
                ...item,
                timestamp: new Date(item.timestamp),
            }));
            displayHistory();
        }
    } catch (error) {
        console.warn('Could not load history from localStorage:', error);
    }
}
