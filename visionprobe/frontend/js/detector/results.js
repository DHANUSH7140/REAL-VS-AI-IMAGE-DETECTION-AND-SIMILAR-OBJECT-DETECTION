window.currentDetectorData = null;

window.runDetection = async function() {
    if (!window.detectorFile) return;
    
    const btn = document.getElementById('detect-btn');
    const panel = document.getElementById('detect-progress');
    const resultsPanel = document.getElementById('detect-results-panel');
    const detailPanel = document.getElementById('detect-detail');
    
    btn.disabled = true;
    resultsPanel.classList.add('hidden');
    detailPanel.classList.add('hidden');
    panel.innerHTML = '';
    panel.classList.remove('hidden');
    
    // Setup fake progress steps for UX
    const steps = [
        "Preprocessing image & extracting EXIF metadata",
        "Running OpenCLIP ViT-L/14 semantic analysis",
        "Running EfficientNetV2-L artifact detection",
        "Computing FFT frequency spectrum analysis",
        "Computing SRM noise residual statistics",
        "Running XGBoost classifier with calibration",
        "Generating GradCAM++ and text explanations"
    ];
    
    let currentStep = 0;
    const stepInterval = setInterval(() => {
        if (currentStep < steps.length) {
            const stepEl = document.createElement('div');
            stepEl.className = 'progress-step active';
            stepEl.innerHTML = `<div class="step-dot active" id="dot-${currentStep}"></div><span>${steps[currentStep]}...</span>`;
            panel.appendChild(stepEl);
            
            if (currentStep > 0) {
                const prev = document.getElementById(`dot-${currentStep - 1}`);
                if (prev) {
                    prev.classList.remove('active');
                    prev.classList.add('done');
                    prev.parentElement.style.color = 'var(--success)';
                }
            }
            currentStep++;
        }
    }, 400);

    try {
        const includeGradcam = document.getElementById('include-gradcam').checked;
        const includeShap = document.getElementById('include-shap').checked;
        
        const data = await apiAnalyzeImage(window.detectorFile, { includeGradcam, includeShap });
        window.currentDetectorData = data;
        
        clearInterval(stepInterval);
        panel.classList.add('hidden');
        
        renderDetectorResults(data);
        
    } catch (err) {
        clearInterval(stepInterval);
        panel.innerHTML = `<div class="progress-step" style="color:var(--danger)">Error: ${err.message}</div>`;
    } finally {
        btn.disabled = false;
    }
};

function renderDetectorResults(data) {
    const resultsPanel = document.getElementById('detect-results-panel');
    const detailPanel = document.getElementById('detect-detail');
    resultsPanel.classList.remove('hidden');
    detailPanel.classList.remove('hidden');
    
    // Verdict
    const isAI = data.verdict === 'AI_GENERATED';
    const banner = document.getElementById('detect-verdict-banner');
    banner.className = `verdict-banner ${isAI ? 'ai' : 'real'}`;
    
    const icon = isAI ? '🤖' : '📸';
    const label = isAI ? 'AI-Generated Image' : 'Authentic Photograph';
    const confText = `Ensemble confidence: ${(data.confidence * 100).toFixed(1)}% ±${(data.confidence_interval * 100).toFixed(1)}%`;
    
    banner.innerHTML = `
        <div class="verdict-icon">${icon}</div>
        <div>
            <div class="verdict-label">${label}</div>
            <div class="verdict-conf">${confText} · ${data.metadata.processing_time_ms}ms processing time</div>
        </div>
    `;
    
    // Score Grid
    const grid = document.getElementById('detect-score-grid');
    grid.innerHTML = '';
    const scores = data.scores;
    
    const models = [
        { key: 'ensemble', name: 'XGBOOST ENSEMBLE' },
        { key: 'clip', name: 'OPENCLIP ViT-L/14' },
        { key: 'efficientnet', name: 'EFFICIENTNETV2-L' },
        { key: 'frequency', name: 'FFT FREQUENCY' },
        { key: 'srm', name: 'SRM NOISE' },
        { key: 'exif', name: 'EXIF METADATA' },
    ];
    
    models.forEach((m, idx) => {
        const val = scores[m.key];
        const pct = (val * 100).toFixed(1);
        const color = val > 0.5 ? 'var(--danger)' : 'var(--success)';
        
        const card = document.createElement('div');
        card.className = 'score-card';
        card.innerHTML = `
            <div class="score-card-name">${m.name}</div>
            <div class="score-card-value" style="color: ${color}">${pct}% AI</div>
            <div class="score-bar">
                <div class="score-bar-fill" id="bar-${m.key}" style="width: 0%; background: ${color}"></div>
            </div>
        `;
        grid.appendChild(card);
        
        // Animate bar
        setTimeout(() => {
            const bar = document.getElementById(`bar-${m.key}`);
            if (bar) bar.style.width = `${val * 100}%`;
        }, 100 + (idx * 50));
    });
    
    // Reasoning List
    const reasoningList = document.getElementById('detect-reasoning');
    reasoningList.innerHTML = '';
    
    data.explanation.reasoning.forEach(reason => {
        let icon = '💡';
        if (reason.toLowerCase().includes('exif') || reason.toLowerCase().includes('camera')) icon = '📸';
        if (reason.toLowerCase().includes('frequency') || reason.toLowerCase().includes('dct')) icon = '🌊';
        if (reason.toLowerCase().includes('srm') || reason.toLowerCase().includes('noise')) icon = '🔍';
        if (reason.toLowerCase().includes('neural') || reason.toLowerCase().includes('clip')) icon = '🧠';
        
        const li = document.createElement('li');
        li.className = 'reasoning-item';
        li.innerHTML = `<span class="reasoning-icon">${icon}</span><span>${reason}</span>`;
        reasoningList.appendChild(li);
    });
    
    // Init charts with first tab
    if (window.renderDetailHeatmap) window.switchDetectTab('heatmap');
}

window.switchDetectTab = function(tabName) {
    document.querySelectorAll('.detail-tab').forEach(t => t.classList.remove('active'));
    const btn = document.querySelector(`.detail-tab[data-tab="${tabName}"]`);
    if (btn) btn.classList.add('active');
    
    const content = document.getElementById('detect-detail-content');
    content.innerHTML = ''; // clear
    
    const data = window.currentDetectorData;
    if (!data) return;
    
    if (tabName === 'heatmap' && window.renderDetailHeatmap) window.renderDetailHeatmap(data);
    else if (tabName === 'frequency' && window.renderDetailFrequency) window.renderDetailFrequency(data);
    else if (tabName === 'shap' && window.renderDetailShap) window.renderDetailShap(data);
    else if (tabName === 'models' && window.renderDetailModels) window.renderDetailModels(data);
    else if (tabName === 'raw' && window.renderDetailRaw) window.renderDetailRaw(data);
};
