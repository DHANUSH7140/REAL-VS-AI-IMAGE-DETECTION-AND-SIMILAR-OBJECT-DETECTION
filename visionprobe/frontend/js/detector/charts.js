window.renderDetailHeatmap = function(data) {
    const container = document.getElementById('detect-detail-content');
    
    const hmBase64 = data.explanation.gradcam_heatmap;
    if (!hmBase64) {
        container.innerHTML = `<div class="card" style="text-align:center; padding: 40px; color:var(--text-muted)">GradCAM heatmap was not enabled or failed to generate.</div>`;
        return;
    }
    
    const wrapper = document.createElement('div');
    wrapper.innerHTML = `
        <div style="display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start;">
            <div style="flex:1; min-width:300px">
                <h4 style="margin-bottom:10px; color:var(--text-muted)">Original Image</h4>
                <img src="${document.getElementById('detect-preview-img').src}" style="width:100%; border-radius:var(--radius); border:1px solid var(--border)">
            </div>
            <div style="flex:1; min-width:300px">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <h4 style="color:var(--text-muted)">GradCAM++ Heatmap</h4>
                    <span style="font-size:12px; color:var(--danger)">Red/warm regions = highly suspicious</span>
                </div>
                <img src="data:image/jpeg;base64,${hmBase64}" style="width:100%; border-radius:var(--radius); border:1px solid var(--border)">
            </div>
        </div>
        <div style="margin-top:16px; padding:12px; background:var(--bg); border:1px solid var(--border); border-radius:var(--radius); font-size:13px; color:var(--text-muted)">
            <strong>What is this?</strong> GradCAM calculates the gradient of the EfficientNetV2 AI prediction score with respect to the final convolutional feature map. It highlights the specific textures and patterns the model used to classify the image as AI-generated.
        </div>
    `;
    container.appendChild(wrapper);
};

window.renderDetailFrequency = function(data) {
    const container = document.getElementById('detect-detail-content');
    const profile = data.explanation.frequency_profile;
    const spectrum2d = data.explanation.frequency_spectrum;
    
    if (!profile || profile.length === 0) {
        container.innerHTML = `<div class="card" style="text-align:center; color:var(--text-muted)">Frequency data is not available.</div>`;
        return;
    }

    container.innerHTML = `
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
            <div style="flex:1; min-width:300px; background:var(--bg); border:1px solid var(--border); border-radius:var(--radius); padding:16px;">
                <h4 style="margin-bottom:16px; text-align:center; color:var(--text-muted)">1D Azimuthal Frequency Profile</h4>
                <canvas id="freq-chart-canvas" width="100%" height="250"></canvas>
            </div>
            <div style="flex:1; min-width:300px; background:var(--bg); border:1px solid var(--border); border-radius:var(--radius); padding:16px;">
                <h4 style="margin-bottom:0px; text-align:center; color:var(--text-muted)">2D Log Power Spectrum</h4>
                <div id="plotly-heatmap" style="width:100%; height:250px;"></div>
            </div>
        </div>
    `;

    // 1D Chart.js
    const ctx = document.getElementById('freq-chart-canvas').getContext('2d');
    const labels = Array.from({length: profile.length}, (_, i) => i * 2);
    
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#8b949e' : '#636c76';
    const gridColor = isDark ? '#30363d' : '#e4e8ec';

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Log Magnitude',
                data: profile,
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: true,
                tension: 0.2,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                annotation: { /* Plugin not naturally loaded, skip or simulate via native bg */ }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Frequency (cycles/px)', color: textColor },
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                },
                y: {
                    title: { display: true, text: 'Log Magnitude', color: textColor },
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                }
            }
        }
    });

    // 2D Plotly Heatmap
    if (spectrum2d && spectrum2d.length > 0 && typeof Plotly !== 'undefined') {
        const pData = [{
            z: spectrum2d,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: false
        }];
        const layout = {
            margin: { t: 10, r: 10, b: 20, l: 30 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { showticklabels: false },
            yaxis: { showticklabels: false, autorange: 'reversed' }
        };
        Plotly.newPlot('plotly-heatmap', pData, layout, {displayModeBar: false});
    }
};

window.renderDetailShap = function(data) {
    const container = document.getElementById('detect-detail-content');
    const shap = data.explanation.shap_features;
    
    if (!shap || shap.length === 0) {
        container.innerHTML = `<div class="card" style="text-align:center; color:var(--text-muted)">SHAP values were not generated.</div>`;
        return;
    }
    
    let html = `<div style="max-width:800px; margin:0 auto;"><div class="shap-chart">`;
    // Find absolute max for scaling
    const maxVal = Math.max(...shap.map(s => s.value));
    
    shap.forEach((item, idx) => {
        const pct = (item.value / maxVal) * 100;
        const colorClass = item.direction === 'AI' ? 'ai' : 'real';
        
        html += `
            <div class="shap-row">
                <div class="shap-name">${item.name}</div>
                <div class="shap-bar-wrap">
                    <div class="shap-bar-fill ${colorClass}" id="shap-bar-${idx}" style="width: 0%"></div>
                </div>
                <div class="shap-value" style="color:var(--${colorClass === 'ai' ? 'danger' : 'success'})">
                    ${item.direction === 'AI' ? '+' : '-'}${item.value.toFixed(3)}
                </div>
            </div>
        `;
    });
    
    html += `</div>
        <div style="margin-top:16px; display:flex; gap:16px; justify-content:center; font-size:12px; color:var(--text-muted)">
            <span style="display:flex; align-items:center; gap:4px;"><span style="display:inline-block; width:12px; height:12px; background:var(--danger); border-radius:2px"></span> Pushes prediction towards AI</span>
            <span style="display:flex; align-items:center; gap:4px;"><span style="display:inline-block; width:12px; height:12px; background:var(--success); border-radius:2px"></span> Pushes prediction towards Real</span>
        </div>
    </div>`;
    
    container.innerHTML = html;
    
    // Animate
    setTimeout(() => {
        shap.forEach((item, idx) => {
            const bar = document.getElementById(`shap-bar-${idx}`);
            if (bar) bar.style.width = `${(item.value / maxVal) * 100}%`;
        });
    }, 50);
};

window.renderDetailModels = function(data) {
    const container = document.getElementById('detect-detail-content');
    // Fetch from model-info synchronously to keep render instant
    // We already passed models via api if needed, but we can hardcode for the view
    
    const descriptions = {
        'clip': 'Semantic analysis via OpenCLIP ViT-L/14 — 768-dim L2-normalized embeddings compared against AI/real text prompts',
        'efficientnet': 'Pixel-level artifact detection via EfficientNetV2-Large — 1280-dim intermediate features with global average pooling',
        'frequency': 'FFT frequency spectrum analysis — mid-frequency periodicity, spectral flatness, and high-frequency energy patterns',
        'srm': 'Spatial Rich Model noise residuals — camera sensor noise signatures via 3 SRM filters × 3 channels with statistical features',
        'exif': 'EXIF metadata verification — camera device signatures, GPS data, software tags, and timestamp correlation'
    };
    
    let html = `
        <table style="width:100%; border-collapse:collapse; text-align:left; font-size:14px">
            <thead>
                <tr style="border-bottom:1px solid var(--border); color:var(--text-muted)">
                    <th style="padding:10px">Detector Module</th>
                    <th style="padding:10px">Score</th>
                    <th style="padding:10px">What it detects</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    Object.entries(data.scores).forEach(([key, val]) => {
        if (key === 'ensemble') return;
        const color = val > 0.5 ? 'var(--danger)' : 'var(--success)';
        html += `
            <tr style="border-bottom:1px solid var(--border-light)">
                <td style="padding:12px 10px; font-weight:500; text-transform:capitalize">${key.replace('_', ' ')}</td>
                <td style="padding:12px 10px; color:${color}; font-weight:600">${(val*100).toFixed(1)}%</td>
                <td style="padding:12px 10px; color:var(--text-muted)">${descriptions[key] || ''}</td>
            </tr>
        `;
    });
    
    html += `</tbody></table>`;
    container.innerHTML = html;
};

window.renderDetailRaw = function(data) {
    const container = document.getElementById('detect-detail-content');
    
    // Create copy without base64 heavily-encoded fields
    const displayData = JSON.parse(JSON.stringify(data));
    if (displayData.explanation.gradcam_heatmap) displayData.explanation.gradcam_heatmap = "[BASE64_JPEG_DATA_OMITTED]";
    if (displayData.explanation.frequency_spectrum) displayData.explanation.frequency_spectrum = "[2D_ARRAY_OMITTED]";
    if (displayData.explanation.frequency_profile) displayData.explanation.frequency_profile = "[1D_ARRAY_OMITTED]";
    
    const text = JSON.stringify(displayData, null, 2);
    
    container.innerHTML = `
        <div style="display:flex; justify-content:flex-end; margin-bottom:8px;">
            <button class="btn-secondary" onclick="navigator.clipboard.writeText(document.getElementById('raw-json').textContent); this.textContent='Copied!'; setTimeout(()=>this.textContent='Copy JSON', 2000)">Copy JSON</button>
        </div>
        <pre id="raw-json" style="background:var(--bg-card); padding:16px; border-radius:var(--radius); border:1px solid var(--border); overflow-x:auto; font-size:13px; color:#c9d1d9; font-family:monospace; margin:0">${text}</pre>
    `;
};
