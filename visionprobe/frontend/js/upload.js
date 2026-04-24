function initDropZone(zoneId, fileInputId, onFile) {
    const zone = document.getElementById(zoneId);
    if (!zone) return;
    const input = document.getElementById(fileInputId);
    
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', e => { if (e.target.files[0]) onFile(e.target.files[0]); });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) onFile(f);
    });
}

async function loadFromURL(module) {
    const input = document.getElementById(module + '-url-input');
    const url = input.value.trim();
    if (!url) return;
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        if (!blob.type.startsWith('image/')) throw new Error('Not an image');
        const file = new File([blob], 'url-image.jpg', {type: blob.type});
        window.loadDetectorFile(file);
    } catch (e) {
        alert('Could not load image from URL: ' + e.message);
    }
}

function showImagePreview(imgElementId, metaElementId, file) {
    const img = document.getElementById(imgElementId);
    const meta = document.getElementById(metaElementId);
    const url = URL.createObjectURL(file);
    img.onload = () => {
        if (meta) meta.textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight}px · ${(file.size/1024).toFixed(0)}KB`;
        URL.revokeObjectURL(url);
    };
    img.src = url;
}

// Detector-specific file handler
window.detectorFile = null;
window.loadDetectorFile = function(file) {
    window.detectorFile = file;
    document.getElementById('detect-preview-area').classList.remove('hidden');
    showImagePreview('detect-preview-img', 'detect-meta', file);
    document.getElementById('detect-btn').disabled = false;
    
    // Hide previous results if any
    document.getElementById('detect-results-panel').classList.add('hidden');
    document.getElementById('detect-detail').classList.add('hidden');
};

document.addEventListener('DOMContentLoaded', () => {
    initDropZone('detect-drop-zone', 'detect-file-input', window.loadDetectorFile);
});
