function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    document.getElementById('theme-icon').textContent = next === 'light' ? '◐' : '◑';
    
    // Redraw charts if they exist
    if (window.currentDetectorData) {
        if (window.renderDetailFrequency) {
            window.renderDetailFrequency(window.currentDetectorData);
        }
        if (window.renderDetailModels) {
            window.renderDetailModels(window.currentDetectorData);
        }
    }
}

// Restore theme on load
(function() {
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    if (saved === 'light') document.getElementById('theme-icon').textContent = '◐';
})();

// Paste support (Ctrl+V anywhere)
document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith('image/')) {
            const file = item.getAsFile();
            if (window.loadDetectorFile) window.loadDetectorFile(file);
            break;
        }
    }
});
