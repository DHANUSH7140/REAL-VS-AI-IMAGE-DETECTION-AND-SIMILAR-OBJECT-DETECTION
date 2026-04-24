const API_BASE = '/api';

async function apiAnalyzeImage(file, options = {}) {
    const fd = new FormData();
    if (typeof file === 'string') {
         fd.append('url', file); // file is actually url string
    } else {
         fd.append('image', file);
    }
    
    if (options.includeGradcam !== undefined)
        fd.append('include_gradcam', options.includeGradcam);
    if (options.includeShap !== undefined)
        fd.append('include_shap', options.includeShap);
    
    const res = await fetch(`${API_BASE}/analyze`, {method:'POST', body:fd});
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}
