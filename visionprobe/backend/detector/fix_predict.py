import re

with open('visionprobe/backend/detector/predict.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix resizing block 1
content = re.sub(
    r'# Resize if too large\s+w, h = pil_img\.size\s+if max\(w, h\) > 2048:\s+scale = 2048 / max\(w, h\)\s+pil_img = pil_img\.resize\(\(int\(w \* scale\), int\(h \* scale\)\), Image\.LANCZOS\)',
    '# Standardize resolution to match training exactly (512x512)\n        pil_img = pil_img.resize((512, 512), Image.LANCZOS)',
    content
)

# Fix resizing block 2
content = re.sub(
    r'w, h = pil_img\.size\s+if max\(w, h\) > 2048:\s+scale = 2048 / max\(w, h\)\s+pil_img = pil_img\.resize\(\(int\(w \* scale\), int\(h \* scale\)\), Image\.LANCZOS\)',
    '# Standardize resolution to match training exactly (512x512)\n        pil_img = pil_img.resize((512, 512), Image.LANCZOS)',
    content
)

# Fix probability distortion 1
distortion1 = r'# Shift decision boundary to reduce false positives \(model is overly sensitive to faces\)\s+threshold = 0\.65\s+label = "AI" if prob >= threshold else "REAL"\s+# Apply non-linear scaling to push confidence away from the threshold\s+import numpy as np\s+if label == "AI":\s+x = min\(1\.0, \(prob - threshold\) / \(1\.0 - threshold\)\)\s+scaled_x = x \*\* 0\.4\s+confidence = 50\.0 \+ \(scaled_x \* 50\.0\)\s+else:\s+x = min\(1\.0, \(threshold - prob\) / threshold\)\s+scaled_x = x \*\* 0\.4\s+confidence = 50\.0 \+ \(scaled_x \* 50\.0\)'
replacement1 = r'''# Rely on the calibrated classifier's genuine probability
        threshold = 0.50
        label = "AI" if prob >= threshold else "REAL"
        
        if label == "AI":
            confidence = prob * 100.0
        else:
            confidence = (1.0 - prob) * 100.0'''

content = re.sub(distortion1, replacement1, content)

# Fix probability distortion 2
distortion2 = r'# Shift decision boundary to reduce false positives\s+threshold = 0\.65\s+label = "AI" if prob >= threshold else "REAL"\s+import numpy as np\s+if label == "AI":\s+x = min\(1\.0, \(prob - threshold\) / \(1\.0 - threshold\)\)\s+scaled_x = x \*\* 0\.4\s+confidence = 50\.0 \+ \(scaled_x \* 50\.0\)\s+else:\s+x = min\(1\.0, \(threshold - prob\) / threshold\)\s+scaled_x = x \*\* 0\.4\s+confidence = 50\.0 \+ \(scaled_x \* 50\.0\)'
replacement2 = r'''# Rely on the calibrated classifier's genuine probability
        threshold = 0.50
        label = "AI" if prob >= threshold else "REAL"
        
        if label == "AI":
            confidence = prob * 100.0
        else:
            confidence = (1.0 - prob) * 100.0'''

content = re.sub(distortion2, replacement2, content)

with open('visionprobe/backend/detector/predict.py', 'w', encoding='utf-8') as f:
    f.write(content)
