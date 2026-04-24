import re

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove element declarations
content = re.sub(r'\s*// Combined Results.*?// Patch Results', '\n      // Patch Results', content, flags=re.DOTALL)
content = re.sub(r'\s*// ROI Canvas.*?// History', '\n      // History', content, flags=re.DOTALL)

# 2. Remove isSimilarMode, updateSimilarModeUI
content = re.sub(r'\s*function isSimilarMode\(\) \{.*?\}\n', '\n', content, flags=re.DOTALL)
content = re.sub(r'\s*function updateSimilarModeUI\(\) \{.*?\}\n', '\n', content, flags=re.DOTALL)

# 3. Remove EPS slider and View Mode buttons
content = re.sub(r'\s*// EPS slider handler.*?// Listen for model selector changes', '\n      // Listen for model selector changes', content, flags=re.DOTALL)

# 4. Remove ROI setup call from model selector change
content = re.sub(r'updateSimilarModeUI\(\);\s*hideAllResults\(\);\s*if \(isSimilarMode\(\) && selectedFile\) \{\s*setupROICanvas\(\);\s*\}', 'hideAllResults();', content, flags=re.DOTALL)

# 5. Remove isSimilarMode check inside setPreview
content = re.sub(r'if \(isSimilarMode\(\)\) \{\s*setupROICanvas\(\);\s*\}', '', content, flags=re.DOTALL)

# 6. Remove resetUI roi lines
content = re.sub(r'hasROI = false;\s*roiCanvasContainer\.classList\.remove\(\'show\'\);\s*roiCoords\.style\.display = \'none\';\s*roiInstructions\.style\.display = \'\';', '', content, flags=re.DOTALL)

# 7. Remove the entire ROI setup and event listener block
# Search from /* ── ROI Canvas Setup down to before /* ── Browse click 
content = re.sub(r'\s*/\* ── ROI Canvas Setup.*?(?=/\* ── Browse click)', '\n\n      ', content, flags=re.DOTALL)

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
