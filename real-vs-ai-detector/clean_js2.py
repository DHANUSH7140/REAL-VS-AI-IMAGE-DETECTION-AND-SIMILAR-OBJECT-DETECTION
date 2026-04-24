import re

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove renderCombinedResults and renderSimilarityResults completely
content = re.sub(r'/\* ── Render combined results.*?/\* ── Render patch analysis results', '/* ── Render patch analysis results', content, flags=re.DOTALL)
content = re.sub(r'/\* ── Render similarity results.*?/\* ── Render patch analysis results', '/* ── Render patch analysis results', content, flags=re.DOTALL)

# Cleanup history label logic
content = re.sub(r'label: data\.label \|\| \(data\.mode === \'similar\'.*?\'Real Image\'\)\),', 'label: data.label || \'Real Image\',', content, flags=re.DOTALL)
content = re.sub(r'confidence: data\.confidence \|\| \(data\.mode === \'similar\'.*?0\)\),', 'confidence: data.confidence || 0,', content, flags=re.DOTALL)
content = re.sub(r'model: data\.model_used \|\| \(data\.mode === \'similar\'.*?\.toUpperCase\(\)\)\),', 'model: data.model_used || \'COMBINED\',', content, flags=re.DOTALL)

# Cleanup checkBtn event listener
new_listener = """
      /* ── Check button (upload + predict) ────────────────────── */
      checkBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        hideError();
        hideAllResults();

        const selectedMode = document.querySelector('input[name="model"]:checked').value;
        const isExplain = selectedMode === 'explain';

        // Show spinner
        checkBtn.disabled = true;
        checkBtn.innerHTML = '<span class="spinner"></span> Analyzing…';

        const formData = new FormData();
        formData.append('file', selectedFile);

        let endpoint;

        // Both "detect" and "explain" use ensemble (or the new unified pipeline)
        formData.append('model', 'ensemble');
        endpoint = '/predict';

        try {
          const resp = await fetch(endpoint, { method: 'POST', body: formData });
          const data = await resp.json();

          if (!resp.ok) {
            showError(data.error || 'Server error.');
            return;
          }

          // ── Detect / Explain mode results ────────────────────
          const isAI = data.label === 'AI Generated';
          resultHeader.className = 'result-header ' + (isAI ? 'ai' : 'real');

          resultLabel.textContent = data.label;
          resultLabel.className = 'result-label ' + (isAI ? 'ai' : 'real');

          resultConf.innerHTML = `Confidence: <strong>${data.confidence}%</strong>`;
          resultModel.textContent = `Model: ${data.model_used}`;

          // Animate gauge
          const pct = data.confidence;
          confidenceGauge.style.setProperty('--gauge-pct', pct);
          if (isAI) {
            confidenceGauge.style.background = `conic-gradient(var(--danger) ${pct}%, rgba(255,255,255,0.06) 0)`;
          } else {
            confidenceGauge.style.background = `conic-gradient(var(--success) ${pct}%, rgba(255,255,255,0.06) 0)`;
          }
          gaugeValue.textContent = `${pct}%`;

          // Individual results (ensemble)
          if (data.individual_results) {
            renderIndividualResults(data.individual_results);
          } else {
            individualResults.style.display = 'none';
          }

          // Explanation card
          if (isExplain) {
            renderExplanation(data.explanation, data.label);

            // Grad-CAM visualization
            if (data.gradcam_url) {
              gradcamCard.style.display = 'block';
              gradcamImg.src = data.gradcam_url;
              originalCard.style.display = 'block';
              originalImg.src = data.image_url;
            } else {
              gradcamCard.style.display = 'none';
              originalCard.style.display = 'none';
            }

            // FFT analysis
            renderFFTAnalysis(data.fft_analysis, 'standard');
          } else {
            explanationCard.style.display = 'none';
            gradcamCard.style.display = 'none';
            originalCard.style.display = 'none';
            fftCard.style.display = 'none';
          }

          // Show results section
          resultsSection.classList.add('show');

          // Add to history
          addToHistory(data);

          // Auto-scroll to explanation if "Explain" mode
          if (isExplain) {
            setTimeout(() => {
              const explCard = document.getElementById('explanationCard');
              if (explCard) explCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 300);
          }

        } catch (err) {
          showError('Could not reach the server. Is Flask running?');
        } finally {
          checkBtn.disabled = false;
          checkBtn.textContent = '🔍 Analyze Image';
        }
      });

      /* ── Prevent default drag on window ─────────────────────── */
"""

content = re.sub(r'/\* ── Check button \(upload \+ predict\).*?/\* ── Prevent default drag on window ─────────────────────── \*/', new_listener, content, flags=re.DOTALL)

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
