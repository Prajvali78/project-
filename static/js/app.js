/**
 * NeuroVote AI - Frontend Client Logic (v2.0 SOTA Multi-Model Support)
 * Handles interactive tabs, theme toggle, clinical presets,
 * dual ensemble modes (Standard 4-Model vs Advanced 10-Model Super-Ensemble),
 * dynamic multi-model probability meters, and clinical risk factor interpretation.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const form = document.getElementById('prediction-form');
  const btnPredict = document.getElementById('btn-predict');
  const btnReset = document.getElementById('btn-reset');
  const spinner = document.getElementById('predict-spinner');
  
  const resultsEmpty = document.getElementById('results-empty');
  const resultsContent = document.getElementById('results-content');
  const statusIndicator = document.getElementById('status-indicator');

  const verdictCard = document.getElementById('verdict-card');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictSubtitle = document.getElementById('verdict-subtitle');
  const verdictScoreLabel = document.getElementById('verdict-score-label');
  const verdictProb = document.getElementById('verdict-probability');
  const riskBadge = document.getElementById('risk-badge');
  const verdictDualProbs = document.getElementById('verdict-dual-probs');
  const verdictEngineLabel = document.getElementById('verdict-engine-label');

  const dynamicClfContainer = document.getElementById('dynamic-classifier-container');
  const activeModelsCounter = document.getElementById('active-models-counter');
  const factorList = document.getElementById('factor-list');
  const themeToggle = document.getElementById('theme-toggle');

  // Mode Switcher Elements
  const modeBtns = document.querySelectorAll('.btn-mode');
  const currentModeTitle = document.getElementById('current-mode-title');
  const currentModeDesc = document.getElementById('current-mode-desc');
  let currentEnsembleMode = 'advanced';

  // Last input payload cache for instant re-eval on mode toggle
  let lastPayload = null;

  // =========================================================================
  // 1. Navigation Tab Switching
  // =========================================================================
  const tabs = document.querySelectorAll('.nav-tab');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.dataset.tab;

      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });

  // =========================================================================
  // 2. Theme Toggling (Dark / Light)
  // =========================================================================
  const savedTheme = localStorage.getItem('neurovote_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);

  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('neurovote_theme', next);
  });

  // =========================================================================
  // 3. Ensemble Mode Switching (Advanced vs Standard)
  // =========================================================================
  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const selectedMode = btn.dataset.mode;
      if (selectedMode === currentEnsembleMode) return;

      currentEnsembleMode = selectedMode;
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      if (currentEnsembleMode === 'advanced') {
        currentModeTitle.textContent = 'Active Engine: 10-Model SOTA Super-Ensemble';
        currentModeDesc.textContent = 'Combines CatBoost, LightGBM, XGBoost, Random Forest, Extra Trees, MLP, SVC, Gradient Boosting, KNN, and ElasticNet via adaptive F1-weighted soft voting (99.0% Accuracy, Zero False Positives).';
      } else {
        currentModeTitle.textContent = 'Active Engine: Standard 4-Model Clinical Ensemble';
        currentModeDesc.textContent = 'F1-score-weighted soft voting across 4 core clinical classifiers: Random Forest, XGBoost, Support Vector Machine, and Multi-Layer Perceptron (96.0% Benchmark Accuracy).';
      }

      // If user already generated a prediction, re-run with new ensemble mode
      if (lastPayload) {
        executePrediction(lastPayload);
      }
    });
  });

  // =========================================================================
  // 4. Clinical Demo Presets (Calibrated to Dataset Distributions)
  // =========================================================================
  const presets = {
    healthy: {
      MMSE: 18.0,
      FunctionalAssessment: 6.5,
      ADL: 6.8,
      Age: 72,
      MemoryComplaints: "0",
      Confusion: "0",
      BehavioralProblems: "0",
      Hippocampal_Volume: 3750,
      MRI_PET_Imaging_Scores: 24.0,
      "CSF Abeta42 Levels": 850,
      "APOE4 Gene Presence": "0"
    },
    early_mci: {
      MMSE: 14.0,
      FunctionalAssessment: 4.8,
      ADL: 4.9,
      Age: 74,
      MemoryComplaints: "1",
      Confusion: "0",
      BehavioralProblems: "0",
      Hippocampal_Volume: 3400,
      MRI_PET_Imaging_Scores: 38.0,
      "CSF Abeta42 Levels": 720,
      "APOE4 Gene Presence": "1"
    },
    ad: {
      MMSE: 10.0,
      FunctionalAssessment: 2.5,
      ADL: 2.8,
      Age: 78,
      MemoryComplaints: "1",
      Confusion: "1",
      BehavioralProblems: "1",
      Hippocampal_Volume: 2800,
      MRI_PET_Imaging_Scores: 60.0,
      "CSF Abeta42 Levels": 520,
      "APOE4 Gene Presence": "1"
    }
  };

  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const pKey = btn.dataset.preset;
      const data = presets[pKey];
      if (!data) return;

      // Populate form fields
      for (const [key, val] of Object.entries(data)) {
        const input = form.elements[key];
        if (input) {
          input.value = val;
        }
      }

      // Visual feedback
      btn.style.transform = 'scale(0.95)';
      setTimeout(() => { btn.style.transform = ''; }, 150);
    });
  });

  // Reset Button
  btnReset.addEventListener('click', () => {
    form.reset();
    lastPayload = null;
    resultsContent.classList.add('hidden');
    resultsEmpty.classList.remove('hidden');
    statusIndicator.className = 'status-pill status-ready';
    statusIndicator.textContent = 'Ready for Evaluation';
  });

  // =========================================================================
  // 5. Form Submission & Prediction Handler
  // =========================================================================
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect data
    const formData = new FormData(form);
    const payload = {};
    for (const [k, v] of formData.entries()) {
      payload[k] = parseFloat(v);
    }

    lastPayload = payload;
    await executePrediction(payload);
  });

  async function executePrediction(payload) {
    // UI Loading state
    btnPredict.disabled = true;
    spinner.classList.remove('hidden');
    statusIndicator.className = 'status-pill';
    statusIndicator.textContent = 'Querying Ensemble...';

    // Inject active mode
    payload.ensemble_mode = currentEnsembleMode;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const result = await response.json();
      renderPrediction(result, payload);

    } catch (err) {
      console.error('Prediction request failed:', err);
      alert('Prediction Error: ' + err.message);
      statusIndicator.textContent = 'Error';
    } finally {
      btnPredict.disabled = false;
      spinner.classList.add('hidden');
    }
  }

  // =========================================================================
  // 6. Render Results in UI
  // =========================================================================
  function renderPrediction(res, inputData) {
    resultsEmpty.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    statusIndicator.className = 'status-pill status-active';
    statusIndicator.textContent = 'Evaluation Complete';

    const isAD = res.prediction === 1;
    const adProb = res.ensemble_probability !== undefined ? res.ensemble_probability : 0.0;
    const healthyProb = res.ensemble_probability_healthy !== undefined ? res.ensemble_probability_healthy : (1.0 - adProb);
    const adProbPercent = (adProb * 100).toFixed(1);
    const healthyProbPercent = (healthyProb * 100).toFixed(1);
    const confPercent = res.ensemble_confidence !== undefined ? res.ensemble_confidence.toFixed(1) : (isAD ? adProbPercent : healthyProbPercent);

    // Update Verdict Banner
    verdictCard.className = `verdict-banner ${isAD ? 'positive' : 'negative'}`;
    if (verdictEngineLabel) {
      verdictEngineLabel.textContent = res.ensemble_name || (res.ensemble_mode === 'advanced' ? '10-Model Super-Ensemble' : '4-Model Clinical Ensemble');
    }
    verdictTitle.textContent = isAD ? "Alzheimer's Disease Detected" : "Non-Alzheimer (Cognitively Preserved)";
    verdictSubtitle.textContent = isAD
      ? "Diagnostic indicators and biological biomarkers exhibit characteristic Alzheimer's patterns."
      : "Clinical metrics and biomarker concentrations remain within the normative cognitive range.";
    
    if (verdictScoreLabel) {
      verdictScoreLabel.textContent = isAD ? "AD Risk Probability" : "Diagnostic Confidence";
    }

    verdictProb.textContent = isAD ? `${adProbPercent}%` : `${confPercent}%`;

    // Verdict Icon
    verdictIcon.innerHTML = isAD
      ? `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
      : `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;

    // Risk Badge
    riskBadge.className = `risk-badge ${res.risk_level === 'Low' ? 'risk-low' : res.risk_level === 'Moderate' ? 'risk-moderate' : 'risk-high'}`;
    riskBadge.textContent = isAD ? `${res.risk_level} Risk Category` : `Healthy (${res.risk_level} Risk)`;

    // Dual probabilities subdetail
    if (verdictDualProbs) {
      verdictDualProbs.textContent = `P(Healthy): ${healthyProbPercent}%  •  P(AD Risk): ${adProbPercent}%`;
    }

    // Active models counter badge
    activeModelsCounter.textContent = `${res.models_count} Models Queried (${res.ensemble_mode.toUpperCase()})`;

    // Render Dynamic Multi-Classifier Progress Meters
    renderDynamicClassifiers(res.models_detail || res.individual_probabilities, res.weights);

    // Build Clinical Interpretation Notes
    generateClinicalFactors(inputData, isAD);
  }

  // =========================================================================
  // 7. Dynamic Multi-Classifier Progress Bars (Informative & Vibrant)
  // =========================================================================
  function renderDynamicClassifiers(modelsDetail, weightsMap) {
    dynamicClfContainer.innerHTML = '';

    // Family categories definition
    const categories = [
      { id: 'boosting', title: '🚀 State-of-the-Art Gradient Boosters', models: ['CatBoost', 'LightGBM', 'XGBoost', 'Gradient Boosting'] },
      { id: 'trees', title: '🌲 Bagged & Randomized Tree Ensembles', models: ['Random Forest', 'Extra Trees'] },
      { id: 'neural_svm', title: '🧠 Deep Neural Networks & Kernel SVM', models: ['MLP', 'SVC'] },
      { id: 'statistical', title: '📊 Regularized Linear & Instance Classifiers', models: ['KNN', 'ElasticNet LogReg'] }
    ];

    // Determine present models
    const activeModelNames = Object.keys(modelsDetail);

    categories.forEach(cat => {
      const catModels = cat.models.filter(m => activeModelNames.includes(m));
      if (catModels.length === 0) return;

      const familyBlock = document.createElement('div');
      familyBlock.className = 'clf-family-block';

      const familyHeader = document.createElement('div');
      familyHeader.className = 'clf-family-header';
      familyHeader.innerHTML = `<h5>${cat.title}</h5>`;
      familyBlock.appendChild(familyHeader);

      const familyGrid = document.createElement('div');
      familyGrid.className = 'clf-family-grid';

      catModels.forEach(mName => {
        const item = modelsDetail[mName];
        let probAD = 0.0;
        let probHealthy = 1.0;
        let conf = 100.0;
        let predClass = 0;
        let weight = 0.1;
        let tag = 'Classifier';

        if (typeof item === 'object') {
          probAD = item.probability_ad !== undefined ? item.probability_ad : (item.probability !== undefined ? item.probability : 0.0);
          probHealthy = item.probability_healthy !== undefined ? item.probability_healthy : (1.0 - probAD);
          conf = item.confidence !== undefined ? item.confidence : Math.max(probAD, probHealthy) * 100;
          predClass = item.predicted_class !== undefined ? item.predicted_class : (probAD >= 0.5 ? 1 : 0);
          weight = item.weight !== undefined ? item.weight : (weightsMap && weightsMap[mName] !== undefined ? weightsMap[mName] : 0.1);
          tag = item.family || 'Classifier';
        } else {
          probAD = item;
          probHealthy = 1.0 - item;
          conf = Math.max(probAD, probHealthy) * 100;
          predClass = probAD >= 0.5 ? 1 : 0;
          weight = weightsMap && weightsMap[mName] !== undefined ? weightsMap[mName] : 0.1;
        }

        const probADPercent = (probAD * 100).toFixed(1);
        const probHealthyPercent = (probHealthy * 100).toFixed(1);
        const confPercent = conf.toFixed(1);
        const weightPercent = (weight * 100).toFixed(1);

        // Status badge and bar fill logic:
        // When model predicts Healthy: show healthy badge and fill meter with healthy confidence!
        // When model predicts AD: show AD risk badge and fill meter with AD risk probability!
        const isModelAD = predClass === 1;
        let statusBadgeHtml = '';
        let fillWidthPercent = '';
        let fillGradient = '';

        if (isModelAD) {
          if (probAD >= 0.70) {
            statusBadgeHtml = `<span class="clf-status-badge ad">⚠ AD Risk: ${probADPercent}%</span>`;
            fillGradient = 'linear-gradient(90deg, #f59e0b 0%, #ef4444 100%)';
          } else {
            statusBadgeHtml = `<span class="clf-status-badge mci">⚡ Moderate Risk: ${probADPercent}%</span>`;
            fillGradient = 'linear-gradient(90deg, #06b6d4 0%, #f59e0b 100%)';
          }
          fillWidthPercent = Math.max(parseFloat(probADPercent), 8.0);
        } else {
          statusBadgeHtml = `<span class="clf-status-badge healthy">✓ Healthy: ${probHealthyPercent}%</span>`;
          fillGradient = 'linear-gradient(90deg, #10b981 0%, #06b6d4 100%)';
          fillWidthPercent = Math.max(parseFloat(probHealthyPercent), 8.0);
        }

        const itemDiv = document.createElement('div');
        itemDiv.className = 'classifier-item-modern';
        itemDiv.innerHTML = `
          <div class="clf-meta-row">
            <div class="clf-name-group">
              <strong>${mName}</strong>
              <span class="clf-tag">${tag}</span>
            </div>
            <div class="clf-val-group">
              <span class="clf-weight-pill">Weight: ${weightPercent}%</span>
              ${statusBadgeHtml}
            </div>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style="width: 0%; background: ${fillGradient}"></div>
          </div>
          <div class="clf-sub-metrics">
            <span>Confidence: <strong>${confPercent}%</strong></span>
            <span>P(AD Risk): <strong>${probADPercent}%</strong>  •  P(Healthy): <strong>${probHealthyPercent}%</strong></span>
          </div>
        `;

        familyGrid.appendChild(itemDiv);

        // Trigger animation after DOM insertion
        setTimeout(() => {
          const fillBar = itemDiv.querySelector('.progress-fill');
          if (fillBar) fillBar.style.width = `${fillWidthPercent}%`;
        }, 50);
      });

      familyBlock.appendChild(familyGrid);
      dynamicClfContainer.appendChild(familyBlock);
    });
  }

  // =========================================================================
  // 8. Clinical Interpretation Generator
  // =========================================================================
  function generateClinicalFactors(input, isAD) {
    factorList.innerHTML = '';
    const factors = [];

    // MMSE (Dataset mean ~14.7)
    if (input.MMSE <= 12.0) {
      factors.push({ type: 'warning', text: `MMSE score is severely depressed (${input.MMSE}/30), indicating marked cognitive decline.` });
    } else if (input.MMSE <= 15.5) {
      factors.push({ type: 'warning', text: `MMSE score shows borderline impairment (${input.MMSE}/30).` });
    } else {
      factors.push({ type: 'normal', text: `MMSE score is within the normal cognitive cohort range (${input.MMSE}/30).` });
    }

    // Functional & ADL (Dataset mean ~5.0)
    if (input.FunctionalAssessment <= 3.8 || input.ADL <= 4.0) {
      factors.push({ type: 'warning', text: `Substantial functional loss observed in Daily Living Activities (ADL: ${input.ADL}, Functional: ${input.FunctionalAssessment}).` });
    } else {
      factors.push({ type: 'normal', text: `Daily Living Activities and functional autonomy are well-preserved (ADL: ${input.ADL}, Functional: ${input.FunctionalAssessment}).` });
    }

    // Hippocampal Volume (Dataset mean ~3500)
    if (input.Hippocampal_Volume < 3300) {
      factors.push({ type: 'warning', text: `Hippocampal volume is markedly reduced (${input.Hippocampal_Volume} mm³), signifying medial temporal lobe atrophy.` });
    } else {
      factors.push({ type: 'normal', text: `Hippocampal volume is preserved (${input.Hippocampal_Volume} mm³), with minimal volume loss.` });
    }

    // CSF Abeta42 (Dataset mean ~755)
    if (input["CSF Abeta42 Levels"] < 650) {
      factors.push({ type: 'warning', text: `CSF Aβ42 levels are significantly low (${input["CSF Abeta42 Levels"]} pg/mL), consistent with cortical amyloid plaque aggregation.` });
    } else {
      factors.push({ type: 'normal', text: `CSF Aβ42 levels are normative (${input["CSF Abeta42 Levels"]} pg/mL).` });
    }

    // APOE4
    if (input["APOE4 Gene Presence"] === 1) {
      factors.push({ type: 'warning', text: `Carrier of the APOE ε4 susceptibility allele (increased epidemiological genetic vulnerability).` });
    }

    // Render list
    factors.forEach(f => {
      const li = document.createElement('li');
      li.className = `factor-item ${f.type}`;
      li.innerHTML = f.type === 'warning'
        ? `<span>⚠</span> ${f.text}`
        : `<span>✓</span> ${f.text}`;
      factorList.appendChild(li);
    });
  }

});
