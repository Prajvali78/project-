/**
 * NeuroVote AI - Frontend Client Logic
 * Handles interactive tabs, demo presets, real-time prediction requests,
 * animated probability progress meters, and clinical risk factor interpretation.
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
  const verdictProb = document.getElementById('verdict-probability');
  const riskBadge = document.getElementById('risk-badge');

  const barRf = document.getElementById('bar-rf');
  const barXgb = document.getElementById('bar-xgb');
  const barSvc = document.getElementById('bar-svc');
  const barMlp = document.getElementById('bar-mlp');

  const pRf = document.getElementById('p-rf');
  const pXgb = document.getElementById('p-xgb');
  const pSvc = document.getElementById('p-svc');
  const pMlp = document.getElementById('p-mlp');

  const wRf = document.getElementById('w-rf');
  const wXgb = document.getElementById('w-xgb');
  const wSvc = document.getElementById('w-svc');
  const wMlp = document.getElementById('w-mlp');

  const factorList = document.getElementById('factor-list');
  const themeToggle = document.getElementById('theme-toggle');

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
  // 3. Clinical Demo Presets
  // =========================================================================
  const presets = {
    healthy: {
      MMSE: 29.0,
      FunctionalAssessment: 9.2,
      ADL: 9.5,
      Age: 68,
      MemoryComplaints: "0",
      Confusion: "0",
      BehavioralProblems: "0",
      Hippocampal_Volume: 3950,
      MRI_PET_Imaging_Scores: 18.0,
      "CSF Abeta42 Levels": 940,
      "APOE4 Gene Presence": "0"
    },
    early_mci: {
      MMSE: 24.5,
      FunctionalAssessment: 5.6,
      ADL: 6.2,
      Age: 74,
      MemoryComplaints: "1",
      Confusion: "0",
      BehavioralProblems: "0",
      Hippocampal_Volume: 3380,
      MRI_PET_Imaging_Scores: 44.0,
      "CSF Abeta42 Levels": 670,
      "APOE4 Gene Presence": "1"
    },
    ad: {
      MMSE: 16.0,
      FunctionalAssessment: 2.6,
      ADL: 3.2,
      Age: 79,
      MemoryComplaints: "1",
      Confusion: "1",
      BehavioralProblems: "1",
      Hippocampal_Volume: 2780,
      MRI_PET_Imaging_Scores: 66.0,
      "CSF Abeta42 Levels": 470,
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
    resultsContent.classList.add('hidden');
    resultsEmpty.classList.remove('hidden');
    statusIndicator.className = 'status-pill status-ready';
    statusIndicator.textContent = 'Ready for Evaluation';
  });

  // =========================================================================
  // 4. Form Submission & Prediction Handler
  // =========================================================================
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // UI Loading state
    btnPredict.disabled = true;
    spinner.classList.remove('hidden');
    statusIndicator.className = 'status-pill';
    statusIndicator.textContent = 'Predicting...';

    // Collect data
    const formData = new FormData(form);
    const payload = {};
    for (const [k, v] of formData.entries()) {
      payload[k] = parseFloat(v);
    }

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
  });

  // =========================================================================
  // 5. Render Results in UI
  // =========================================================================
  function renderPrediction(res, inputData) {
    resultsEmpty.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    statusIndicator.className = 'status-pill status-active';
    statusIndicator.textContent = 'Evaluation Complete';

    const isAD = res.prediction === 1;
    const adProbPercent = (res.ensemble_probability * 100).toFixed(1);

    // Update Verdict Banner
    verdictCard.className = `verdict-banner ${isAD ? 'positive' : 'negative'}`;
    verdictTitle.textContent = isAD ? "Alzheimer's Disease Detected" : "Non-Alzheimer (Cognitively Preserved)";
    verdictSubtitle.textContent = isAD
      ? "Diagnostic indicators and neuro-imaging biomarkers exhibit characteristic AD patterns."
      : "Clinical metrics and biomarker concentrations remain within the normative cognitive range.";
    
    verdictProb.textContent = `${adProbPercent}%`;

    // Verdict Icon
    verdictIcon.innerHTML = isAD
      ? `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
      : `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;

    // Risk Badge
    riskBadge.className = `risk-badge ${res.risk_level === 'Low' ? 'risk-low' : res.risk_level === 'Moderate' ? 'risk-moderate' : 'risk-high'}`;
    riskBadge.textContent = `${res.risk_level} Risk Category`;

    // Update Base Classifiers Progress Bars & Probabilities
    const probs = res.individual_probabilities;
    const weights = res.weights;

    updateBar(barRf, pRf, wRf, probs['Random Forest'], weights['Random Forest']);
    updateBar(barXgb, pXgb, wXgb, probs['XGBoost'], weights['XGBoost']);
    updateBar(barSvc, pSvc, wSvc, probs['SVC'], weights['SVC']);
    updateBar(barMlp, pMlp, wMlp, probs['MLP'], weights['MLP']);

    // Build Clinical Interpretation Notes
    generateClinicalFactors(inputData, isAD);
  }

  function updateBar(barEl, textEl, weightEl, probValue, weightValue) {
    if (!barEl || probValue === undefined) return;
    const percent = (probValue * 100).toFixed(1);
    barEl.style.width = `${percent}%`;
    textEl.textContent = `${percent}% P(AD)`;
    if (weightEl && weightValue !== undefined) {
      weightEl.textContent = `Weight: ${(weightValue * 100).toFixed(1)}%`;
    }
  }

  function generateClinicalFactors(input, isAD) {
    factorList.innerHTML = '';

    const factors = [];

    // MMSE
    if (input.MMSE <= 21) {
      factors.push({ type: 'warning', text: `MMSE score is severely depressed (${input.MMSE}/30), indicating marked cognitive decline.` });
    } else if (input.MMSE <= 26) {
      factors.push({ type: 'warning', text: `MMSE score shows mild cognitive borderline impairment (${input.MMSE}/30).` });
    } else {
      factors.push({ type: 'normal', text: `MMSE score is within the normal cognitive range (${input.MMSE}/30).` });
    }

    // Functional & ADL
    if (input.FunctionalAssessment <= 4.0 || input.ADL <= 4.5) {
      factors.push({ type: 'warning', text: `Substantial functional loss observed in Daily Living Activities (ADL: ${input.ADL}, Functional: ${input.FunctionalAssessment}).` });
    } else {
      factors.push({ type: 'normal', text: `Daily Living Activities and functional autonomy are well-preserved (ADL: ${input.ADL}).` });
    }

    // Hippocampal Volume
    if (input.Hippocampal_Volume < 3200) {
      factors.push({ type: 'warning', text: `Hippocampal volume is markedly reduced (${input.Hippocampal_Volume} mm³), signifying medial temporal lobe atrophy.` });
    } else {
      factors.push({ type: 'normal', text: `Hippocampal volume is preserved (${input.Hippocampal_Volume} mm³), with minimal volume loss.` });
    }

    // CSF Abeta42
    if (input["CSF Abeta42 Levels"] < 600) {
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
