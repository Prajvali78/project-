/**
 * NeuroVote AI - Frontend Client Logic (v2.5 Full SOTA Clinical Decision Support)
 * Integrates:
 * 1. Multi-tab navigation & Dark/Light mode switcher
 * 2. Dual Ensemble engines (Standard 4-Model vs Advanced 10-Model Super-Ensemble)
 * 3. Clinical demo presets calibrated to longitudinal dataset distributions
 * 4. Dynamic Multi-Classifier posterior meters grouped by algorithmic family
 * 5. Patient-Specific Local Feature Attribution (SHAP Waterfall decomposition)
 * 6. Clinical Disease Staging & 3-Year Longitudinal Progression Trajectory
 * 7. Interactive 'What-If' Therapeutic Counterfactual Intervention Simulator
 * 8. Formal Printable Clinical Diagnostic Report (Print / PDF ready)
 * 9. Cohort Batch Screening & Hospital Triage Dashboard with live CSV exporter
 */

document.addEventListener('DOMContentLoaded', () => {
  // =========================================================================
  // DOM Elements
  // =========================================================================
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

  // Staging & Trajectory Elements
  const diseaseStageTitle = document.getElementById('disease-stage-title');
  const diseaseStageDesc = document.getElementById('disease-stage-desc');
  const trajectoryGrid = document.getElementById('trajectory-grid');

  // Local SHAP Elements
  const localShapList = document.getElementById('local-shap-list');

  // What-If Simulator Elements
  const btnToggleIntervention = document.getElementById('btn-toggle-intervention');
  const interventionCard = document.getElementById('intervention-simulator-card');
  const btnCloseSim = document.getElementById('btn-close-sim');
  const simFunctional = document.getElementById('sim-functional');
  const simAdl = document.getElementById('sim-adl');
  const simCsf = document.getElementById('sim-csf');
  const simMmse = document.getElementById('sim-mmse');
  const lblSimFunc = document.getElementById('lbl-sim-func');
  const lblSimAdl = document.getElementById('lbl-sim-adl');
  const lblSimCsf = document.getElementById('lbl-sim-csf');
  const lblSimMmse = document.getElementById('lbl-sim-mmse');
  const simBaseRisk = document.getElementById('sim-base-risk');
  const simPostRisk = document.getElementById('sim-post-risk');
  const simReductionBadge = document.getElementById('sim-reduction-badge');

  // Clinical Report Modal Elements
  const btnOpenReport = document.getElementById('btn-open-report');
  const btnCloseReport = document.getElementById('btn-close-report');
  const btnPrintReport = document.getElementById('btn-print-report');
  const reportModal = document.getElementById('report-modal');

  // Batch Screening Elements
  const btnLoadDemoCohort = document.getElementById('btn-load-demo-cohort');
  const batchFileInput = document.getElementById('batch-file-input');
  const triageTableBody = document.getElementById('triage-table-body');
  const batchSearchInput = document.getElementById('batch-search');
  const btnExportBatchCsv = document.getElementById('btn-export-batch-csv');
  const filterBtns = document.querySelectorAll('.btn-filter');

  // KPI Elements
  const kpiTotal = document.getElementById('kpi-total');
  const kpiHigh = document.getElementById('kpi-high');
  const kpiHighPct = document.getElementById('kpi-high-pct');
  const kpiBorderline = document.getElementById('kpi-borderline');
  const kpiBorderlinePct = document.getElementById('kpi-borderline-pct');
  const kpiNormal = document.getElementById('kpi-normal');
  const kpiNormalPct = document.getElementById('kpi-normal-pct');
  const kpiAvgRisk = document.getElementById('kpi-avg-risk');
  const countAll = document.getElementById('count-all');
  const countHigh = document.getElementById('count-high');
  const countBorderline = document.getElementById('count-borderline');
  const countNormal = document.getElementById('count-normal');

  // Mode Switcher Elements
  const modeBtns = document.querySelectorAll('.btn-mode');
  const currentModeTitle = document.getElementById('current-mode-title');
  const currentModeDesc = document.getElementById('current-mode-desc');
  let currentEnsembleMode = 'advanced';

  // Application State
  let lastPayload = null;
  let latestPredictionResult = null;
  let currentSimPatient = null;
  let simDebounceTimer = null;
  let batchCohortData = [];
  let currentBatchFilter = 'all';
  let batchSearchQuery = '';

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

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('neurovote_theme', next);
    });
  }

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

      // If user already evaluated a patient, re-run with new ensemble mode
      if (lastPayload) {
        executePrediction(lastPayload);
      }
    });
  });

  // =========================================================================
  // 4. Clinical Demo Presets
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

      for (const [key, val] of Object.entries(data)) {
        const input = form.elements[key];
        if (input) {
          input.value = val;
        }
      }

      btn.style.transform = 'scale(0.95)';
      setTimeout(() => { btn.style.transform = ''; }, 150);
    });
  });

  // Reset Button
  btnReset.addEventListener('click', () => {
    form.reset();
    lastPayload = null;
    latestPredictionResult = null;
    currentSimPatient = null;
    resultsContent.classList.add('hidden');
    resultsEmpty.classList.remove('hidden');
    if (interventionCard) interventionCard.classList.add('hidden');
    statusIndicator.className = 'status-pill status-ready';
    statusIndicator.textContent = 'Ready for Evaluation';
  });

  // =========================================================================
  // 5. Form Submission & Prediction Handler
  // =========================================================================
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const payload = {};
    for (const [k, v] of formData.entries()) {
      payload[k] = parseFloat(v);
    }

    lastPayload = payload;
    await executePrediction(payload);
  });

  async function executePrediction(payload) {
    btnPredict.disabled = true;
    spinner.classList.remove('hidden');
    statusIndicator.className = 'status-pill';
    statusIndicator.textContent = 'Querying Ensemble...';

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
      latestPredictionResult = result;
      currentSimPatient = Object.assign({}, payload);
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

    verdictIcon.innerHTML = isAD
      ? `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
      : `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;

    riskBadge.className = `risk-badge ${res.risk_level === 'Low' ? 'risk-low' : res.risk_level === 'Moderate' ? 'risk-moderate' : 'risk-high'}`;
    riskBadge.textContent = isAD ? `${res.risk_level} Risk Category` : `Healthy (${res.risk_level} Risk)`;

    if (verdictDualProbs) {
      verdictDualProbs.textContent = `P(Healthy): ${healthyProbPercent}%  •  P(AD Risk): ${adProbPercent}%`;
    }

    activeModelsCounter.textContent = `${res.models_count} Models Queried (${res.ensemble_mode.toUpperCase()})`;

    // 1. Render Multi-Classifier Progress Meters
    renderDynamicClassifiers(res.models_detail || res.individual_probabilities, res.weights);

    // 2. Render Staging & 3-Year Trajectory
    renderStagingAndTrajectory(res);

    // 3. Render Patient-Specific Local Feature Attribution (SHAP)
    renderLocalShapAttributions(res.local_attributions);

    // 4. Reset & Initialize What-If Simulator state
    initInterventionSimulator(adProbPercent);

    // 5. Build Clinical Interpretation Notes
    generateClinicalFactors(inputData, isAD);
  }

  // =========================================================================
  // 7. Dynamic Multi-Classifier Progress Bars
  // =========================================================================
  function renderDynamicClassifiers(modelsDetail, weightsMap) {
    dynamicClfContainer.innerHTML = '';

    const categories = [
      { id: 'boosting', title: '🚀 State-of-the-Art Gradient Boosters', models: ['CatBoost', 'LightGBM', 'XGBoost', 'Gradient Boosting'] },
      { id: 'trees', title: '🌲 Bagged & Randomized Tree Ensembles', models: ['Random Forest', 'Extra Trees'] },
      { id: 'neural_svm', title: '🧠 Deep Neural Networks & Kernel SVM', models: ['MLP', 'SVC'] },
      { id: 'statistical', title: '📊 Regularized Linear & Instance Classifiers', models: ['KNN', 'ElasticNet LogReg'] }
    ];

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
  // 8. Staging Milestones & 3-Year Prognosis Trajectory
  // =========================================================================
  function renderStagingAndTrajectory(res) {
    if (!res.disease_stage) return;

    if (diseaseStageTitle) diseaseStageTitle.textContent = res.disease_stage.title;
    if (diseaseStageDesc) diseaseStageDesc.textContent = res.disease_stage.description;

    // Reset milestone steps
    for (let i = 1; i <= 5; i++) {
      const stepEl = document.getElementById(`step-stage-1`.replace('1', i));
      if (stepEl) {
        stepEl.classList.remove('active', 'stage-danger');
      }
    }

    const tierNum = res.disease_stage.tier.replace('stage-', '');
    const activeStep = document.getElementById(`step-stage-${tierNum}`);
    if (activeStep) {
      activeStep.classList.add('active');
      if (parseInt(tierNum) >= 3) {
        activeStep.classList.add('stage-danger');
      }
    }

    // Render Trajectory Grid
    if (trajectoryGrid && res.progression_trajectory) {
      trajectoryGrid.innerHTML = '';
      res.progression_trajectory.forEach(node => {
        const nodeDiv = document.createElement('div');
        nodeDiv.className = 'traj-node';
        
        const isHigh = node.ad_risk_pct >= 65;
        const isMid = node.ad_risk_pct >= 35 && node.ad_risk_pct < 65;
        const colorClass = isHigh ? 'text-danger' : (isMid ? 'text-warning' : 'text-success');
        const fillGradient = isHigh
          ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
          : (isMid ? 'linear-gradient(90deg, #06b6d4, #f59e0b)' : 'linear-gradient(90deg, #10b981, #06b6d4)');

        nodeDiv.innerHTML = `
          <span class="t-lbl">${node.label}</span>
          <div class="t-val ${colorClass}">${node.ad_risk_pct}%</div>
          <div class="traj-bar">
            <div class="traj-fill" style="width: ${Math.max(node.ad_risk_pct, 6)}%; background: ${fillGradient}"></div>
          </div>
          <small style="font-size: 0.68rem; color: var(--text-dim); margin-top: 0.25rem; display: block;">
            Preserved: ${node.healthy_pct}%
          </small>
        `;
        trajectoryGrid.appendChild(nodeDiv);
      });
    }
  }

  // =========================================================================
  // 9. Patient-Specific Local Feature Attribution (SHAP Waterfall)
  // =========================================================================
  function renderLocalShapAttributions(attributions) {
    if (!localShapList) return;
    localShapList.innerHTML = '';

    if (!attributions || attributions.length === 0) {
      localShapList.innerHTML = `<p class="text-muted" style="font-size: 0.85rem; padding: 0.5rem 0;">Local SHAP breakdown is computing or unavailable for this instance.</p>`;
      return;
    }

    attributions.forEach(item => {
      const isRisk = item.direction === 'risk';
      const pillClass = isRisk ? 'risk' : 'protective';
      const pillSymbol = isRisk ? '+ ' : '- ';
      const fillGradient = isRisk
        ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
        : 'linear-gradient(90deg, #10b981, #06b6d4)';

      const row = document.createElement('div');
      row.className = 'shap-row-item';
      row.innerHTML = `
        <div class="shap-row-meta">
          <div class="shap-feat-name">
            <strong>${item.feature}</strong> = <span>${typeof item.value === 'number' ? (Number.isInteger(item.value) ? item.value : item.value.toFixed(1)) : item.value}</span>
          </div>
          <span class="shap-impact-pill ${pillClass}">${pillSymbol}${item.impact_pct}% Impact</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${Math.max(item.impact_pct, 5)}%; background: ${fillGradient}"></div>
        </div>
        <div class="shap-insight-text">${item.insight}</div>
      `;
      localShapList.appendChild(row);
    });
  }

  // =========================================================================
  // 10. Interactive 'What-If' Intervention Simulator
  // =========================================================================
  if (btnToggleIntervention && interventionCard) {
    btnToggleIntervention.addEventListener('click', () => {
      interventionCard.classList.toggle('hidden');
      if (!interventionCard.classList.contains('hidden')) {
        interventionCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  }

  if (btnCloseSim && interventionCard) {
    btnCloseSim.addEventListener('click', () => {
      interventionCard.classList.add('hidden');
    });
  }

  function initInterventionSimulator(baseRiskPct) {
    if (simFunctional) simFunctional.value = 0;
    if (simAdl) simAdl.value = 0;
    if (simCsf) simCsf.value = 0;
    if (simMmse) simMmse.value = 0;

    if (lblSimFunc) lblSimFunc.textContent = '+0.0';
    if (lblSimAdl) lblSimAdl.textContent = '+0.0';
    if (lblSimCsf) lblSimCsf.textContent = '+0 pg/mL';
    if (lblSimMmse) lblSimMmse.textContent = '+0.0';

    if (simBaseRisk) simBaseRisk.textContent = `${baseRiskPct}%`;
    if (simPostRisk) simPostRisk.textContent = `${baseRiskPct}%`;
    if (simReductionBadge) {
      simReductionBadge.textContent = 'Adjust sliders to simulate therapeutic risk reduction';
      simReductionBadge.style.opacity = '0.7';
    }
  }

  function setupSimulatorListeners() {
    const triggerSimulation = () => {
      if (!currentSimPatient) return;
      clearTimeout(simDebounceTimer);
      simDebounceTimer = setTimeout(runSimulateIntervention, 120);
    };

    if (simFunctional) {
      simFunctional.addEventListener('input', (e) => {
        if (lblSimFunc) lblSimFunc.textContent = `+${parseFloat(e.target.value).toFixed(2)}`;
        triggerSimulation();
      });
    }

    if (simAdl) {
      simAdl.addEventListener('input', (e) => {
        if (lblSimAdl) lblSimAdl.textContent = `+${parseFloat(e.target.value).toFixed(2)}`;
        triggerSimulation();
      });
    }

    if (simCsf) {
      simCsf.addEventListener('input', (e) => {
        if (lblSimCsf) lblSimCsf.textContent = `+${e.target.value} pg/mL`;
        triggerSimulation();
      });
    }

    if (simMmse) {
      simMmse.addEventListener('input', (e) => {
        if (lblSimMmse) lblSimMmse.textContent = `+${parseFloat(e.target.value).toFixed(1)}`;
        triggerSimulation();
      });
    }
  }
  setupSimulatorListeners();

  async function runSimulateIntervention() {
    if (!currentSimPatient) return;

    const payload = {
      patient: currentSimPatient,
      delta_functional: parseFloat(simFunctional.value || 0),
      delta_adl: parseFloat(simAdl.value || 0),
      delta_csf: parseFloat(simCsf.value || 0),
      delta_mmse: parseFloat(simMmse.value || 0)
    };

    try {
      const resp = await fetch('/api/simulate_intervention', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (data.status === 'success') {
        if (simBaseRisk) simBaseRisk.textContent = `${data.baseline_risk_pct}%`;
        if (simPostRisk) simPostRisk.textContent = `${data.post_risk_pct}%`;
        if (simReductionBadge) {
          simReductionBadge.style.opacity = '1';
          if (data.risk_reduction_pct > 0) {
            simReductionBadge.textContent = `-${data.risk_reduction_pct}% Absolute Risk Reduction (${data.relative_improvement_pct}% Relative)`;
          } else {
            simReductionBadge.textContent = 'Baseline level (no intervention applied yet)';
          }
        }
      }
    } catch (err) {
      console.error('Simulation failed:', err);
    }
  }

  // =========================================================================
  // 11. Printable Clinical Diagnostic Report Modal
  // =========================================================================
  if (btnOpenReport && reportModal) {
    btnOpenReport.addEventListener('click', () => {
      if (!latestPredictionResult || !lastPayload) {
        alert('Please evaluate a patient first before exporting a diagnostic report.');
        return;
      }
      populateClinicalReport(latestPredictionResult, lastPayload);
      reportModal.classList.remove('hidden');
    });
  }

  if (btnCloseReport && reportModal) {
    btnCloseReport.addEventListener('click', () => {
      reportModal.classList.add('hidden');
    });
  }

  if (reportModal) {
    reportModal.addEventListener('click', (e) => {
      if (e.target === reportModal) {
        reportModal.classList.add('hidden');
      }
    });
  }

  if (btnPrintReport) {
    btnPrintReport.addEventListener('click', () => {
      window.print();
    });
  }

  function populateClinicalReport(res, patient) {
    const isAD = res.prediction === 1;
    const adProbPct = (res.ensemble_probability * 100).toFixed(1);
    const confPct = res.ensemble_confidence !== undefined ? res.ensemble_confidence.toFixed(1) : adProbPct;

    // Header Meta
    const repRef = document.getElementById('rep-ref');
    const repDate = document.getElementById('rep-date');
    const repEngine = document.getElementById('rep-engine');
    if (repRef) repRef.textContent = `NVA-2026-${Math.floor(1000 + Math.random() * 9000)}`;
    if (repDate) repDate.textContent = new Date().toLocaleDateString('en-US', {
      year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
    if (repEngine) repEngine.textContent = res.ensemble_name;

    // Summary Banner
    const repVerdict = document.getElementById('rep-verdict');
    const repSubtext = document.getElementById('rep-subtext');
    const repConfScore = document.getElementById('rep-conf-score');
    const repRiskBadge = document.getElementById('rep-risk-badge');

    if (repVerdict) repVerdict.textContent = isAD ? "Alzheimer's Disease Detected (Positive Case)" : "Non-Alzheimer (Cognitively Preserved)";
    if (repSubtext) repSubtext.textContent = isAD
      ? "Diagnostic indicators and biological biomarkers exhibit characteristic Alzheimer's pathological patterns."
      : "Clinical metrics and biological biomarker concentrations remain within the normative cognitive range.";
    if (repConfScore) repConfScore.textContent = `${confPct}%`;
    if (repRiskBadge) repRiskBadge.textContent = `${res.risk_level} Risk Category`;

    // Biomarkers Table
    const tbody = document.getElementById('rep-biomarkers-tbody');
    if (tbody) {
      tbody.innerHTML = '';
      const rows = [
        { name: 'Mini-Mental State Exam (MMSE)', val: `${patient.MMSE} / 30`, ref: '16.0 - 22.0', status: patient.MMSE < 13 ? 'Severely Depressed (Impaired)' : (patient.MMSE < 16 ? 'Borderline Mild Impairment' : 'Normative Cohort Range') },
        { name: 'Functional Assessment', val: `${patient.FunctionalAssessment} / 10`, ref: '5.5 - 8.0', status: patient.FunctionalAssessment < 4.0 ? 'Impaired Autonomy' : 'Preserved Independence' },
        { name: 'Activities of Daily Living (ADL)', val: `${patient.ADL} / 10`, ref: '5.5 - 8.0', status: patient.ADL < 4.0 ? 'Impaired Daily Living' : 'Independent Functioning' },
        { name: 'Patient Age', val: `${patient.Age} Years`, ref: '60 - 90 Years', status: 'Demographic Baseline' },
        { name: 'Memory Complaints', val: patient.MemoryComplaints == 1 ? 'Present' : 'Absent', ref: '0 (Absent)', status: patient.MemoryComplaints == 1 ? 'Subjective Memory Deficit' : 'No Subjective Decline' },
        { name: 'Episodes of Disorientation / Confusion', val: patient.Confusion == 1 ? 'Present' : 'Absent', ref: '0 (Absent)', status: patient.Confusion == 1 ? 'Temporal/Spatial Disorientation' : 'Oriented' },
        { name: 'Behavioral / Neuropsychiatric Symptoms', val: patient.BehavioralProblems == 1 ? 'Present' : 'Absent', ref: '0 (Absent)', status: patient.BehavioralProblems == 1 ? 'Neuropsychiatric Manifestation' : 'Normal Affect' },
        { name: 'Hippocampal Volume (MRI Volumetric)', val: `${patient.Hippocampal_Volume} mm³`, ref: '> 3600 mm³', status: patient.Hippocampal_Volume < 3300 ? 'Significant Medial Temporal Atrophy' : 'Preserved Parenchyma' },
        { name: 'MRI / PET Imaging Uptake Score', val: `${patient.MRI_PET_Imaging_Scores} SUVr`, ref: '< 30.0 SUVr', status: patient.MRI_PET_Imaging_Scores > 45 ? 'Elevated Cortical Amyloid Tracer Uptake' : 'Low Tracer Binding' },
        { name: 'CSF Amyloid-Beta 42 (Aβ42)', val: `${patient['CSF Abeta42 Levels']} pg/mL`, ref: '> 750 pg/mL', status: patient['CSF Abeta42 Levels'] < 650 ? 'Pathologically Depleted (Cortical Plaque)' : 'Normal Solubility' },
        { name: 'APOE ε4 Allele Status', val: patient['APOE4 Gene Presence'] == 1 ? 'Carrier (ε4+)' : 'Non-carrier (ε4-)', ref: 'Non-carrier (ε2/ε3 or ε3/ε3)', status: patient['APOE4 Gene Presence'] == 1 ? 'Elevated Genetic Susceptibility' : 'Neutral Genetic Risk' }
      ];

      rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${r.name}</strong></td>
          <td>${r.val}</td>
          <td>${r.ref}</td>
          <td><span style="font-weight: 600; color: ${r.status.includes('Impaired') || r.status.includes('Depressed') || r.status.includes('Atrophy') || r.status.includes('Placque') || r.status.includes('Susceptibility') ? '#b91c1c' : '#15803d'}">${r.status}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Consensus Grid
    const repConsensus = document.getElementById('rep-consensus-grid');
    if (repConsensus && res.models_detail) {
      repConsensus.innerHTML = '';
      for (const [mName, mMeta] of Object.entries(res.models_detail)) {
        const cell = document.createElement('div');
        cell.className = 'rep-consensus-cell';
        const pRisk = (mMeta.probability_ad * 100).toFixed(1);
        const isMAd = mMeta.predicted_class === 1;
        cell.innerHTML = `
          <div class="rep-m-title">${mName}</div>
          <div class="rep-m-prob" style="color: ${isMAd ? '#dc2626' : '#16a34a'}">${isMAd ? 'AD ' + pRisk + '%' : 'Healthy ' + (100 - pRisk).toFixed(1) + '%'}</div>
          <div class="rep-m-sub">Weight: ${(mMeta.weight * 100).toFixed(1)}%</div>
        `;
        repConsensus.appendChild(cell);
      }
    }

    // Top SHAP attributions
    const repShap = document.getElementById('rep-shap-list');
    if (repShap && res.local_attributions) {
      repShap.innerHTML = '';
      res.local_attributions.slice(0, 5).forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${item.feature}</strong> (${item.direction === 'risk' ? '🔴 Risk Factor' : '🟢 Protective'}): ${item.insight} [${item.impact_pct}% contribution]`;
        repShap.appendChild(li);
      });
    }

    // Clinical Action Recommendations
    const repActionBox = document.getElementById('rep-action-box');
    if (repActionBox) {
      if (isAD) {
        repActionBox.innerHTML = `
          <div style="background: #fef2f2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 6px;">
            <strong style="color: #991b1b; display: block; margin-bottom: 0.35rem;">Urgent Clinical Action Protocol:</strong>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.82rem; color: #7f1d1d; line-height: 1.6;">
              <li>Immediate referral to a Memory Disorders Clinic / Board-Certified Behavioral Neurologist.</li>
              <li>Schedule high-resolution 3T volumetric structural MRI and diagnostic Tau-PET imaging.</li>
              <li>Evaluate eligibility for monoclonal anti-amyloid antibody therapy (e.g. Lecanemab/Donanemab) with baseline ARIA risk screening.</li>
              <li>Implement formal neuropsychological cognitive rehabilitation and safety measures for activities of daily living.</li>
            </ul>
          </div>
        `;
      } else {
        repActionBox.innerHTML = `
          <div style="background: #f0fdf4; border-left: 4px solid #22c55e; padding: 1rem; border-radius: 6px;">
            <strong style="color: #166534; display: block; margin-bottom: 0.35rem;">Standard Monitoring Protocol:</strong>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.82rem; color: #14532d; line-height: 1.6;">
              <li>Patient demonstrates preserved cognitive function and normative biomarker stability.</li>
              <li>Follow-up cognitive screening and MMSE re-evaluation recommended in 12 months.</li>
              <li>Encourage adherence to the Mediterranean-DASH Intervention for Neurodegenerative Delay (MIND) diet and aerobic exercise.</li>
            </ul>
          </div>
        `;
      }
    }
  }

  // =========================================================================
  // 12. Cohort Batch Screening & Hospital Triage Dashboard
  // =========================================================================
  if (btnLoadDemoCohort) {
    btnLoadDemoCohort.addEventListener('click', async () => {
      btnLoadDemoCohort.disabled = true;
      btnLoadDemoCohort.innerHTML = `<span class="btn-spinner" style="display:inline-block; vertical-align:middle; width:14px; height:14px; margin-right:6px;"></span> Querying 25 Cohort Patients...`;
      try {
        const demoResp = await fetch('/api/demo_cohort');
        const demoData = await demoResp.json();
        if (demoData.status === 'success' && demoData.patients) {
          const screenResp = await fetch('/api/batch_screen', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patients: demoData.patients })
          });
          const screenData = await screenResp.json();
          handleBatchScreenResults(screenData);
        }
      } catch (err) {
        console.error('Demo cohort load error:', err);
        alert('Batch screening error: ' + err.message);
      } finally {
        btnLoadDemoCohort.disabled = false;
        btnLoadDemoCohort.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          Load Demo Screening Cohort (25 Patients)
        `;
      }
    });
  }

  if (batchFileInput) {
    batchFileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const resp = await fetch('/api/batch_screen', {
          method: 'POST',
          body: formData
        });
        const screenData = await resp.json();
        if (screenData.status === 'success') {
          handleBatchScreenResults(screenData);
        } else {
          alert('Upload failed: ' + screenData.message);
        }
      } catch (err) {
        console.error('Batch CSV error:', err);
        alert('CSV Upload error: ' + err.message);
      } finally {
        batchFileInput.value = '';
      }
    });
  }

  function handleBatchScreenResults(data) {
    batchCohortData = data.patients || [];
    
    // Update KPI cards
    if (kpiTotal) kpiTotal.textContent = data.total_screened;
    if (kpiHigh) kpiHigh.textContent = data.summary.high_risk;
    if (kpiHighPct) kpiHighPct.textContent = `${data.summary.high_risk_pct}% Immediate Referral`;
    if (kpiBorderline) kpiBorderline.textContent = data.summary.borderline;
    if (kpiBorderlinePct) kpiBorderlinePct.textContent = `${data.summary.borderline_pct}% Monitoring Protocol`;
    if (kpiNormal) kpiNormal.textContent = data.summary.normal;
    if (kpiNormalPct) kpiNormalPct.textContent = `${data.summary.normal_pct}% Cognitively Intact`;
    if (kpiAvgRisk) kpiAvgRisk.textContent = `${data.summary.avg_risk_pct}%`;

    // Filter counts
    if (countAll) countAll.textContent = data.total_screened;
    if (countHigh) countHigh.textContent = data.summary.high_risk;
    if (countBorderline) countBorderline.textContent = data.summary.borderline;
    if (countNormal) countNormal.textContent = data.summary.normal;

    renderBatchTable();
  }

  // Filter Buttons
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentBatchFilter = btn.dataset.filter;
      renderBatchTable();
    });
  });

  // Search Input
  if (batchSearchInput) {
    batchSearchInput.addEventListener('input', (e) => {
      batchSearchQuery = e.target.value.toLowerCase().trim();
      renderBatchTable();
    });
  }

  function renderBatchTable() {
    if (!triageTableBody) return;
    triageTableBody.innerHTML = '';

    if (batchCohortData.length === 0) {
      triageTableBody.innerHTML = `
        <tr>
          <td colspan="8" class="text-center text-muted" style="padding: 2.5rem;">
            No cohort uploaded yet. Click <strong>"Load Demo Screening Cohort (25 Patients)"</strong> or upload a CSV file above.
          </td>
        </tr>
      `;
      return;
    }

    // Filter data
    const filtered = batchCohortData.filter(p => {
      const matchFilter = currentBatchFilter === 'all' || p.triage_tier === currentBatchFilter;
      const matchSearch = batchSearchQuery === '' || p.patient_id.toLowerCase().includes(batchSearchQuery);
      return matchFilter && matchSearch;
    });

    if (filtered.length === 0) {
      triageTableBody.innerHTML = `
        <tr>
          <td colspan="8" class="text-center text-muted" style="padding: 2.5rem;">
            No patients match the selected filter/search criteria.
          </td>
        </tr>
      `;
      return;
    }

    filtered.forEach(p => {
      const tr = document.createElement('tr');
      const isHigh = p.triage_tier === 'high';
      const isBorder = p.triage_tier === 'borderline';
      const riskClass = isHigh ? 'text-danger' : (isBorder ? 'text-warning' : 'text-success');

      tr.innerHTML = `
        <td><strong>${p.patient_id}</strong></td>
        <td>${p.age}</td>
        <td>${p.mmse.toFixed(1)}</td>
        <td>${p.functional.toFixed(1)}</td>
        <td><span class="badge ${p.apoe4 === 1 ? 'badge-accent' : 'badge-pill'}">${p.apoe4 === 1 ? 'ε4 Carrier' : 'Non-carrier'}</span></td>
        <td class="${riskClass}"><strong>${p.ad_risk_pct}%</strong></td>
        <td class="text-success">${p.healthy_pct}%</td>
        <td><span class="triage-pill ${p.triage_tier}">${p.triage}</span></td>
      `;
      triageTableBody.appendChild(tr);
    });
  }

  // Export Annotated CSV Download
  if (btnExportBatchCsv) {
    btnExportBatchCsv.addEventListener('click', () => {
      if (batchCohortData.length === 0) {
        alert('Please run cohort screening first before exporting data.');
        return;
      }

      const headers = ['PatientID', 'Age', 'MMSE', 'FunctionalAssessment', 'APOE4_Presence', 'AD_Risk_Pct', 'Healthy_Pct', 'Triage_Priority'];
      const rows = batchCohortData.map(p => [
        p.patient_id,
        p.age,
        p.mmse,
        p.functional,
        p.apoe4,
        p.ad_risk_pct,
        p.healthy_pct,
        `"${p.triage}"`
      ]);

      const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement('a');
      link.setAttribute('href', encodedUri);
      link.setAttribute('download', `neurovote_triage_cohort_${new Date().toISOString().slice(0, 10)}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  // =========================================================================
  // 13. Clinical Interpretation Generator
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
