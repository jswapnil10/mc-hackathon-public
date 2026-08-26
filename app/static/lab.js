const $ = (selector) => document.querySelector(selector);
const state = { run: null, roundIndex: 0, turnIndex: 0, status: null, atlas: null, benchmark: null, externalValidation: null };
const phases = ['pre_transaction', 'transaction', 'post_transaction'];
const phaseExplanations = {
  pre_transaction: 'Before money moves',
  transaction: 'While the payment is being requested',
  post_transaction: 'After money reaches the receiving side',
};
const actionClasses = new Set(['allow', 'monitor', 'step_up', 'hold', 'block']);

const money = (value) => new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
}).format(Number(value) || 0);
const pct = (value) => `${Math.round((Number(value) || 0) * 100)}%`;
const precisePct = (value, digits = 1) => `${((Number(value) || 0) * 100).toFixed(digits)}%`;
const integer = (value) => new Intl.NumberFormat('en-IN').format(Number(value) || 0);
const score = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(1) : '—';
const pretty = (value) => String(value || '—').replaceAll('_', ' ').replace(/\b\w/g, character => character.toUpperCase());
const shortModel = (value) => String(value || 'Open model').split('/').pop();
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;',
}[character]));
const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, Number(value) || 0));
const duration = (milliseconds) => {
  const totalSeconds = Math.max(0, Math.round((Number(milliseconds) || 0) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes ? `${minutes}m ${String(seconds).padStart(2, '0')}s` : `${seconds}s`;
};

function chips(items, tone = '') {
  if (!items || !items.length) return '<span class="chip">None recorded</span>';
  return items.map(item => `<span class="chip ${tone}">${escapeHtml(item)}</span>`).join('');
}

function metricBar(label, value) {
  const ratio = clamp(value, 0, 1);
  return `<div class="metric-line">
    <div class="metric-label"><span>${escapeHtml(label)}</span><b>${pct(ratio)}</b></div>
    <progress class="metric-track" max="100" value="${Math.round(ratio * 100)}">${Math.round(ratio * 100)}%</progress>
  </div>`;
}

function renderSubmissionProfile(profile) {
  const diversity = (profile && profile.diversity) || {};
  $('#criteriaDiversity').textContent = `${Number(diversity.attack_family_count) || 0} families`;
  $('#criteriaDiversityNote').textContent = `${Number(diversity.payment_surface_count) || 0} payment surfaces · ${Number(diversity.observable_event_type_count) || 0} event types · ${Number(diversity.legitimate_control_count) || 0} look-alikes.`;
}

function renderThreatAtlas() {
  const data = state.atlas;
  if (!data) return;
  const summary = data.summary || {};
  $('#atlasVectorCount').textContent = integer(summary.vector_count);
  $('#atlasSourceCount').textContent = integer(summary.source_count);
  $('#atlasRailCount').textContent = integer(summary.rail_count);

  const selectedCategory = $('#atlasCategory').value;
  const selectedRail = $('#atlasRail').value;
  const vectors = (data.vectors || []).filter(vector => (
    (selectedCategory === 'all' || vector.category === selectedCategory)
    && (selectedRail === 'all' || (vector.rails || []).includes(selectedRail))
  ));
  $('#atlasResultCount').textContent = `${vectors.length} of ${Number(summary.vector_count) || 0} vectors shown`;
  $('#atlasVectorGrid').innerHTML = vectors.map(vector => {
    const source = (vector.sources || [])[0] || {};
    const sourceUrl = String(source.url || '').startsWith('https://') ? source.url : '#';
    const sourceLabel = source.publisher || 'Source';
    return `<article class="atlas-vector ${vector.novel_holdout ? 'novel' : ''}">
      <header><span>${escapeHtml(vector.id)} · ${escapeHtml(vector.simulation_family)}</span>${vector.novel_holdout ? '<em>SEALED NOVEL TEST</em>' : ''}</header>
      <h3>${escapeHtml(vector.name)}</h3>
      <p>${escapeHtml((vector.defensive_observables || []).slice(0, 2).join(' · '))}</p>
      <div class="atlas-meta">
        ${(vector.rails || []).slice(0, 3).map(item => `<span>${escapeHtml(item)}</span>`).join('')}
        ${(vector.lifecycle_phases || []).map(item => `<span>${escapeHtml(pretty(item))}</span>`).join('')}
      </div>
      <footer><span>PLAUSIBILITY ${Number(vector.plausibility) || 0}/5 · NOVELTY ${Number(vector.novelty) || 0}/5 · ${escapeHtml(vector.research_confidence)} confidence</span><a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(sourceLabel)} ↗</a></footer>
    </article>`;
  }).join('') || '<p class="phase-message">No vectors match these filters.</p>';
}

async function loadThreatAtlas() {
  const response = await fetch('/api/v2/threat-atlas');
  if (!response.ok) throw new Error(`threat atlas endpoint returned ${response.status}`);
  state.atlas = await response.json();
  const categories = [...new Set((state.atlas.vectors || []).map(vector => vector.category))].sort();
  const rails = [...new Set((state.atlas.vectors || []).flatMap(vector => vector.rails || []))].sort();
  $('#atlasCategory').innerHTML = '<option value="all">All categories</option>' + categories.map(item => `<option value="${escapeHtml(item)}">${escapeHtml(pretty(item))}</option>`).join('');
  $('#atlasRail').innerHTML = '<option value="all">All rails</option>' + rails.map(item => `<option value="${escapeHtml(item)}">${escapeHtml(pretty(item))}</option>`).join('');
  renderThreatAtlas();
}

function renderBenchmark(data) {
  state.benchmark = data;
  if (!data) {
    $('#foundryState').textContent = 'No population build is available yet. Run the Scenario Foundry.';
    return;
  }
  const dataset = data.dataset || {};
  const fidelity = data.fidelity || {};
  const quality = data.data_quality || {};
  const defense = data.defense || {};
  const metrics = (defense.metrics || {}).combined_hidden_test || {};
  const novel = (defense.metrics || {}).novel_vector_test || {};
  const confusion = metrics.confusion_matrix || {};

  $('#foundryQualityState').textContent = quality.status === 'passed' ? 'ALL QUALITY GATES PASSED' : 'QUALITY REVIEW NEEDED';
  $('#foundryEvents').textContent = integer(dataset.event_count);
  $('#foundryCampaigns').textContent = integer(dataset.scenario_count);
  $('#foundryVectors').textContent = integer(dataset.attack_vector_count);
  $('#foundryFidelity').textContent = `${score(fidelity.score)} / 100`;
  const splitCounts = dataset.split_counts || {};
  const maximumSplit = Math.max(1, ...Object.values(splitCounts).map(Number));
  $('#splitBars').innerHTML = Object.entries(splitCounts).map(([name, count]) => `<div class="split-row"><span>${escapeHtml(pretty(name))}</span><progress max="${maximumSplit}" value="${Number(count) || 0}">${Number(count) || 0}</progress><b>${integer(count)}</b></div>`).join('');
  $('#foundryState').textContent = `${integer(quality.row_count)} rows · ${Number(quality.checks?.length) || 0} checks · generated ${new Date(data.generated_at).toLocaleString()}.`;

  $('#benchmarkModel').textContent = `${pretty(defense.selected_model)} · validation-selected`;
  $('#benchmarkPrauc').textContent = precisePct(metrics.pr_auc, 2);
  $('#benchmarkNovelRecall').textContent = precisePct(novel.recall, 1);
  $('#benchmarkF1').textContent = precisePct(metrics.f1, 1);
  $('#benchmarkFpr').textContent = precisePct(metrics.hard_false_positive_rate ?? metrics.false_positive_rate, 2);
  $('#benchmarkThreshold').textContent = `Threshold ${Number(defense.threshold_selected_on_validation || 0).toFixed(3)}`;
  $('#matrixTn').textContent = integer(confusion.tn);
  $('#matrixFp').textContent = integer(confusion.fp);
  $('#matrixFn').textContent = integer(confusion.fn);
  $('#matrixTp').textContent = integer(confusion.tp);

  const lifecycle = defense.lifecycle_results || {};
  $('#benchmarkPhases').innerHTML = phases.map(phase => {
    const item = lifecycle[phase] || {};
    return `<div class="benchmark-phase"><div class="benchmark-phase-head"><span>${escapeHtml(pretty(phase))}</span><b>${precisePct(item.recall, 1)} fraud caught</b></div><progress max="100" value="${clamp((Number(item.recall) || 0) * 100, 0, 100)}">${precisePct(item.recall, 1)}</progress><small>${integer(item.events)} hidden-test events · ${precisePct(1 - (Number(item.false_positive_rate) || 0), 1)} legitimate users protected</small></div>`;
  }).join('');
  $('#familyResults').innerHTML = Object.entries(defense.family_results || {}).sort().map(([family, item]) => `<tr><td>${escapeHtml(family)}</td><td>${integer(item.events)}</td><td>${precisePct(item.recall, 1)}</td><td>${precisePct(item.value_weighted_recall, 1)}</td><td>${precisePct(item.mean_risk_score, 1)}</td></tr>`).join('');
  $('#benchmarkGenerated').textContent = `${integer(metrics.event_count)} sealed test events · ${integer((data.threat_atlas || {}).novel_holdout_vector_count)} novel vectors`;

  $('#criteriaDiversity').textContent = `${integer((data.threat_atlas || {}).vector_count)} vectors`;
  $('#criteriaDiversityNote').textContent = `${integer((data.threat_atlas || {}).attack_family_count)} families · ${integer((data.threat_atlas || {}).rail_count)} rails · ${integer((data.threat_atlas || {}).source_count)} authoritative sources.`;
  $('#criteriaFidelity').textContent = `${score(fidelity.score)} / 100`;
  $('#criteriaFidelityNote').textContent = `Measured against declared priors · ${quality.status === 'passed' ? 'all quality gates passed' : 'quality review required'}.`;
  $('#criteriaDetection').textContent = `${precisePct(metrics.f1, 1)} F1`;
  $('#criteriaDetectionNote').textContent = `${precisePct(novel.recall, 1)} novel-vector recall · ${precisePct(metrics.hard_false_positive_rate ?? metrics.false_positive_rate, 2)} hard false positives.`;
}

async function loadBenchmark() {
  const response = await fetch('/api/v2/benchmark');
  if (response.status === 404) {
    renderBenchmark(null);
    return;
  }
  if (!response.ok) throw new Error(`benchmark endpoint returned ${response.status}`);
  renderBenchmark(await response.json());
}

function renderExternalValidation(data) {
  state.externalValidation = data;
  if (!data) {
    $('#externalAssessment').textContent = 'NOT RUN LOCALLY';
    $('#externalAssessment').className = 'proof-badge validation-caveat';
    $('#externalDoesNotProve').textContent = 'Download the optional public dataset and run the external-validation command to create this evidence.';
    return;
  }
  const defense = data.defense || {};
  const metrics = defense.test_metrics || {};
  const confusion = metrics.confusion_matrix || {};
  const quality = data.data_quality || {};
  const remediation = quality.remediation || {};
  const dataset = data.dataset || {};
  const temporal = defense.temporal_slices || {};
  const early = temporal.early_test || {};
  const late = temporal.late_test || {};
  const review = (metrics.review_capacity || {})['top_0.1_percent'] || {};
  const worstDrift = ((data.robustness || {}).worst_drift_features || [])[0] || {};
  const supportedClaim = (data.claim_validation || []).find(item => item.status === 'supported');

  $('#externalAssessment').textContent = pretty(data.overall_assessment || 'share with caveats');
  $('#externalAssessment').className = 'proof-badge validation-caveat';
  $('#externalPrauc').textContent = precisePct(metrics.pr_auc, 1);
  $('#externalPraucCi').textContent = metrics.pr_auc_95ci
    ? `95% interval ${precisePct(metrics.pr_auc_95ci[0], 1)}–${precisePct(metrics.pr_auc_95ci[1], 1)}`
    : '95% interval not available';
  $('#externalPrecision').textContent = precisePct(metrics.precision, 1);
  $('#externalRecall').textContent = precisePct(metrics.recall, 1);
  $('#externalFpr').textContent = precisePct(metrics.false_positive_rate, 3);
  $('#externalTn').textContent = integer(confusion.tn);
  $('#externalFp').textContent = integer(confusion.fp);
  $('#externalFn').textContent = integer(confusion.fn);
  $('#externalTp').textContent = integer(confusion.tp);
  $('#externalDatasetSize').textContent = `${integer(metrics.event_count)} future events · ${integer(metrics.fraud_event_count)} fraud`;
  $('#externalReviewCapture').textContent = `${precisePct(review.recall, 1)} of fraud surfaced`;
  $('#externalReviewCount').textContent = `by reviewing the top ${integer(review.review_count)} scores (0.1% of payments)`;
  $('#externalTemporal').innerHTML = [
    ['Earlier half', early],
    ['Later half', late],
  ].map(([label, item]) => `<div class="temporal-row"><span>${escapeHtml(label)}</span><div><b>${precisePct(item.pr_auc, 1)} fraud-ranking quality</b><small>${precisePct(item.recall, 1)} fraud caught · ${precisePct(item.false_positive_rate, 3)} legitimate payments flagged</small></div></div>`).join('');
  $('#externalDriftNote').textContent = worstDrift.feature
    ? `${worstDrift.feature} changed the most between training and future payments (population stability index ${Number(worstDrift.psi).toFixed(2)}). The later decline means the defense should be monitored and retrained.`
    : 'No drift summary is available.';
  $('#externalSupports').textContent = supportedClaim
    ? `On future payments, fraud-ranking quality reached ${precisePct(metrics.pr_auc, 1)} while fraud made up only ${precisePct(metrics.fraud_prevalence, 3)} of the data.`
    : 'The independent test evidence is available in the downloadable validation report.';
  $('#externalDoesNotProve').textContent = (data.limitations || [])[0] || 'Institution-specific shadow scoring is still required.';
  $('#externalQuality').textContent = `${score(quality.score)} / 100 source-data quality · validated after cleanup`;
  $('#externalRemediation').textContent = `${integer(remediation.exact_duplicate_rows_removed)} exact duplicate rows were removed before chronological splitting. Published counts reconciled at ${integer(remediation.raw_row_count)} rows and ${integer(remediation.raw_fraud_count)} frauds; ${integer(remediation.modeled_row_count)} rows entered validation.`;
  if (String(dataset.source_url || '').startsWith('https://')) $('#externalSource').href = dataset.source_url;
}

async function loadExternalValidation() {
  const response = await fetch('/api/v2/external-validation');
  if (response.status === 404) {
    renderExternalValidation(null);
    return;
  }
  if (!response.ok) throw new Error(`external validation endpoint returned ${response.status}`);
  renderExternalValidation(await response.json());
}

function renderRoundEvaluation(evaluation, report, round = {}) {
  if (!evaluation || !Object.keys(evaluation).length) return;
  const fidelity = (evaluation && evaluation.fidelity) || {};
  const efficacy = (evaluation && evaluation.detection_efficacy) || {};
  const live = (evaluation && evaluation.live_feasibility) || {};
  $('#criteriaFidelity').textContent = `${score(fidelity.score)} / 100`;
  $('#criteriaFidelityNote').textContent = `${Number(fidelity.legitimate_look_alike_cases) || 0} legitimate look-alikes tested · ${fidelity.truth_boundary_clean ? 'Blue never saw the answer key' : 'answer-key boundary needs review'}.`;
  $('#criteriaDetection').textContent = `${pct(efficacy.value_prevented_ratio ?? report.value_prevented_ratio)} protected`;
  $('#criteriaDetectionNote').textContent = `${pct(efficacy.hard_false_positive_rate ?? report.hard_false_positive_rate)} of legitimate look-alikes wrongly stopped · ${score(efficacy.balanced_lifecycle_defense_score ?? report.balanced_lifecycle_defense_score)} overall defense score.`;
  $('#criteriaNovelty').textContent = 'Two-speed defense';
  const storedGuardedEvents = Number(efficacy.fast_guard_actionable_event_count);
  const guardedEvents = Number.isFinite(storedGuardedEvents)
    ? storedGuardedEvents
    : (((round.blue || {}).attack_turns || []).filter(turn => ['step_up', 'hold', 'block'].includes(((turn.risk_synthesis || {}).operational_minimum_action || (turn.risk_synthesis || {}).minimum_action))).length);
  const modelEscalations = Number(efficacy.fast_guard_intervention_count) || 0;
  $('#criteriaNoveltyNote').textContent = `${guardedEvents} events had an immediate minimum-safe action before GenAI finished · ${modelEscalations} weaker model action${modelEscalations === 1 ? '' : 's'} corrected.`;
  $('#criteriaLive').textContent = live.round_duration_ms == null
    ? `${pct(live.pre_model_fast_path_coverage)} fast path`
    : `${duration(live.round_duration_ms)} battle`;
  const callSummary = live.blue_model_call_count == null
    ? ''
    : `${Number(live.blue_model_call_count)} Blue calls · ${Number(live.model_calls_per_blue_event || 0).toFixed(1)} per event · `;
  $('#criteriaLiveNote').textContent = live.model_call_p95_ms == null
    ? `${callSummary}Inline guard is separate from asynchronous open-model reasoning.`
    : `${callSummary}95% of Qwen calls finished within ${duration(live.model_call_p95_ms)} · ${Number(live.case_parallelism) || 1} isolated cases can run in parallel.`;
}

function updateLatencyEstimate() {
  const rounds = Number($('#rounds').value) || 1;
  $('#latencyEstimate').textContent = rounds === 1
    ? 'One round: Red plans once, Blue reviews each reached event, and the Referee scores once. No adaptation replay is performed.'
    : `${rounds} rounds: after each non-final round, Blue tests a candidate defense and Red receives limited Referee feedback before the next battle.`;
}

function currentRound() {
  return state.run && state.run.rounds ? state.run.rounds[state.roundIndex] : null;
}

async function loadStatus() {
  const response = await fetch('/api/v2/status');
  if (!response.ok) throw new Error(`status endpoint returned ${response.status}`);
  const data = await response.json();
  state.status = data;
  renderSubmissionProfile(data.submission_profile);
  $('#systemMode').textContent = String(data.mode || 'open model arena').replaceAll('_', ' ');
  $('#redModel').textContent = data.models.red;
  $('#blueModel').textContent = data.models.blue_detector_active
    ? `${shortModel(data.models.blue)} + HistGB`
    : `${shortModel(data.models.blue)} · guard fallback`;
  $('#refereeModel').textContent = data.models.referee;
  $('#redModelPill').textContent = shortModel(data.models.red);
  $('#blueModelPill').textContent = data.models.blue_detector_active
    ? `${shortModel(data.models.blue)} + ML`
    : shortModel(data.models.blue);
  $('#attackFamily').innerHTML = data.attack_families.map(family => (
    `<option value="${escapeHtml(family.id)}">${escapeHtml(family.id)} · ${escapeHtml(family.name)}</option>`
  )).join('');
  updateLatencyEstimate();
  if (data.latest_run_available) await loadLatest();
}

async function loadLatest() {
  const response = await fetch('/api/v2/latest');
  if (!response.ok) return;
  const savedRun = await response.json();
  const latestRound = savedRun.rounds && savedRun.rounds[savedRun.rounds.length - 1];
  if (!savedRun.submission_profile || !latestRound || !latestRound.submission_evaluation) return;
  state.run = savedRun;
  state.roundIndex = Math.max(0, state.run.rounds.length - 1);
  state.turnIndex = 0;
  renderRun(false);
}

function renderTimeline(turns) {
  const safeTurns = Array.isArray(turns) ? turns : [];
  if (!safeTurns.length) {
    $('#eventTimeline').innerHTML = '<p class="phase-message">No Blue event decisions were recorded.</p>';
    renderSelectedTurn(0);
    return;
  }
  $('#eventTimeline').innerHTML = safeTurns.map((turn, index) => {
    const event = turn.event || {};
    const decision = turn.decision || {};
    const action = actionClasses.has(decision.action) ? decision.action : 'monitor';
    return `<button type="button" class="event" data-turn-index="${index}">
      <span class="event-index">${String(index + 1).padStart(2, '0')}</span>
      <span><span class="event-title">${escapeHtml(pretty(event.event_type))}</span><span class="event-phase">${escapeHtml(pretty(event.lifecycle_phase || 'phase not recorded'))}</span></span>
      <span class="action ${action}">${escapeHtml(pretty(decision.action))}</span>
    </button>`;
  }).join('');
  document.querySelectorAll('[data-turn-index]').forEach(button => {
    button.addEventListener('click', () => renderSelectedTurn(Number(button.dataset.turnIndex)));
  });
  renderSelectedTurn(state.turnIndex);
}

function renderSelectedTurn(index) {
  const round = currentRound();
  const turns = round && round.blue ? (round.blue.attack_turns || []) : [];
  if (!turns.length) {
    $('#previousEvent').disabled = true;
    $('#nextEvent').disabled = true;
    return;
  }
  state.turnIndex = clamp(index, 0, turns.length - 1);
  const turn = turns[state.turnIndex];
  const event = turn.event || {};
  const investigation = turn.investigation || {};
  const decision = turn.decision || {};
  const riskSynthesis = turn.risk_synthesis || {};
  const mlRisk = turn.ml_risk || null;
  const action = actionClasses.has(decision.action) ? decision.action : 'monitor';

  document.querySelectorAll('[data-turn-index]').forEach((button, buttonIndex) => {
    button.classList.toggle('active', buttonIndex === state.turnIndex);
  });
  $('#selectedPhase').textContent = pretty(event.lifecycle_phase || 'phase not recorded');
  $('#selectedSequence').textContent = `EVENT ${String(event.sequence || state.turnIndex + 1).padStart(2, '0')}`;
  $('#selectedEvent').textContent = pretty(event.event_type);
  $('#selectedAction').className = `action ${action}`;
  $('#selectedAction').textContent = pretty(decision.action);
  $('#selectedRisk').textContent = `${pretty(decision.risk_level)} risk`;
  $('#selectedConfidence').textContent = `${pct(decision.confidence)} confidence`;
  $('#selectedSummary').textContent = decision.decision_summary || 'No decision summary was recorded.';
  $('#selectedDelivery').textContent = `${pretty(event.source_system || 'synthetic event stream')} · ${pretty(event.decision_lane || 'decision lane not recorded')} · ${Number(event.latency_budget_ms) || '—'} ms budget`;
  $('#selectedDetector').textContent = mlRisk
    ? (mlRisk.above_threshold ? 'Above alert threshold' : 'Below alert threshold')
    : 'Detector unavailable';
  $('#selectedDetectorScore').textContent = mlRisk
    ? `${precisePct(mlRisk.cumulative_session_risk, 1)} cumulative risk`
    : 'Qwen + guard fallback';
  $('#selectedDetectorDetail').innerHTML = mlRisk
    ? chips([
      `Event score ${precisePct(mlRisk.per_event_risk, 1)}`,
      `Threshold ${precisePct(mlRisk.alert_threshold, 1)}`,
      `Model ${mlRisk.model_hash || 'local champion'}`,
    ], mlRisk.above_threshold ? 'red' : 'blue')
    : chips(['Train scripts/train_detector.py to activate'], 'guard');
  $('#selectedGuard').textContent = `${pretty(riskSynthesis.operational_minimum_action || riskSynthesis.minimum_action || 'allow')} minimum`;
  $('#selectedGuardScore').textContent = `${Number(riskSynthesis.observable_risk_score) || 0} observable risk points`;
  $('#selectedGuardSignals').innerHTML = chips((riskSynthesis.indicators || []).map(item => pretty(item.code)), 'guard');
  $('#selectedTools').innerHTML = chips((investigation.requested_tools || []).map(pretty), 'blue');
  $('#selectedReasons').innerHTML = chips((decision.reason_codes || []).map(pretty), 'blue');
  $('#selectedMitigation').textContent = decision.mitigation || 'No mitigation was recorded.';
  $('#previousEvent').disabled = state.turnIndex === 0;
  $('#nextEvent').disabled = state.turnIndex === turns.length - 1;
}

function renderLifecycle(report) {
  const metrics = report.lifecycle_metrics;
  const hasLifecycle = metrics && Object.keys(metrics).length;
  $('#balancedScore').textContent = score(report.balanced_lifecycle_defense_score ?? report.blue_score);
  $('#worstPhase').textContent = hasLifecycle ? pretty(report.worst_lifecycle_phase) : 'Run a new battle';
  $('#balanceGap').textContent = hasLifecycle ? `${score(report.lifecycle_balance_gap)} pts` : 'Not available';

  if (!hasLifecycle) {
    $('#phaseCards').innerHTML = `<article class="phase-card empty lifecycle-empty">
      <div class="phase-message">This saved battle predates lifecycle scoring.<br>Run a new battle to measure pre-transaction, transaction and post-transaction defense.</div>
    </article>`;
    return;
  }

  $('#phaseCards').innerHTML = phases.map((phase, index) => {
    const item = metrics[phase] || { status: 'not_in_scenario' };
    const title = pretty(phase);
    if (item.status === 'prevented_before_phase') {
      return `<article class="phase-card protected">
        <div class="phase-head"><div class="phase-name"><small>0${index + 1} · STOPPED EARLIER</small><h3>${escapeHtml(title)}</h3><span>${escapeHtml(phaseExplanations[phase])}</span></div><div class="phase-score" title="Blue prevented the attack from reaching this stage">✓</div></div>
        <div class="phase-message">Blue stopped the attack before this payment stage began.<br><b>${escapeHtml(pretty(item.first_actionable_event))}</b> was never reached.</div>
        <div class="phase-foot"><span>0 of ${Number(item.event_count) || 0} planned events reviewed</span><span>PROTECTED BY EARLIER ACTION</span></div>
      </article>`;
    }
    if (item.status !== 'reached') {
      return `<article class="phase-card empty">
        <div class="phase-head"><div class="phase-name"><small>0${index + 1} · NOT TESTED</small><h3>${escapeHtml(title)}</h3><span>${escapeHtml(phaseExplanations[phase])}</span></div><div class="phase-score">—</div></div>
        <div class="phase-message">This attack family did not exercise this lifecycle phase.</div>
      </article>`;
    }

    const phaseScore = Number(item.phase_score) || 0;
    const tone = phaseScore >= 75 ? 'good' : phaseScore >= 50 ? 'watch' : 'weak';
    const transition = item.transition_escape_rate == null
      ? 'LAST STAGE'
      : item.transition_escape_rate > 0
        ? 'ATTACK REACHED NEXT STAGE'
        : 'ATTACK STOPPED HERE';
    const response = item.response_time_seconds == null
      ? 'Blue did not take an alerting action in this stage'
      : `Blue responded ${Number(item.response_time_seconds)}s later in simulated time`;
    return `<article class="phase-card ${tone}">
      <div class="phase-head"><div class="phase-name"><small>0${index + 1} · ${item.opportunity_detected ? 'WARNING DETECTED' : 'WARNING MISSED'}</small><h3>${escapeHtml(title)}</h3><span>${escapeHtml(phaseExplanations[phase])}</span></div><div class="phase-score" title="Defense score for this payment stage, out of 100">${Math.round(phaseScore)}</div></div>
      <div class="phase-opportunity"><b>First chance to act: ${escapeHtml(pretty(item.first_actionable_event))}</b>${escapeHtml(response)}</div>
      ${metricBar('How quickly Blue reacted', item.response_score)}
      ${metricBar('Potential harm stopped', item.consequence_control_ratio)}
      ${metricBar('Legitimate users left unharmed', item.legitimate_safety_rate)}
      <div class="phase-foot"><span>${Number(item.evaluated_event_count) || 0} of ${Number(item.event_count) || 0} events reviewed</span><span>${transition}</span></div>
    </article>`;
  }).join('');
}

function renderImpact(report, scenario) {
  const potential = Number(report.total_value_at_risk_inr) || 0;
  const protectedValue = Number(report.value_prevented_inr) || 0;
  const realized = Number(report.realized_impact_inr ?? Math.max(0, potential - protectedValue)) || 0;
  const realizedRatio = Number(report.realized_impact_ratio ?? (potential ? realized / potential : 0)) || 0;
  const capability = Number(report.red_capability_score ?? report.red_score) || 0;
  const lifecycleCount = report.lifecycle_metrics
    ? Object.values(report.lifecycle_metrics).filter(item => item.status !== 'not_in_scenario').length
    : 0;

  $('#redCapability').textContent = score(capability);
  $('#redCapabilityNote').textContent = lifecycleCount
    ? `This attack exercised ${lifecycleCount} payment stage${lifecycleCount === 1 ? '' : 's'}. The score also rewards stealth and deeper event sequences.`
    : 'Measures how far the attack progressed, how quietly it moved, and how many stages it tested.';
  $('#realizedImpact').textContent = money(realized);
  $('#realizedImpactRatio').textContent = `${pct(realizedRatio)} of at-risk synthetic money was lost`;
  $('#attackPotential').textContent = money(potential);
  $('#valueProtected').textContent = money(protectedValue);
  $('#valueAtRisk').textContent = `of ${money(potential)} at risk`;
  $('#impactAmount').textContent = money(realized);

  let verdict = 'Low-to-moderate capability and contained impact.';
  if (capability >= 70 && realizedRatio >= .4) verdict = 'High-capability, high-impact: a critical control gap.';
  else if (capability >= 70 && realizedRatio < .25) verdict = 'High-capability attack, low realized impact: defended well.';
  else if (capability < 70 && realizedRatio >= .4) verdict = 'Simpler attack, high impact: a structural control gap.';
  $('#impactVerdict').textContent = verdict;

  const bubble = $('#impactBubble');
  const capabilityBand = capability >= 67 ? 'cap-high' : capability >= 34 ? 'cap-mid' : 'cap-low';
  const impactBand = realizedRatio >= .67 ? 'impact-high' : realizedRatio >= .34 ? 'impact-mid' : 'impact-low';
  const valueBand = potential >= 50000 ? 'value-high' : potential >= 10000 ? 'value-mid' : 'value-low';
  bubble.className = `impact-bubble ${capabilityBand} ${impactBand} ${valueBand}`;
  bubble.textContent = scenario.attack_family || 'RED';
}

function renderLearning(round) {
  const learning = round.blue_adaptation;
  if (!learning) {
    const active = round.blue.active_playbook || {};
    const activeTools = (active.preferred_tools || []).map(pretty);
    $('#blueLearning').innerHTML = `<small>BLUE LEARNING LOOP · ACTIVE PLAYBOOK V${Number(active.version) || 1}</small>
      <p>${active.version > 1 ? 'This defense was promoted by the prior round replay.' : 'Final round: no later battle needs a new candidate.'}</p>
      <div class="chip-list">${chips(activeTools, 'blue')}</div>`;
    return;
  }
  const candidate = (learning.strategy && learning.strategy.proposed_playbook) || {};
  const replayReport = learning.replay && learning.replay.referee ? learning.replay.referee : {};
  const replayScore = replayReport.balanced_lifecycle_defense_score ?? replayReport.blue_score;
  const weakest = replayReport.worst_lifecycle_phase ? ` · weakest phase: ${pretty(replayReport.worst_lifecycle_phase)}` : '';
  $('#blueLearning').innerHTML = `<small>BLUE LEARNING LOOP · ${learning.promoted ? 'PROMOTED' : 'REJECTED'}</small>
    <p>${escapeHtml(learning.promotion_reason || 'Replay completed.')}</p>
    <div class="chip-list">${chips((candidate.preferred_tools || []).map(pretty), 'blue')}</div>
    <p>Replay lifecycle score: ${score(replayScore)}${escapeHtml(weakest)}</p>`;
}

function renderRound(index) {
  state.roundIndex = clamp(index, 0, state.run.rounds.length - 1);
  state.turnIndex = 0;
  const round = currentRound();
  const report = round.referee || {};
  const plan = round.red.plan || {};
  const scenario = round.red.scenario || {};
  const feedback = round.feedback_released_to_red || {};
  renderRoundEvaluation(round.submission_evaluation || {}, report, round);

  renderLifecycle(report);
  renderImpact(report, scenario);
  $('#outcome').textContent = pretty(report.outcome);
  $('#detectionTime').textContent = report.time_to_detect_seconds == null
    ? 'No warning was detected'
    : `warning detected at ${report.time_to_detect_seconds}s in simulated time`;
  $('#falsePositives').textContent = pct(report.hard_false_positive_rate);
  $('#strategyFamily').textContent = `${scenario.attack_family || '—'} · ${pretty(scenario.difficulty)} · ${pretty(plan.target_lifecycle_phase || 'lifecycle not recorded')}`;
  $('#strategyObjective').textContent = plan.objective || 'No objective was recorded.';
  $('#strategyFocus').innerHTML = chips((plan.focus_stage_ids || plan.stage_emphasis || []).map(pretty), 'red');
  $('#adaptationGoal').textContent = plan.adaptation_goal || 'Not recorded in this earlier run.';
  $('#adaptationHypothesis').textContent = plan.adaptation_hypothesis || 'No adaptation hypothesis was recorded.';
  $('#parameterChanges').innerHTML = chips((plan.parameter_changes || []).map(item => `${pretty(item.parameter)} → ${item.value}`), 'red');
  renderTimeline((round.blue && round.blue.attack_turns) || []);
  renderLearning(round);

  $('#feedbackOutcome').textContent = pretty(feedback.outcome);
  $('#feedbackStage').textContent = pretty(feedback.detected_stage_id);
  $('#feedbackReasons').innerHTML = chips((feedback.coarse_reason_categories || []).map(pretty), 'blue');
  $('#feedbackRatio').textContent = pct(feedback.value_prevented_ratio);
  document.querySelectorAll('.round-tabs button').forEach((button, buttonIndex) => {
    button.classList.toggle('active', buttonIndex === state.roundIndex);
  });
}

function renderRun(scrollToResults = true) {
  $('#results').classList.remove('hidden');
  $('#runTitle').textContent = state.run.run_id;
  $('#redModelPill').textContent = shortModel(state.run.model_configuration.red);
  const detector = state.run.model_configuration.blue_detector || {};
  $('#blueModelPill').textContent = detector.active
    ? `${shortModel(state.run.model_configuration.blue)} + ML`
    : `${shortModel(state.run.model_configuration.blue)} · fallback`;
  $('#roundTabs').innerHTML = state.run.rounds.map((_, index) => `<button type="button" data-round="${index}">Round ${index + 1}</button>`).join('');
  document.querySelectorAll('[data-round]').forEach(button => {
    button.addEventListener('click', () => renderRound(Number(button.dataset.round)));
  });
  renderRound(state.roundIndex);
  if (scrollToResults) $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function runProgressStages(totalRounds) {
  const stages = [
    { key: 'red_planning', label: '1 · Red plans' },
    { key: 'simulation', label: '2 · Arena builds events' },
    { key: 'blue_investigation', label: '3 · Blue decides' },
    { key: 'referee_scoring', label: '4 · Referee scores' },
  ];
  if (Number(totalRounds) > 1) stages.push({ key: 'blue_adaptation', label: '5 · Blue tests an update' });
  stages.push({ key: 'completed', label: `${stages.length + 1} · Report ready` });
  return stages;
}

function renderRunProgress(progress) {
  const totalRounds = Number(progress.total_rounds) || Number($('#rounds').value) || 1;
  const roundNumber = clamp(progress.round_number || 1, 1, totalRounds);
  const steps = runProgressStages(totalRounds);
  const normalizedStage = progress.stage === 'blue_replay' ? 'blue_adaptation' : progress.stage;
  let currentIndex = steps.findIndex(item => item.key === normalizedStage);
  const intermediateRoundComplete = progress.stage === 'round_complete' && roundNumber < totalRounds;
  if (progress.stage === 'completed' || (progress.stage === 'round_complete' && !intermediateRoundComplete)) {
    currentIndex = steps.length - 1;
  } else if (intermediateRoundComplete) {
    currentIndex = steps.length - 1;
  }

  $('#runRoundLabel').textContent = `ROUND ${roundNumber} OF ${totalRounds} · ${progress.status === 'error' ? 'STOPPED' : 'LIVE'}`;
  $('#runStateTitle').textContent = progress.headline || 'Running the synthetic battle';
  $('#runStateText').textContent = progress.detail || 'Waiting for a verified stage update.';
  $('#runProgressSteps').classList.toggle('has-adaptation', totalRounds > 1);
  $('#runProgressSteps').innerHTML = steps.map((step, index) => {
    const stateClass = progress.stage === 'completed' || index < currentIndex
      ? 'done'
      : index === currentIndex && !intermediateRoundComplete
        ? 'current'
        : '';
    return `<li class="${stateClass}">${escapeHtml(step.label)}</li>`;
  }).join('');

  const completedEvents = Number(progress.completed_events);
  const totalCapacity = Number(progress.total_event_capacity);
  if (Number.isFinite(completedEvents) && Number.isFinite(totalCapacity)) {
    $('#runProgressCount').textContent = `${integer(completedEvents)} event decisions completed · up to ${integer(totalCapacity)} events across ${integer(progress.case_count)} isolated cases`;
  } else if (progress.stage === 'simulation') {
    $('#runProgressCount').textContent = `${integer(progress.attack_event_count)} attack events · ${integer(progress.control_case_count)} legitimate look-alike cases · answer key still sealed`;
  } else if (progress.stage === 'completed') {
    $('#runProgressCount').textContent = `Completed in ${duration(progress.duration_ms)} · opening the scored report`;
  } else {
    $('#runProgressCount').textContent = 'Verified live update from the server orchestration pipeline.';
  }
}

function startProgressPolling(progressId) {
  let stopped = false;
  let timer = null;
  const poll = async () => {
    if (stopped) return;
    try {
      const response = await fetch(`/api/v2/run-progress/${encodeURIComponent(progressId)}`, { cache: 'no-store' });
      if (response.ok) renderRunProgress(await response.json());
    } catch (_error) {
      // The battle request remains authoritative; a missed telemetry poll must not stop it.
    }
    if (!stopped) timer = window.setTimeout(poll, 750);
  };
  poll();
  return () => {
    stopped = true;
    if (timer) window.clearTimeout(timer);
  };
}

$('#previousEvent').addEventListener('click', () => renderSelectedTurn(state.turnIndex - 1));
$('#nextEvent').addEventListener('click', () => renderSelectedTurn(state.turnIndex + 1));
$('#rounds').addEventListener('change', updateLatencyEstimate);
$('#atlasCategory').addEventListener('change', renderThreatAtlas);
$('#atlasRail').addEventListener('change', renderThreatAtlas);

$('#runFoundry').addEventListener('click', async () => {
  const button = $('#runFoundry');
  button.disabled = true;
  $('#foundryState').textContent = 'Generating population, opening sealed splits and benchmarking candidate defenses…';
  try {
    const response = await fetch('/api/v2/benchmark/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ variants_per_vector: 6, legitimate_events: 2400, seed: 20260824 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `benchmark returned ${response.status}`);
    renderBenchmark(payload);
    $('#defenseBenchmark').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    $('#foundryState').textContent = `Benchmark could not complete: ${error.message}`;
  } finally {
    button.disabled = false;
  }
});

$('#runForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('#errorState').classList.add('hidden');
  $('#runState').classList.remove('hidden');
  $('#runButton').disabled = true;
  const progressId = globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function'
    ? globalThis.crypto.randomUUID()
    : `browser-${Date.now()}-${Math.floor(Math.random() * 1000000)}`;
  renderRunProgress({
    status: 'running',
    stage: 'preparing',
    headline: 'Preparing the synthetic payment arena',
    detail: 'The browser sent the selected battle settings to the server.',
    round_number: 1,
    total_rounds: Number($('#rounds').value),
  });
  const stopProgressPolling = startProgressPolling(progressId);
  const startedAt = Date.now();
  $('#runElapsed').textContent = 'Elapsed 0s';
  const elapsedTimer = setInterval(() => {
    $('#runElapsed').textContent = `Elapsed ${duration(Date.now() - startedAt)}`;
  }, 1000);
  try {
    const response = await fetch('/api/v2/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        attack_family: $('#attackFamily').value,
        difficulty: $('#difficulty').value,
        rounds: Number($('#rounds').value),
        seed: Date.now() % 100000000,
        progress_id: progressId,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error([payload.error, payload.hint].filter(Boolean).join(' '));
    state.run = payload;
    state.roundIndex = payload.rounds.length - 1;
    state.turnIndex = 0;
    renderRun(true);
  } catch (error) {
    $('#errorState').textContent = error.message;
    $('#errorState').classList.remove('hidden');
  } finally {
    stopProgressPolling();
    clearInterval(elapsedTimer);
    $('#runState').classList.add('hidden');
    $('#runButton').disabled = false;
  }
});

async function initialize() {
  await loadStatus();
  await Promise.all([loadThreatAtlas(), loadBenchmark(), loadExternalValidation()]);
}

initialize().catch(error => {
  $('#errorState').textContent = `Could not load MasterGuard AI: ${error.message}`;
  $('#errorState').classList.remove('hidden');
});
