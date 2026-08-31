const $ = (selector) => document.querySelector(selector);
const state = { atlas: null, benchmark: null };
const phases = ['pre_transaction', 'transaction', 'post_transaction'];

const money = (value) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(Number(value) || 0);
const pct = (value) => `${Math.round((Number(value) || 0) * 100)}%`;
const precisePct = (value, digits = 1) => `${((Number(value) || 0) * 100).toFixed(digits)}%`;
const integer = (value) => new Intl.NumberFormat('en-IN').format(Number(value) || 0);
const score = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(1) : '—';
const pretty = (value) => String(value || '—').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, Number(value) || 0));

const set = (id, val) => { const el = $(id); if (el) el.textContent = val; };
const setHtml = (id, val) => { const el = $(id); if (el) el.innerHTML = val; };

function chips(items, tone = '') {
  if (!items || !items.length) return '<span class="chip">None recorded</span>';
  return items.map(item => `<span class="chip ${tone}">${escapeHtml(item)}</span>`).join('');
}
function infoIcon(label, definition) {
  return `<button type="button" class="metric-info" aria-label="${escapeHtml(label)} definition" data-tooltip="${escapeHtml(definition)}">i</button>`;
}
function metricTitle(label, definition) {
  return `<span class="metric-title">${escapeHtml(label)} ${infoIcon(label, definition)}</span>`;
}

const metricDefinitions = {
  precision: 'Of attack and legitimate comparison cases alerted in this stage, the share that were actually the attack.',
  recall: 'Of attack opportunities available in this stage, the share Blue alerted before the stage ended.',
  f1: 'A balance of precision and recall. It is high only when alerts are accurate and the attack is caught.',
};

// ---- Threat Atlas ----

function renderThreatAtlas() {
  const data = state.atlas;
  if (!data) return;
  const summary = data.summary || {};
  set('#atlasVectorCount', integer(summary.vector_count));
  set('#atlasSourceCount', integer(summary.source_count));
  set('#atlasRailCount', integer(summary.rail_count));

  const cat = $('#atlasCategory');
  const rail = $('#atlasRail');
  const selectedCategory = cat ? cat.value : 'all';
  const selectedRail = rail ? rail.value : 'all';
  const vectors = (data.vectors || []).filter(v => (
    (selectedCategory === 'all' || v.category === selectedCategory)
    && (selectedRail === 'all' || (v.rails || []).includes(selectedRail))
  ));
  set('#atlasResultCount', `${vectors.length} of ${Number(summary.vector_count) || 0} vectors shown`);
  setHtml('#atlasVectorGrid', vectors.map(vector => {
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
  }).join('') || '<p class="phase-message">No vectors match these filters.</p>');
}

async function loadThreatAtlas() {
  const response = await fetch('/api/v2/threat-atlas');
  if (!response.ok) throw new Error(`threat atlas endpoint returned ${response.status}`);
  state.atlas = await response.json();
  const categories = [...new Set((state.atlas.vectors || []).map(v => v.category))].sort();
  const rails = [...new Set((state.atlas.vectors || []).flatMap(v => v.rails || []))].sort();
  const cat = $('#atlasCategory');
  const rail = $('#atlasRail');
  if (cat) cat.innerHTML = '<option value="all">All categories</option>' + categories.map(item => `<option value="${escapeHtml(item)}">${escapeHtml(pretty(item))}</option>`).join('');
  if (rail) rail.innerHTML = '<option value="all">All rails</option>' + rails.map(item => `<option value="${escapeHtml(item)}">${escapeHtml(pretty(item))}</option>`).join('');
  renderThreatAtlas();
}

// ---- Defense Benchmark ----

function renderBenchmark(data) {
  state.benchmark = data;
  if (!data) {
    set('#foundryState', 'No population build is available yet. Run the Scenario Foundry.');
    return;
  }
  const dataset = data.dataset || {};
  const fidelity = data.fidelity || {};
  const quality = data.data_quality || {};
  const defense = data.defense || {};
  const metrics = (defense.metrics || {}).combined_hidden_test || {};
  const novel = (defense.metrics || {}).novel_vector_test || {};
  const confusion = metrics.confusion_matrix || {};
  const checks = quality.checks || [];
  const passedChecks = checks.filter(item => item.passed).length;

  set('#foundryQualityState', quality.status === 'passed'
    ? `${passedChecks} / ${checks.length} DATA CHECKS PASSED`
    : `${passedChecks} / ${checks.length} DATA CHECKS PASSED · REVIEW NEEDED`);
  set('#foundryEvents', integer(dataset.event_count));
  set('#foundryCampaigns', integer(dataset.scenario_count));
  set('#foundryVectors', integer(dataset.attack_vector_count));
  set('#foundryFidelity', `${score(fidelity.score)} / 100`);
  const splitCounts = dataset.split_counts || {};
  const maximumSplit = Math.max(1, ...Object.values(splitCounts).map(Number));
  setHtml('#splitBars', Object.entries(splitCounts).map(([name, count]) => `<div class="split-row"><span>${escapeHtml(pretty(name))}</span><progress max="${maximumSplit}" value="${Number(count) || 0}">${Number(count) || 0}</progress><b>${integer(count)}</b></div>`).join(''));
  set('#foundryState', `${integer(dataset.attack_event_count)} attack events + ${integer(dataset.legitimate_event_count)} legitimate events = ${integer(quality.row_count)} total rows · fixed seed ${dataset.seed} · generated ${new Date(data.generated_at).toLocaleString()}.`);

  set('#benchmarkModel', `SYNTHETIC BENCHMARK · ${pretty(defense.selected_model)} · VALIDATION-SELECTED`);
  set('#benchmarkPrauc', precisePct(metrics.pr_auc, 2));
  set('#benchmarkNovelRecall', precisePct(novel.recall, 1));
  set('#benchmarkF1', precisePct(metrics.f1, 1));
  set('#benchmarkFpr', precisePct(metrics.hard_false_positive_rate ?? metrics.false_positive_rate, 2));
  set('#benchmarkThreshold', `Technical threshold ${Number(defense.threshold_selected_on_validation || 0).toFixed(3)}`);
  const detectedAttacks = Number(confusion.tp) || 0;
  const missedAttacks = Number(confusion.fn) || 0;
  const legitimatePassed = Number(confusion.tn) || 0;
  const falseAlerts = Number(confusion.fp) || 0;
  const novelConfusion = novel.confusion_matrix || {};
  set('#benchmarkPraucNote', `Ranking quality across ${integer(metrics.event_count)} sealed synthetic events. Higher is better.`);
  set('#benchmarkNovelRecallNote', `${integer(novelConfusion.tp)} of ${integer(novel.fraud_event_count)} attack events from completely withheld vectors were caught.`);
  set('#benchmarkF1Note', `${integer(detectedAttacks)} attacks caught · ${integer(missedAttacks)} missed · ${integer(falseAlerts)} legitimate events falsely alerted.`);
  set('#benchmarkFprNote', `Rate measured across ${integer(metrics.hard_control_event_count)} difficult legitimate-control events. Lower is better.`);
  set('#matrixTn', integer(confusion.tn));
  set('#matrixFp', integer(confusion.fp));
  set('#matrixFn', integer(confusion.fn));
  set('#matrixTp', integer(confusion.tp));

  const lifecycle = defense.lifecycle_results || {};
  setHtml('#benchmarkPhases', phases.map(phase => {
    const item = lifecycle[phase] || {};
    const precision = Number(item.precision) || 0;
    const recall = Number(item.recall) || 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return `<div class="benchmark-phase">
      <div class="benchmark-phase-head"><span>${escapeHtml(pretty(phase))}</span><b>${precisePct(recall, 1)} fraud caught</b></div>
      <progress max="100" value="${clamp(recall * 100, 0, 100)}">${precisePct(recall, 1)}</progress>
      <div class="benchmark-phase-kpis">
        <span>${metricTitle('Precision', metricDefinitions.precision)}<b>${precisePct(precision, 1)}</b></span>
        <span>${metricTitle('Recall', metricDefinitions.recall)}<b>${precisePct(recall, 1)}</b></span>
        <span>${metricTitle('F1', metricDefinitions.f1)}<b>${precisePct(f1, 1)}</b></span>
      </div>
      <small>${integer(item.events)} sealed hidden-test events · ${precisePct(1 - (Number(item.false_positive_rate) || 0), 1)} legitimate users protected</small>
    </div>`;
  }).join(''));
  setHtml('#familyResults', Object.entries(defense.family_results || {}).sort().map(([family, item]) => `<tr><td>${escapeHtml(family)}</td><td>${integer(item.events)}</td><td>${precisePct(item.recall, 1)}</td><td>${precisePct(item.value_weighted_recall, 1)}</td><td>${precisePct(item.mean_risk_score, 1)}</td></tr>`).join(''));
  set('#benchmarkGenerated', `${integer(detectedAttacks)} of ${integer(detectedAttacks + missedAttacks)} attacks caught · ${integer(falseAlerts)} of ${integer(legitimatePassed + falseAlerts)} legitimate events flagged`);
}

async function loadBenchmark() {
  const response = await fetch('/api/v2/benchmark');
  if (response.status === 404) { renderBenchmark(null); return; }
  if (!response.ok) throw new Error(`benchmark endpoint returned ${response.status}`);
  renderBenchmark(await response.json());
}

// ---- Event listeners ----

const atlasCategory = $('#atlasCategory');
const atlasRail = $('#atlasRail');
if (atlasCategory) atlasCategory.addEventListener('change', renderThreatAtlas);
if (atlasRail) atlasRail.addEventListener('change', renderThreatAtlas);

const runFoundryBtn = $('#runFoundry');
if (runFoundryBtn) {
  runFoundryBtn.addEventListener('click', async () => {
    runFoundryBtn.disabled = true;
    set('#foundryState', 'Generating population, opening sealed splits and benchmarking candidate defenses…');
    try {
      const response = await fetch('/api/v2/benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variants_per_vector: 6, legitimate_events: 2400, seed: 20260824 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `benchmark returned ${response.status}`);
      renderBenchmark(payload);
      const bench = $('#defenseBenchmark');
      if (bench) bench.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
      set('#foundryState', `Benchmark could not complete: ${error.message}`);
    } finally {
      runFoundryBtn.disabled = false;
    }
  });
}

// ---- Init ----

async function initialize() {
  await Promise.all([loadThreatAtlas(), loadBenchmark()]);
}

initialize().catch(error => {
  const err = $('#errorState');
  if (err) {
    err.textContent = `Could not load evidence: ${error.message}`;
    err.classList.remove('hidden');
  }
});
