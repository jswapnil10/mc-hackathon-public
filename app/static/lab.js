const $ = (selector) => document.querySelector(selector);
const state = { run: null, roundIndex: 0, status: null };

const money = (value) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(value || 0);
const pct = (value) => `${Math.round((value || 0) * 100)}%`;
const pretty = (value) => String(value || '—').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const shortModel = (value) => String(value || 'Open model').split('/').pop();

function chips(items, tone = '') {
  if (!items || !items.length) return '<span class="chip">No changes</span>';
  return items.map(item => `<span class="chip ${tone}">${item}</span>`).join('');
}

async function loadStatus() {
  const response = await fetch('/api/v2/status');
  const data = await response.json();
  state.status = data;
  $('#systemMode').textContent = data.mode.replaceAll('_', ' ');
  $('#redModel').textContent = data.models.red;
  $('#blueModel').textContent = data.models.blue;
  $('#refereeModel').textContent = data.models.referee;
  $('#redModelPill').textContent = shortModel(data.models.red);
  $('#blueModelPill').textContent = shortModel(data.models.blue);
  $('#attackFamily').innerHTML = data.attack_families.map(family => `<option value="${family.id}">${family.id} · ${family.name}</option>`).join('');
  if (data.latest_run_available) loadLatest();
}

async function loadLatest() {
  const response = await fetch('/api/v2/latest');
  if (!response.ok) return;
  state.run = await response.json();
  state.roundIndex = state.run.rounds.length - 1;
  renderRun();
}

function renderTimeline(turns) {
  $('#eventTimeline').innerHTML = turns.map((turn, index) => {
    const tools = turn.investigation.requested_tools.map(name => `<span>⌁ ${pretty(name)}</span>`).join('');
    return `<div class="event">
      <div class="event-index">${String(index + 1).padStart(2, '0')}</div>
      <div><div class="event-title">${pretty(turn.event.event_type)}</div><div class="event-summary">${turn.decision.decision_summary}</div><div class="event-tools">${tools}</div></div>
      <span class="action ${turn.decision.action}">${pretty(turn.decision.action)}</span>
    </div>`;
  }).join('');
}

function renderRound(index) {
  state.roundIndex = index;
  const round = state.run.rounds[index];
  const report = round.referee;
  const plan = round.red.plan;
  const scenario = round.red.scenario;
  const feedback = round.feedback_released_to_red;
  $('#blueScore').textContent = Number(report.blue_score).toFixed(1);
  $('#redScore').textContent = Number(report.red_score).toFixed(1);
  $('#outcome').textContent = pretty(report.outcome);
  $('#detectionTime').textContent = report.time_to_detect_seconds == null ? 'Not detected' : `detected in ${report.time_to_detect_seconds}s`;
  $('#valueProtected').textContent = money(report.value_prevented_inr);
  $('#valueAtRisk').textContent = `of ${money(report.total_value_at_risk_inr)} at risk`;
  $('#falsePositives').textContent = pct(report.hard_false_positive_rate);
  $('#strategyFamily').textContent = `${scenario.attack_family} · ${pretty(scenario.difficulty)}`;
  $('#strategyObjective').textContent = plan.objective;
  $('#adaptationHypothesis').textContent = plan.adaptation_hypothesis;
  $('#parameterChanges').innerHTML = chips(plan.parameter_changes.map(item => `${pretty(item.parameter)} → ${item.value}`), 'red');
  renderTimeline(round.blue.attack_turns);
  $('#feedbackOutcome').textContent = pretty(feedback.outcome);
  $('#feedbackStage').textContent = pretty(feedback.detected_stage_id);
  $('#feedbackReasons').innerHTML = chips(feedback.coarse_reason_categories.map(pretty), 'blue');
  $('#feedbackRatio').textContent = pct(feedback.value_prevented_ratio);
  document.querySelectorAll('.round-tabs button').forEach((button, buttonIndex) => button.classList.toggle('active', buttonIndex === index));
}

function renderRun() {
  $('#results').classList.remove('hidden');
  $('#runTitle').textContent = state.run.run_id;
  $('#redModelPill').textContent = shortModel(state.run.model_configuration.red);
  $('#blueModelPill').textContent = shortModel(state.run.model_configuration.blue);
  $('#roundTabs').innerHTML = state.run.rounds.map((_, index) => `<button type="button" data-round="${index}">Round ${index + 1}</button>`).join('');
  document.querySelectorAll('[data-round]').forEach(button => button.addEventListener('click', () => renderRound(Number(button.dataset.round))));
  renderRound(state.roundIndex);
  $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function rotateRunMessage() {
  const messages = ['Red is planning a bounded campaign…','Safety gate is validating synthetic behavior…','Blue is selecting investigation tools…','Referee is opening sealed truth…'];
  let index = 0;
  $('#runStateText').textContent = messages[0];
  return setInterval(() => { index = (index + 1) % messages.length; $('#runStateText').textContent = messages[index]; }, 3000);
}

$('#runForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('#errorState').classList.add('hidden');
  $('#runState').classList.remove('hidden');
  $('#runButton').disabled = true;
  const messageTimer = rotateRunMessage();
  try {
    const response = await fetch('/api/v2/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        attack_family: $('#attackFamily').value,
        difficulty: $('#difficulty').value,
        rounds: Number($('#rounds').value),
        seed: Date.now() % 100000000,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error([payload.error, payload.hint].filter(Boolean).join(' '));
    state.run = payload;
    state.roundIndex = payload.rounds.length - 1;
    renderRun();
  } catch (error) {
    $('#errorState').textContent = error.message;
    $('#errorState').classList.remove('hidden');
  } finally {
    clearInterval(messageTimer);
    $('#runState').classList.add('hidden');
    $('#runButton').disabled = false;
  }
});

loadStatus().catch(error => {
  $('#errorState').textContent = `Could not load the Agent Arena: ${error.message}`;
  $('#errorState').classList.remove('hidden');
});
