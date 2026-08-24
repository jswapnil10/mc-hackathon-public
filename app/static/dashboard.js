const COLORS = { cyan: "#3ee4c2", blue: "#53a8ff", red: "#ff647c", amber: "#ffbf69", grid: "#21344a", muted: "#8fa5bc" };

const percent = value => `${(Number(value) * 100).toFixed(1)}%`;
const money = value => new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value);

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = Number(canvas.getAttribute("height"));
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  return { ctx, width, height };
}

function drawGroupedBars(canvas, rows, keyA, keyB, colorA, colorB) {
  const { ctx, width, height } = setupCanvas(canvas);
  const pad = { left: 40, right: 12, top: 12, bottom: 35 };
  const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
  const max = Math.max(...rows.flatMap(row => [row[keyA], row[keyB]]), 1);
  ctx.font = "10px system-ui"; ctx.fillStyle = COLORS.muted; ctx.textAlign = "center";
  rows.forEach((row, i) => {
    const slot = plotW / rows.length, barW = Math.min(20, slot * .27), x = pad.left + slot * i + slot / 2;
    const hA = row[keyA] / max * plotH, hB = row[keyB] / max * plotH;
    ctx.fillStyle = colorA; ctx.fillRect(x - barW - 2, pad.top + plotH - hA, barW, hA);
    ctx.fillStyle = colorB; ctx.fillRect(x + 2, pad.top + plotH - hB, barW, hB);
    ctx.fillStyle = COLORS.muted; ctx.fillText(row.day || row.bucket, x, height - 10);
  });
  ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(width - pad.right, pad.top + plotH); ctx.stroke();
}

function renderResults(target, data, kind) {
  const root = document.querySelector(target);
  root.innerHTML = Object.entries(data).map(([name, values]) => {
    const value = kind === "attack" ? values.recall : values.false_positive_rate;
    const label = kind === "attack" ? `${values.detected}/${values.events} caught` : `${values.false_positives}/${values.events} flagged`;
    return `<div class="result-row ${kind === "legitimate" && value > .1 ? "danger" : ""}">
      <span title="${name}">${name.replaceAll("_", " ")}</span>
      <div class="progress" title="${label}"><i style="width:${Math.min(100, value * 100)}%"></i></div>
      <strong>${percent(value)}</strong>
    </div>`;
  }).join("");
}

function drawNetwork(canvas, network) {
  const { ctx, width, height } = setupCanvas(canvas);
  const accounts = network.nodes.filter(node => node.kind === "account");
  const beneficiaries = network.nodes.filter(node => node.kind === "beneficiary");
  const positions = {};
  accounts.forEach((node, i) => positions[node.id] = { x: 65 + (i % 2) * 75, y: 28 + i * (height - 56) / Math.max(1, accounts.length - 1) });
  beneficiaries.forEach((node, i) => positions[node.id] = { x: width - 95, y: 55 + i * (height - 110) / Math.max(1, beneficiaries.length - 1) });
  ctx.lineWidth = 1;
  network.edges.forEach(edge => {
    const a = positions[edge.source], b = positions[edge.target]; if (!a || !b) return;
    ctx.strokeStyle = "rgba(83,168,255,.2)"; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.bezierCurveTo(width * .43, a.y, width * .57, b.y, b.x, b.y); ctx.stroke();
  });
  network.nodes.forEach(node => {
    const p = positions[node.id]; const beneficiary = node.kind === "beneficiary";
    ctx.fillStyle = beneficiary ? COLORS.red : COLORS.blue; ctx.beginPath(); ctx.arc(p.x, p.y, beneficiary ? 7 : 4, 0, Math.PI * 2); ctx.fill();
    if (beneficiary) { ctx.fillStyle = "#d7e5f3"; ctx.font = "10px system-ui"; ctx.fillText(node.label, p.x + 12, p.y + 3); }
  });
}

function renderCampaigns(campaigns) {
  document.querySelector("#campaign-count").textContent = `${campaigns.length} campaigns`;
  document.querySelector("#campaign-table").innerHTML = campaigns.map(row => `<tr>
    <td><strong>${row.scenario}</strong></td><td>${row.family}</td>
    <td><span class="tag ${row.difficulty === "Hard" ? "red" : row.difficulty === "Medium" ? "amber" : "green"}">${row.difficulty}</span></td>
    <td>${row.events}</td><td>${money(row.average_amount)}</td><td>${row.start}</td>
  </tr>`).join("");
}

async function loadTransactions(kind = "all") {
  const response = await fetch(`/api/transactions?kind=${encodeURIComponent(kind)}`);
  const data = await response.json();
  document.querySelector("#transaction-table").innerHTML = data.rows.map(row => {
    const truth = row.label_fraud ? '<span class="tag red">Fraud</span>' : '<span class="tag green">Legitimate</span>';
    const decision = row.predicted_fraud ? '<span class="tag red">Flagged</span>' : '<span class="tag green">Allowed</span>';
    const riskClass = row.risk_score >= .7 ? "risk-high" : row.risk_score >= .4 ? "risk-medium" : "risk-low";
    return `<tr><td>${row.event_ts}</td><td>${truth}</td><td>${row.scenario_id || row.legitimate_control || "Ordinary"}</td><td class="risk-number ${riskClass}">${percent(row.risk_score)}</td><td>${decision}</td></tr>`;
  }).join("") || '<tr><td colspan="5">No transactions match this view.</td></tr>';
}

async function init() {
  try {
    const response = await fetch("/api/dashboard");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Dashboard data could not be loaded.");
    document.querySelector("#metric-events").textContent = data.summary.events.toLocaleString("en-IN");
    document.querySelector("#metric-fraud").textContent = `${data.summary.fraud_events.toLocaleString("en-IN")} fraudulent`;
    document.querySelector("#metric-precision").textContent = percent(data.summary.precision);
    document.querySelector("#metric-recall").textContent = percent(data.summary.recall);
    document.querySelector("#metric-fpr").textContent = percent(data.summary.false_positive_rate);
    drawGroupedBars(document.querySelector("#timeline-chart"), data.timeline, "legitimate", "fraud", COLORS.cyan, COLORS.red);
    drawGroupedBars(document.querySelector("#risk-chart"), data.risk_histogram, "legitimate", "fraud", COLORS.cyan, COLORS.red);
    renderResults("#attack-results", data.attack_results, "attack");
    renderResults("#legitimate-results", data.legitimate_results, "legitimate");
    renderCampaigns(data.campaigns);
    document.querySelector("#network-caption").textContent = `${data.network.scenario} · ${data.network.edges.length} synthetic transfers`;
    drawNetwork(document.querySelector("#network-canvas"), data.network);
    document.querySelector("#limitations-text").textContent = data.limitations.join(" ");
    await loadTransactions();
  } catch (error) {
    document.querySelector("main").innerHTML = `<div class="error-box"><h2>Dashboard unavailable</h2><p>${error.message}</p></div>`;
  }
}

document.querySelectorAll(".filter").forEach(button => button.addEventListener("click", async () => {
  document.querySelectorAll(".filter").forEach(item => item.classList.remove("active"));
  button.classList.add("active"); await loadTransactions(button.dataset.kind);
}));

window.addEventListener("resize", () => window.clearTimeout(window.__redraw) || (window.__redraw = window.setTimeout(init, 150)));
init();
