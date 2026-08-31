# MasterGuard AI — Codex Handoff

Last updated: 31 August 2026 (Asia/Kolkata)

This file is the starting context for a new Codex instance working on this repository. Read it before changing branches, deleting untracked files, redesigning the UI, changing replay behavior, or publishing a deployment.

## 1. Product objective

**Product name:** MasterGuard AI — *Attack. Adapt. Defend.*

MasterGuard AI is a safe adversarial-AI payment-security lab for the Mastercard Innovation Challenge 2026. It demonstrates a closed loop:

1. **Identify:** maintain a source-backed Threat Atlas of emerging GenAI-enabled payment-fraud vectors.
2. **Generate:** let a bounded Red agent plan safe synthetic campaigns and materialize realistic payment-event sequences plus difficult legitimate look-alikes.
3. **Defend:** let a Blue agent investigate observable events, choose mitigations, and measure detection quality and customer harm.
4. **Adapt:** let a deterministic Referee reveal only declassified feedback, after which Red evolves and Blue can propose a safer playbook.

The submission is judged on:

- Diversity of attacks identified
- Fidelity of attack simulation
- Detection algorithms and efficacy
- Novelty
- Real-world feasibility in live payments

The user wants a technically credible GenAI-versus-GenAI system, not a conventional classifier-only project. The deterministic guard and Blue ML detector are supporting evidence layers; Red and Blue agents remain central to the live architecture.

## 2. Current Git state

- Current branch: `codex/offline-replay-deployment`
- Tracking branch: `origin/codex/offline-replay-deployment`
- Remote: `https://github.com/u367403_ual/mc-hackathon.git`
- Integrated UI source: `origin/new-ui-deployment` (`a53ba25` and `d4a0ef2`)

The branch now combines the offline replay deployment with the `new-ui-deployment` work. The integration preserves the public/offline replay disclosure, configurable repository link, recorded-model provenance, instant one-to-five-round catalog, and **Start battle** button while adding the dedicated Evidence workspace, live grouped event stream, money counterfactual, and revised Arena report.

There are also user-owned/untracked artifacts and temporary directories. Inspect them before deciding what belongs in Git. In particular:

- `deliverables/MasterGuard_AI_Attack_Flow.pptx`
- `MasterGuard_AI_Mastercard_Inspired_Deck.pptx`
- `MasterGuard_AI_Submission_Deck.pptx`
- Corresponding deck working directories
- Browser-review PNG files
- `.tmp_attack_flow_slide/`
- `.tmp_submission_deck/`
- `.playwright-mcp/`

Do not delete these as cleanup without explicit user approval. The final attack-flow deck is intentionally in `deliverables/`; temporary build directories may be excluded later after confirming the final deliverables are safe.

## 3. Architecture and information boundary

The main execution path is:

```text
Battle configuration
  → Red GenAI planner
  → deterministic scenario compiler and safety gate
  → synthetic payment arena
  → Blue sequence guard + Blue-only ML evidence + Blue GenAI investigator
  → deterministic Referee
  → separated Red and Blue feedback loops
```

### Red output

The Red planner produces a structured `RedPlan` containing the objective, lifecycle focus, bounded parameter changes, adaptation goal/hypothesis, and reasoning summary. The compiler converts this into an approved `ScenarioSpec`.

### Arena output

The simulator intentionally separates two data contracts:

- `ObservedEvent[]`: event time, lifecycle phase, event type, observable signals, attributes, source system, decision lane, and latency budget. Blue may inspect this.
- `TruthRecord[]`: true stage, attack family, scenario ID, intervention point, fraud label, value at risk, and other answer-key fields. Only the Referee may inspect this.

### Blue output

For each observable event, Blue produces a `BlueTurn` containing investigation requests, evidence packets, risk, confidence, action, reason codes, decision summary, and mitigation. Actions are bounded to `allow`, `monitor`, `step_up`, `hold`, and `block`.

### Referee output

The Referee joins Blue decisions with sealed truth and produces phase-aware scores including detection, simulated decision latency, value protected, false positives, legitimate friction, and lifecycle balance.

Blue must never receive `attack_family`, `scenario_id`, fraud labels, value-at-risk truth, or Referee answer-key data during a battle.

Important implementation areas:

- `sentinelloop/red_agent.py`
- `sentinelloop/simulation.py`
- `sentinelloop/blue_agent.py`
- `sentinelloop/referee.py`
- `sentinelloop/orchestrator.py`
- `red_team_agent/attack_cards.json`
- `sentinelloop/blue_ml/`

## 4. Payment lifecycle and metrics

The solution is explicitly divided into:

- **Pre-transaction / Prevent:** identify precursor risk before value moves.
- **Transaction / Decide:** intervene during payment initiation or authorization.
- **Post-transaction / Contain:** detect linked behavior, stop subsequent movement, and contain a campaign.

The user considers pre-transaction detection the highest-impact capability. Precision, recall, and F1 answer whether the system caught fraud accurately; simulated decision latency and value protected express the consequence of catching it early or late.

The UI must keep metric definitions understandable to a first-time judge. Important metrics should retain information icons/tooltips and explicit source labels. Single-battle metrics must not be confused with population benchmark claims.

## 5. Two supported execution modes

### A. Public/offline replay mode

Set:

```bash
export DEMO_MODE=precomputed
python -m app.server
```

This mode:

- Calls no LLM or external model API.
- Loads recorded, bounded agent runs from `data/demo_runs/`.
- Supports one through five rounds for every attack-family/difficulty selection.
- Preserves recorded Red plans, payment events, Blue decisions, adaptations, and Referee reports.
- Displays an aesthetic disclosure explaining the model-deployment constraint.
- Uses the primary button label **Start battle**, never “Load replay.”
- Still labels the result as a precomputed replay and retains originating-model provenance in the report.

Current catalog: 54 JSON artifacts.

- 27 one-round Qwen 3.5 9B recordings: 9 attack families × 3 difficulties.
- 27 complete five-round adaptive recordings whose metadata identifies Claude Opus 4.8 as the originating Red/Blue model.

The Claude provenance does **not** affect offline execution. The server selects artifacts by attack family, difficulty, and requested round count—not by originating model. Do not claim that the five-round artifacts were generated by Qwen. Replacing them later with Qwen-generated five-round artifacts requires no architecture change.

The public Render blueprint enables `DEMO_MODE=precomputed` by default.

### B. Live open-model mode

Install Ollama and an appropriate Qwen model, then use settings similar to:

```bash
export MODEL_BASE_URL=http://127.0.0.1:11434/v1
export MODEL_API_KEY=ollama
export RED_MODEL_ID=qwen3.5:9b
export BLUE_MODEL_ID=qwen3.5:9b
export MODEL_STRUCTURED_OUTPUT_MODE=json_schema
export MODEL_REASONING_EFFORT=none
unset DEMO_MODE
python -m app.server
```

The application does not download or start Ollama automatically. Read `README.md` for current installation, model compatibility, timeout, and Docker/vLLM instructions.

Reasoning effort is set to `none` by default for direct structured generation and local latency reliability. The gateway remains compatible with endpoints that do not implement a reasoning parameter.

## 6. New-laptop quick start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export DEMO_MODE=precomputed
python -m app.server
```

Open:

```text
http://127.0.0.1:8501/
```

Health check:

```text
http://127.0.0.1:8501/healthz
```

Expected response:

```json
{"service":"masterguard-ai","status":"ok"}
```

Run tests:

```bash
python -m unittest discover -s tests
```

At this handoff, after integrating `new-ui-deployment`, the complete deterministic suite passed **89 tests**. Browser validation also covered the Arena, Evidence page, run guide, Threat Atlas filters, and one- and five-round recorded battles. Re-run the complete suite after cloning or merging.

## 7. Deployment

The intended public deployment path is Render using `render.yaml` and `deploy/Dockerfile`.

For the public judging demo:

- Keep `DEMO_MODE=precomputed`.
- Keep model credentials out of Git and out of browser responses.
- Set `SOURCE_REPOSITORY_URL` to the repository judges should visit.
- The disclosure CTA will link to `${SOURCE_REPOSITORY_URL}#quick-start-ollama-and-qwen`.
- Verify `/healthz`, the root page, one-round selection, and five-round selection after deployment.

For live generation, clear `DEMO_MODE` and provide a private authenticated OpenAI-compatible model endpoint. Never expose the model service or its credential to the browser.

## 8. UI/product decisions that should be preserved

- Brand: **MasterGuard AI — Attack. Adapt. Defend.**
- Arena is the first/primary experience. Identify, Generate, and Defend live together on the dedicated `/evidence` workspace, with external proof remaining in the Arena.
- Primary button: **Start battle** in both live and recorded modes.
- Public replay mode must clearly say **Recorded agent replay** and **Recorded architecture**, not imply a currently connected live model.
- The replay disclaimer should remain visible, polished, and non-apologetic.
- The disclosure explains that no LLM is deployed on the public host, that no model/API call occurs while loading a replay, what evidence is preserved, and how to run the live system from the repository.
- Actual originating models remain visible as provenance inside a loaded run.
- Dropdowns remain functional and populated from the server's available catalog.
- Metrics use plain English, information icons, and visible source labels.
- The Arena report retains the grouped Red attack event stream, Blue action markings, no-defense loss counterfactual, and event-by-event investigation view.
- Five-round offline battles must remain instant and reliable.
- No real customer data, credentials, phishing content, or external attack execution belongs in the demo.

## 9. Evidence surfaces already built

- Threat Atlas: 31 researched vectors, 13 unique sources, 7 payment rails, 9 bounded simulator families.
- Scenario Foundry: reproducible synthetic population generation and declared-prior fidelity reporting.
- Defense Benchmark: sealed synthetic testing including known patterns, withheld vectors, hard legitimate controls, PR-AUC/F1/recall/false-positive evidence, and family/phase breakdowns.
- External validation: public anonymized card-fraud data with caveats and evidence boundaries.
- Agent Arena: Red plan, Blue event decisions, phase-aware Referee metrics, feedback rounds, and replay provenance.
- Presentations and documents under `deliverables/` and the root working tree.

Do not present catalogue counts, population benchmark metrics, external-validation metrics, and one battle's metrics as if they came from the same dataset or run. The UI currently labels their sources separately.

## 10. Completed UI integration

`origin/new-ui-deployment` has been merged into `codex/offline-replay-deployment`. The result includes:

- `/` — Arena-first battle configuration, external validation, report and offline disclosure.
- `/evidence` — Threat Atlas, Scenario Foundry and sealed Defense Benchmark.
- `/run-guide` — plain-English execution stages and output boundaries.
- Grouped attack/look-alike/ambient event data in live progress and saved round reports.
- `no_defense_loss_inr` and `loss_avoided_inr` Referee fields for the money counterfactual.

Keep the Evidence page and Arena contracts synchronized when changing APIs or report fields. Older recorded artifacts do not contain every newly added field, so UI rendering must continue to use safe defaults.

## 11. Recommended first commands for the next Codex instance

```bash
git status --short --branch
git log -5 --oneline --decorate
git diff --check
python -m unittest discover -s tests
```

Then inspect recent history and the deployment configuration:

```bash
git log -8 --oneline --decorate
git diff origin/codex/offline-replay-deployment...HEAD --stat
sed -n '1,180p' render.yaml
```

Do not run destructive Git cleanup commands. Do not overwrite user-owned presentation assets. Do not push, create a GitHub repository, or change repository visibility without the user's authorization and the missing repository details.

## 12. Portable references

- Main project guide: `README.md`
- Environment template: `.env.example`
- Render deployment: `render.yaml`
- Container deployment: `deploy/Dockerfile`
- Web server: `app/server.py`
- Main UI: `app/templates/lab.html`, `app/static/lab.css`, `app/static/lab.js`
- Run explanation: `app/static/run-guide.html`
- Dashboard tests: `tests/test_dashboard.py`
- Attack-flow presentation: `deliverables/MasterGuard_AI_Attack_Flow.pptx`

The Mastercard GenAI whitepaper reviewed during development was referenced from `~/Downloads/Gen_AI_whitepaper.pdf` on the original laptop and is not guaranteed to be inside this repository. If the next phase needs exact citations from that document, transfer it separately and keep its licensing/distribution constraints in mind.
