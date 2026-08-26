# SentinelLoop v2 — Agent-first architecture

## Product thesis

SentinelLoop is a controlled adversarial laboratory in which one GenAI agent designs and adapts synthetic fraud campaigns, a two-speed Blue defense combines gradient-boosting screening with GenAI investigation and mitigation, and a deterministic Referee measures who succeeded.

The original statistical baseline remains in the repository as historical scaffolding. It is not used by the v2 runtime or the competition demo.

## The three worlds

### Red world

Red receives source-grounded attack cards, bounded parameters, the selected difficulty and a small amount of declassified feedback from the previous round. It decides:

- whether to stress the pre-transaction, transaction or post-transaction phase;
- which stages within that phase are the focus;
- the campaign objective and adaptation hypothesis;
- one to four bounded parameter changes tied to those focus stages;
- how the next variant should test the defense.

Red cannot create real messages, credentials, targets, personal data, phishing URLs or payment-rail actions. Its plan is compiled deterministically and rejected if it exceeds the attack card's bounds.

### Blue world

Blue receives only the event fields a real defensive system might observe. It never receives `attack_family`, `scenario_id`, stage names or fraud labels.

Before any GenAI call, a deterministic sequence guard joins the visible case across lifecycle phases and a Blue-only `HistGradientBoostingClassifier` scores causal tabular features. Both use only current and prior observable evidence and never read sealed truth. Together they form the fast payment-data-plane layer: the classifier score is evidence rather than an automatic verdict, and Qwen may strengthen the minimum safe action but cannot weaken the guard.

Blue then acts as an investigator. It selects from nine read-only tools:

- case-risk synthesis;
- timeline summary;
- entity linkage;
- velocity profile;
- payment context;
- legitimate-alternative search;
- behavioral biometrics;
- communication risk;
- evidence quality.

After the tools return evidence, a separate Blue decision step chooses `allow`, `monitor`, `step_up`, `hold` or `block`, with reason codes, cited evidence and a mitigation explanation.

The legitimacy boundary is explicit. Same-case entity repetition proves continuity only; it does not prove an established relationship. `legitimate_context` can resolve an alert only when the legitimate-alternatives evidence contains independent verification or qualifying historical context.

A `hold` pauses synthetic value movement but does not stop investigation. Later events are still evaluated so the hold can be resolved with legitimate context or escalated to `block`. Only `block` terminates the case.

Between rounds, a Blue Strategist receives a declassified episode summary and proposes a bounded playbook: preferred evidence tools, defensive focus codes and investigation guidance. The candidate is replayed on the exact same attack and legitimate controls. The deterministic Referee—not Blue—promotes it only when safety is preserved and prevention, realized impact, lifecycle resilience or detection timing measurably improves. Wider evidence-tool usage alone does not qualify.

After the Referee scores a round, observable events and post-outcome labels may be written to an append-only Blue training log. At an explicitly configured generation boundary, a challenger detector is trained from scratch with replay data. It is promoted only when at least one judge-facing metric measurably improves and chain recall plus hard-negative false positives stay inside strict non-regression tolerances. Retraining is off in the low-latency UI path unless enabled for a learning-loop demonstration.

### Referee world

The Referee is deterministic because neither competing model should grade itself. It alone owns the sealed mapping between visible event IDs and ground truth. It scores:

- whether and when Blue detected the campaign;
- how much synthetic value Blue protected;
- whether intervention happened at prevent, decide or contain time;
- hard false positives on legitimate look-alike cases;
- customer friction caused by step-up, hold or block actions;
- how much of the event stream Blue evaluated;
- how broadly Blue used the approved evidence layers;
- per-phase response speed, consequence control and legitimate-customer safety;
- a balanced lifecycle score weighted toward the weakest reached phase; and
- Red lifecycle reach, stealth, breadth and stage depth, kept separate from realized financial impact.

Only outcome, detection stage, detection time, protected-value ratio, false-positive rate and coarse reason categories are returned to Red. Blue's prompt, evidence details, thresholds and reasoning never cross that boundary.

## Closed loop

```text
Red GenAI plan
      ↓
bounded compiler + safety gate
      ↓
synthetic observable event stream ───────┐
      ↓                                  │ sealed truth
fast observable sequence guard          │
      ├──────────────┐                   │
      ↓              ↓                   │
Blue gradient      Blue Investigator     │
boosting score ──→ evidence tools        │
      ↓                                  │
Blue Decision Agent                     │
      ↓                                  ↓
          deterministic Referee scoring
                       ↓
          ┌────────────┴────────────┐
          ↓                         ↓
 coarse feedback to Red     episode summary to Blue Strategist
          ↓                         ↓
 Red's next variant         candidate defense playbook
                                    ↓
                       same-case deterministic replay
                                    ↓
                        promote only after safe gain

 sealed post-outcome labels → Blue learning log → challenger detector
                                                ↓
                              promote only after safe measurable gain
```

The learning channels remain separate: Red never receives Blue's playbook, detector score, threshold, features, weights or private evidence. Live Blue never receives the attack family, scenario ID, stage IDs or fraud label. Only the offline retraining job can consume Referee-opened labels after a decision is complete.

## Open-model deployment

The application uses the OpenAI-compatible chat-completions protocol as an interface, not a vendor dependency. The default is `Qwen/Qwen3.5-9B`, served by vLLM with strict JSON-schema output. Red and Blue may share one inference server while remaining isolated by prompt, memory and allowed data.

Live placement is intentionally two-speed. A pure deterministic calculation produces the minimum action before Qwen is called. Independently verified legitimate context may resolve an event immediately only when that minimum remains `allow` and there is no unresolved alert. All other events use one evidence-grounded Qwen response for both investigation and action, while isolated attack and look-alike cases may run concurrently. Transaction events declare a 300 ms lane, pre-transaction events a 2 s lane, and post-transaction events a 5 s streaming lane. The prototype records actual model and end-to-end battle latency but does not pretend local Qwen is a 300 ms hard dependency: model investigation and explanation belong in the asynchronous control plane while the guardrail can sit inline.

For local development, SentinelLoop automatically discovers an already-installed Qwen model exposed by Ollama at `127.0.0.1:11434` when no model environment variables are set. Explicit environment configuration always wins, including the private vLLM address used in deployment.

For a public deployment, keep the model server private. Expose only the Flask/Gunicorn web service. The web service injects the model URL and API key on the server side; the browser never receives them.

Two practical deployment shapes are supported:

1. One GPU host running both containers from `deploy/docker-compose.qwen.yml`.
2. A small public CPU web container using `deploy/Dockerfile`, connected to a private Qwen/vLLM GPU endpoint through environment variables.

The second option is generally easier for a hackathon link because the UI can restart cheaply without reloading model weights.

## Local commands

```bash
# Inspect active configuration without printing the API key
python3 -m sentinelloop config

# With a compatible Qwen endpoint already running
python3 -m sentinelloop run --attack-family ATO-01 --difficulty medium --rounds 2

# Live benchmark on a bounded family/difficulty matrix
python3 -m sentinelloop benchmark --families ATO-01 EVADE-01 --difficulties medium hard

# Start the Agent Arena at http://127.0.0.1:8501
python3 -m app.server

# Run the full regression suite
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s red_team_agent/tests -v
```

## Deployment safeguards still required before a public launch

- Place authentication and request throttling in front of the run endpoint.
- Keep the vLLM port private; an API key alone is not a complete public perimeter.
- Add a persistent run store if more than one web replica is used.
- Pin container image versions and model revisions for the final submission.
- Publish a repository license and third-party attribution file after the team confirms the desired license.
