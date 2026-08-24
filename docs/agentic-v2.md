# SentinelLoop v2 — Agent-first architecture

## Product thesis

SentinelLoop is not a classifier with an LLM-shaped interface. It is a controlled adversarial laboratory in which one GenAI agent designs and adapts synthetic fraud campaigns, a second GenAI agent investigates the resulting payment behavior and chooses mitigation, and a deterministic Referee measures who succeeded.

The original statistical baseline remains in the repository as historical scaffolding. It is not used by the v2 runtime or the competition demo.

## The three worlds

### Red world

Red receives source-grounded attack cards, bounded parameters, the selected difficulty and a small amount of declassified feedback from the previous round. It decides:

- which stages deserve emphasis;
- the campaign objective and adaptation hypothesis;
- up to four bounded parameter changes;
- how the next variant should test the defense.

Red cannot create real messages, credentials, targets, personal data, phishing URLs or payment-rail actions. Its plan is compiled deterministically and rejected if it exceeds the attack card's bounds.

### Blue world

Blue receives only the event fields a real defensive system might observe. It never receives `attack_family`, `scenario_id`, stage names or fraud labels.

For each visible event, Blue first acts as an investigator. It selects from five read-only tools:

- timeline summary;
- entity linkage;
- velocity profile;
- payment context;
- legitimate-alternative search.

After the tools return evidence, a separate Blue decision step chooses `allow`, `monitor`, `step_up`, `hold` or `block`, with reason codes, cited evidence and a mitigation explanation.

### Referee world

The Referee is deterministic because neither competing model should grade itself. It alone owns the sealed mapping between visible event IDs and ground truth. It scores:

- whether and when Blue detected the campaign;
- how much synthetic value Blue protected;
- whether intervention happened at prevent, decide or contain time;
- hard false positives on legitimate look-alike cases;
- customer friction caused by step-up, hold or block actions.

Only outcome, detection stage, detection time, protected-value ratio, false-positive rate and coarse reason categories are returned to Red. Blue's prompt, evidence details, thresholds and reasoning never cross that boundary.

## Closed loop

```text
Red GenAI plan
      ↓
bounded compiler + safety gate
      ↓
synthetic observable event stream ───────┐
      ↓                                  │ sealed truth
Blue Investigator → evidence tools      │
      ↓                                  │
Blue Decision Agent                     │
      ↓                                  ↓
          deterministic Referee scoring
                       ↓
          coarse declassified feedback
                       ↓
               Red's next variant
```

## Open-model deployment

The application uses the OpenAI-compatible chat-completions protocol as an interface, not a vendor dependency. The default is `Qwen/Qwen3.5-9B`, served by vLLM with strict JSON-schema output. Red and Blue may share one inference server while remaining isolated by prompt, memory and allowed data.

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
