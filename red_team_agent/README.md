# SentinelLoop Red Team Agent

This folder contains the first standalone Red Team Agent for the payment-security lab. It plans **safe, synthetic attack scenarios** that a simulator can later turn into transaction, session, identity, and graph events.

The agent does not contact victims, create phishing material, collect personal data, use real accounts, or connect to payment rails.

## What exists now

The first version has five layers:

1. **Attack-card catalogue** — six research-backed fraud families with observable stages, difficulty profiles, source references, legitimate look-alikes, and safe parameter bounds.
2. **Planner** — chooses an attack family, objective, difficulty, and important stages. It works offline by default and can optionally use the OpenAI Responses API.
3. **Deterministic compiler** — turns the bounded decision into a replayable scenario specification using synthetic identifiers and a seed.
4. **Safety gate** — rejects unknown events, out-of-range parameters, unsafe field names, non-synthetic entity identifiers, missing sources, and missing safety constraints.
5. **Mutation controller** — accepts only coarse Referee feedback and creates the next bounded variant. It does not receive Blue Team rules, scores, model internals, or thresholds.

## Closed-loop position

```text
Research-backed attack cards
            |
            v
    Red planner decision
            |
            v
 Deterministic compiler --> Safety gate --> Scenario specification
                                              |
                                              v
                              Simulator / Blue Team / Referee
                                              |
                                  coarse feedback only
                                              |
                                              v
                                  bounded next mutation
```

The output at this stage is a **scenario specification**, not a finished transaction dataset. A later simulator will expand each stage into many synthetic events and the Referee will score Blue Team performance.

## Attack families

| ID | Scenario |
|---|---|
| `APP-01` | GenAI-amplified authorized push payment scam |
| `ATO-01` | GenAI-assisted account takeover and rapid transfer |
| `BEC-01` | GenAI-enhanced supplier impersonation payment diversion |
| `MULE-01` | AI-coordinated mule-network fan-in and dispersal |
| `SYNID-01` | GenAI synthetic-identity onboarding and payment activation |
| `EVADE-01` | Feedback-guided low-and-slow payment evasion |

## Run it without an API key

From the project root:

```bash
python3 -m red_team_agent list
python3 -m red_team_agent show --family ATO-01
python3 -m red_team_agent plan \
  --family ATO-01 \
  --difficulty medium \
  --seed 20260819 \
  --output red_team_agent/runs/ato_medium.json
python3 -m red_team_agent validate red_team_agent/runs/ato_medium.json
```

The same seed, family, and difficulty produce the same scenario.

## Try one Referee feedback cycle

```bash
python3 -m red_team_agent mutate \
  --scenario red_team_agent/runs/ato_medium.json \
  --feedback red_team_agent/examples/referee_feedback.json \
  --output red_team_agent/runs/ato_mutation_01.json
python3 -m red_team_agent validate red_team_agent/runs/ato_mutation_01.json
```

For example, if the Referee says Blue detected `device_novelty`, the bounded controller reduces that probability in the next variant. Blue's exact feature weights or decision threshold are never disclosed.

## Enable the GenAI planner

The offline planner is enough for development and testing. To let an OpenAI model choose among the same bounded cards, configure `OPENAI_API_KEY` in your local terminal or secret manager and run:

```bash
python3 -m red_team_agent plan \
  --backend openai \
  --difficulty medium \
  --output red_team_agent/runs/ai_planned.json
```

Do not paste the key into chat or commit it to the repository. The default model is `gpt-5.4-mini`; set `RED_AGENT_MODEL` locally to override it.

The model is deliberately not allowed to author raw payment events. It returns a small structured planning decision; the local compiler and safety gate remain authoritative.

## Run the tests

```bash
python3 -m unittest discover -s red_team_agent/tests -v
```

## What is needed from you

Nothing is required to run the offline version. Before enabling the live GenAI planner, you will need:

- OpenAI API access with the key stored locally as `OPENAI_API_KEY`.
- A preferred spend limit for API-backed development.
- Later, a decision about which two or three attack families should receive the deepest hackathon demo treatment. The full catalogue can remain broader.

## Next engineering step

Build the **scenario simulator** that converts these validated specifications into timestamped synthetic event streams. That creates the data boundary the Blue Team Agent and Referee will consume.
