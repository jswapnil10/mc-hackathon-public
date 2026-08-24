# SentinelLoop architecture

```text
                 ┌────────────────────────────┐
                 │ Attack library / taxonomy  │
                 └─────────────┬──────────────┘
                               │ scenario parameters
                 ┌─────────────▼──────────────┐
                 │ Red-team simulator         │
                 │ normal + attack events     │
                 └─────────────┬──────────────┘
                               │ synthetic event stream
                 ┌─────────────▼──────────────┐
                 │ Feature & graph pipeline   │
                 └───────┬─────────────┬──────┘
                         │             │
             ┌───────────▼───┐   ┌─────▼───────────┐
             │ Risk model    │   │ Ring detector   │
             └───────┬───────┘   └─────┬───────────┘
                     └─────────┬───────┘
                               ▼
                 ┌────────────────────────────┐
                 │ Alert explanation + policy │
                 │ allow / challenge / hold   │
                 └─────────────┬──────────────┘
                               │ misses & weak signals
                 ┌─────────────▼──────────────┐
                 │ Mutation controller        │
                 │ creates next scenario run  │
                 └────────────────────────────┘
```

## Decisioning latency model

- **Inline (target under 150 ms):** point-in-time transaction risk features and classifier.
- **Near-real-time (seconds to minutes):** graph enrichment, campaign clustering, alert enrichment.
- **Offline (per simulation run):** mutation selection, retraining, calibration, evaluation, and fidelity checks.

## Mitigation policy for the prototype

| Risk band | Action | Example reason |
|---|---|---|
| Low | Allow and log | Known device/payee, normal behavior |
| Medium | Step-up / challenge | New device and material behavior change |
| High | Temporary hold and analyst queue | ATO pattern or strong ring association |
| Critical | Hold and linked-entity review | Coordinated mule-ring evidence |

The dashboard must always show the score, top reasons, proposed action, and a simulated outcome. The model recommends actions; it does not autonomously make real payment decisions.
