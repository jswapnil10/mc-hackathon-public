# White-paper implementation map

This document records how SentinelLoop turns the Mastercard/Glenbrook generative-AI fraud guidance into a testable adversarial lab. The product remains a synthetic security evaluation environment; it is not connected to customers, payment rails or real case-management systems.

## 1. Defend the full payment lifecycle

Every Red plan now declares one of three lifecycle targets:

- `pre_transaction` — identity, session, profile, beneficiary and payout-destination changes;
- `transaction` — payment or payout authorization and value movement;
- `post_transaction` — dispersal, settlement, dispute investigation, refund and containment.

Red must also name concrete focus stages in that phase. Every proposed parameter change must belong to one of those stages, so lifecycle focus changes the simulated behavior rather than serving as a descriptive label.

## 2. Use layered evidence instead of one fraud score

Blue can combine nine read-only evidence views:

- pre-model cross-phase case-risk synthesis;
- timeline and sequence;
- entity relationships;
- velocity and concentration;
- payment context;
- legitimate alternative explanations;
- behavioral automation indicators;
- communication-risk metadata;
- evidence quality and consistency.

Each evidence packet records its source, the event through which it is current, and a confidence value. Blue must cite available evidence IDs in its action.

Entity evidence distinguishes continuity inside the current case from verified historical trust. A repeated sender-beneficiary edge does not become a legitimate relationship merely because it appears in several events.

## 3. Join detection to operational action

Blue chooses a proportional response: `allow`, `monitor`, `step_up`, `hold` or `block`. A deterministic policy gate enforces compatible risk levels and continuity across the case. Before Qwen runs, an observable-only fast path computes a minimum action; the model may strengthen it but cannot weaken it. Positive `legitimate_context` requires independent verification or qualifying established history.

`hold` is stateful. It pauses synthetic value movement while investigation continues. Later legitimate context can resolve it, while continued risk can preserve the hold or escalate to `block`. Only `block` terminates event evaluation.

## 4. Learn through controlled adversarial replay

SentinelLoop has two separate feedback paths:

1. Red receives only coarse outcome, timing, protected-value and reason categories, then proposes another bounded campaign.
2. Blue's post-episode Strategist receives a declassified defensive summary, then proposes approved evidence tools, focus codes and investigation guidance.

Blue cannot install its own proposal. The arena replays the exact same attack and legitimate look-alike cases with the candidate playbook. The deterministic Referee promotes the candidate only when:

- Blue score does not fall;
- protected value does not fall;
- hard false positives do not increase;
- legitimate customer friction does not increase; and
- prevention, realized impact, lifecycle resilience or detection timing measurably improves.

Evidence coverage remains visible as a diagnostic but cannot promote a strategy by itself.

This makes the loop an evaluation discipline, not unverified self-modification.

## 5. Preserve explainability and governance

- Blue never receives fraud labels, attack-family IDs, scenario IDs, Red stage IDs or intervention truth.
- Red never receives Blue prompts, playbooks, private evidence or thresholds.
- The Referee alone opens sealed truth after decisions are complete.
- Structured model calls, requested tools, evidence packets, actions, policy adjustments, replay results and promotion reasons remain in the run record.
- Red behavior is limited to curated event templates, synthetic identifiers and numeric bounds enforced outside the model.

## 6. Measure both fraud reduction and customer harm

The deterministic report includes:

- attack detection rate and time to detect;
- lifecycle phase of detection;
- separate pre-transaction, transaction and post-transaction phase scorecards;
- response speed from the first actionable opportunity in each reached phase;
- consequence-control ratio and downstream value controlled from that phase;
- legitimate-customer safety for phase-matched look-alike events;
- transition escape, showing whether the campaign advanced to its next planned phase;
- a balanced lifecycle defense score that combines the macro phase average with the weakest reached phase;
- Red capability from lifecycle reach, stealth, phase breadth and stage depth;
- realized impact after Blue's intervention, kept separate from Red capability;
- protected value and protected-value ratio;
- event-evaluation coverage;
- evidence-tool coverage;
- hard false-positive rate;
- legitimate-customer friction;
- Blue defense score and Red evasion score.

Legitimate look-alikes are replayed alongside attacks. A defense cannot win merely by holding or blocking every case.

## 7. Broaden scenario coverage

The catalog now contains nine source-backed synthetic families. Added post-white-paper review:

- `AGENT-01` — agent recognition, consumer consent, signed order intent, payment-container consistency and purchase-scope mismatch;
- `PAYOUT-01` — merchant payout destination manipulation, unusual sales velocity and accelerated settlement;
- `DISPUTE-01` — coordinated dispute submission, evidence inconsistency and refund containment.

Together with APP scams, account takeover, supplier diversion, mule behavior, synthetic identity and adaptive evasion, the demo covers agentic commerce plus pre-transaction, transaction and post-transaction controls.

## Deliberately not claimed yet

- Automatic ingestion of live threat intelligence. A future Threat Research Agent should require source provenance and human approval before a new card enters the executable catalog.
- Real biometric, voice, image or document analysis. The current lab uses defensive metadata signals only.
- Production payment intervention. All actions affect only the synthetic arena. The event contracts and two-speed placement are implementation-ready, but still require institution-approved streaming adapters and shadow calibration.
- Autonomous policy deployment. Candidate defenses require deterministic replay promotion, and a production design would add human change approval.

The primary source is Mastercard and Glenbrook's [Generative AI: Preparing Your Fraud Organization](https://www.mastercard.com/us/en/business/cybersecurity-fraud-prevention/cybersecurity/generative-ai-report.html).
