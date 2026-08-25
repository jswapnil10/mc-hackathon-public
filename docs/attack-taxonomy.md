# GenAI-powered payment-fraud taxonomy

This is an ideation and simulation catalogue, not an operational playbook. The nine **Implemented** families are represented as bounded synthetic event sequences; the remaining entries are research candidates rather than executable attacks.

| ID | Attack family | Payment surface | GenAI role | Observable defensive signals | Prototype status |
|---|---|---|---|---|---|
| AGENT-01 | Agentic-commerce intent and checkout manipulation | Agentic checkout / tokenized payment | Coordinates a synthetic shopping-agent journey and adapts trust-signal fidelity | Agent signature, consent, intent scope, payment-container match, replay freshness | **Implemented** |
| ATO-01 | AI-assisted account takeover | Mobile banking / UPI | Adapts bounded session and payment behavior | New device, new payee, session novelty, atypical amount/velocity | **Implemented** |
| MULE-01 | Coordinated mule-ring cash-out | P2P / UPI / bank transfer | Adapts bounded network and dispersal behavior | Beneficiary fan-in, shared device/network, bursty transfers, graph communities | **Implemented** |
| EVADE-01 | Adaptive fraud mutation | All rails | Agent selects bounded variants using coarse Referee feedback | Feature drift, near-threshold behavior, changing route/amount/time | **Implemented** |
| SYNID-01 | Synthetic-identity account creation | Onboarding | Represents identity consistency and automation signals without generating identity artifacts | Thin-file behavior, linked attributes, early high-risk use | **Implemented** |
| APP-01 | Authorized-push-payment scam | P2P transfer | Represents personalized persuasion through defensive metadata only | New payee, urgency-like short session, unusual transfer destination | **Implemented** |
| BEC-01 | Business payment redirection | Invoice / bank transfer | Represents supplier impersonation through profile and payment context | New beneficiary for known supplier, approval-context anomalies | **Implemented** |
| PAYOUT-01 | Merchant payout destination manipulation | Merchant acquiring / marketplace payout | Coordinates bounded merchant-profile and settlement behavior | Destination novelty, profile-change recency, sales velocity, payout value | **Implemented** |
| DISPUTE-01 | Dispute evidence and refund abuse | Card disputes / marketplace refunds | Represents scaled submissions using behavioral and evidence-quality signals | Cross-case linkage, submission timing, evidence conflict, refund value | **Implemented** |
| CARD-01 | Card testing and credential validation | Card-not-present | Generated checkout automation and text variation | Low-value velocity, merchant dispersion, decline bursts | Backlog |
| QR-01 | QR/payee substitution | QR payments | Convincing impersonation and visual/text generation | New recipient, QR-to-payee mismatch, abrupt beneficiary change | Backlog |
| DEEP-01 | Deepfake social-engineering authorization | Call center / payment approval | Voice or video impersonation | New device/session, unusual transaction sequence, trust-channel mismatch | Backlog |
| LAUND-01 | Transaction laundering | Merchant acquiring | Generated merchant content and adaptive routing | Category mismatch, unusual ticket size, connected settlement patterns | Backlog |
| BOT-01 | Fraud-operations automation | Multi-channel | Agent orchestration, translation, and rapid experimentation | Cross-channel correlation, impossible velocity, coordinated infrastructure | Backlog |

## MVP scenario narratives

### ATO-01: account takeover

A normally low-to-medium velocity customer uses a recently unseen device and network. Their payment session is short, a new beneficiary is created, and an unusual P2P transfer is attempted. Legitimate travel, device upgrades, and genuine first-time recipients are generated as close controls.

### MULE-01: coordinated cash-out

Several compromised or recruited sender accounts pay a small number of mule beneficiaries over a short interval. The network includes shared device or coarse-network infrastructure and structured fan-in. Controls include genuine shared-family devices, busy merchants, and popular beneficiaries.

### EVADE-01: adaptive mutation

The generator observes only high-level model feedback such as misses or low-confidence alerts, then varies permitted scenario parameters: transaction amounts, timing, beneficiary concentration, account aging, and campaign cadence. It does not use real victims, credentials, payment tools, or outbound communications.

### AGENT-01: agentic-commerce intent manipulation

A synthetic shopping agent presents bounded identity, consumer-consent, order-intent and payment-container telemetry. The attack tests mismatches between signed intent and the resulting purchase sequence without generating credentials or executable checkout automation. Legitimate controls present a complete verified agent, consent and payment trust chain.
