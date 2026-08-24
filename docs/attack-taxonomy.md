# GenAI-powered payment-fraud taxonomy

This is an ideation and simulation catalogue, not an operational playbook. The prototype implements the three scenarios marked **MVP**; the remaining entries establish breadth and are candidates for later simulations.

| ID | Attack family | Payment surface | GenAI role | Observable defensive signals | Prototype status |
|---|---|---|---|---|---|
| ATO-01 | AI-phishing-led account takeover | Mobile banking / UPI | Personalized scam content induces credential or session compromise | New device, new payee, session novelty, atypical amount/velocity | **MVP** |
| MULE-01 | Coordinated mule-ring cash-out | P2P / UPI / bank transfer | Automated targeting and campaign coordination | Beneficiary fan-in, shared device/network, bursty transfers, graph communities | **MVP** |
| EVADE-01 | Adaptive fraud mutation | All rails | Agent selects variants that lower prior detector confidence | Feature drift, near-threshold behavior, changing route/amount/time | **MVP** |
| SYNID-01 | Synthetic-identity account creation | Onboarding | Generated identity artifacts and conversational application assistance | Thin-file behavior, linked attributes, early high-risk use | Backlog |
| CARD-01 | Card testing and credential validation | Card-not-present | Generated checkout automation and text variation | Low-value velocity, merchant dispersion, decline bursts | Backlog |
| QR-01 | QR/payee substitution | QR payments | Convincing impersonation and visual/text generation | New recipient, QR-to-payee mismatch, abrupt beneficiary change | Backlog |
| DEEP-01 | Deepfake social-engineering authorization | Call center / payment approval | Voice or video impersonation | New device/session, unusual transaction sequence, trust-channel mismatch | Backlog |
| SCAM-01 | Authorized-push-payment scam | P2P transfer | Tailored persuasion at scale | New payee, urgency-like short session, unusual transfer destination | Backlog |
| MERCH-01 | Merchant / refund abuse | Ecommerce | Generated storefront/support interactions | Refund asymmetry, account/merchant linkage, timing anomalies | Backlog |
| LAUND-01 | Transaction laundering | Merchant acquiring | Generated merchant content and adaptive routing | Category mismatch, unusual ticket size, connected settlement patterns | Backlog |
| BEC-01 | Business payment redirection | Invoice / bank transfer | Impersonated supplier communication | New beneficiary for known supplier, approval-context anomalies | Backlog |
| BOT-01 | Fraud-operations automation | Multi-channel | Agent orchestration, translation, and rapid experimentation | Cross-channel correlation, impossible velocity, coordinated infrastructure | Backlog |

## MVP scenario narratives

### ATO-01: account takeover

A normally low-to-medium velocity customer uses a recently unseen device and network. Their payment session is short, a new beneficiary is created, and an unusual P2P transfer is attempted. Legitimate travel, device upgrades, and genuine first-time recipients are generated as close controls.

### MULE-01: coordinated cash-out

Several compromised or recruited sender accounts pay a small number of mule beneficiaries over a short interval. The network includes shared device or coarse-network infrastructure and structured fan-in. Controls include genuine shared-family devices, busy merchants, and popular beneficiaries.

### EVADE-01: adaptive mutation

The generator observes only high-level model feedback such as misses or low-confidence alerts, then varies permitted scenario parameters: transaction amounts, timing, beneficiary concentration, account aging, and campaign cadence. It does not use real victims, credentials, payment tools, or outbound communications.
