# Blue-Team ML Evolution — Research & Design

**What this is.** A research-grounded design for evolving SentinelLoop (this repo) from an LLM-only Blue defender into a **hybrid ML + LLM defender** that (1) scores every stage of an attack, (2) retrains on a batch cadence driven by loop feedback, (3) is trained on **simulated legit + fraud tabular rows** with matched hard-negatives, and (4) eventually faces a Red agent that synthesises **entirely new** attacks — not just parameter variants. It takes inspiration from the sibling card-fraud repo (`mastecard-hackathon`) **without reusing its card features**, and expands each scenario's attribute surface with realistic, source-cited signals.

> Provenance: produced by a multi-agent deep-research workflow (2 repo-recon agents, 6 per-scenario web-research agents, 4 architecture agents, 1 skeptic critic). ~105 candidate attributes were proposed and filtered through the critic's keep/cut/fix verdicts. All web sources are listed in §12.

---

## 1. Executive summary

The single most important finding is a **warning, not a feature**: nearly every proposed attribute, as first drafted, would *leak the label* in this simulator. The upstream repo (`mastecard-hackathon`) already fought and documented this battle; SentinelLoop must import its **disciplines**, not its columns. Concretely:

1. **Leakage discipline first (§2).** Every new attribute needs a *control-emission spec* with overlapping legit/fraud distributions, must be present (not NaN) on legit look-alikes, and the simulator's hard-coded `sender_baseline_amount_inr = 2500` and 1–2-event control chains must be fixed. Without this, the attributes are worthless.
2. **Consolidate duplicates into canonical cross-family attributes (§3).** The six research agents independently proposed ~5 names each for "receiver fan-in," "onward-dispersal latency," "dormancy," "device/SIM-binding age," "KYC tier," and "Confirmation-of-Payee." Standardise one name, dtype, and `parameter_bounds` per concept.
3. **Expand each scenario (§4)** with the surviving, realistic signals — behavioural, device/network, velocity/temporal, graph, identity/KYC, monetary, channel/auth — each tagged with the stage it becomes observable, so Blue can score at **every** stage.
4. **Add an ML detector the Blue agent consumes as evidence (§5).** A `HistGradientBoostingClassifier` scores each `ObservedEvent` from the strictly-prior `visible_history` window that `blue_agent.py` already maintains; the score is injected as a deterministic `EvidencePacket` (`ml_risk_score`) that the LLM decider reads. Cumulative session risk is a running max, mapped to stage-aware thresholds.
5. **Generational retraining (§6):** inference every round, **full retrain every N rounds** (frozen "champion" scorer during a generation), with replay buffer, drift trigger, and a champion/challenger gate.
6. **Rows come from the event stream, not `src/simulator` (§7).** Derive one training row per `ObservedEvent` joined to its sealed `TruthRecord`; a bulk batch builder sweeps families × difficulties × seeds to reach realistic (~0.75%) imbalance with matched controls.
7. **Novel-attack generation is a two-loop Red (§8):** keep the current bounded card path; add an outer "Scenario Architect" that proposes *candidate* cards grounded in threat intel (RAG), admitted through an **expandable allowlist + review gate** — never by loosening `safety.py`.

**Priority order to build:** §2 leakage fixes + §7 row builder → §5 ML detector + per-stage scoring → §6 retraining loop → §3/§4 attribute expansion (incrementally, each with its control spec) → §8 novel-attack generation.

---

## 2. Non-negotiable leakage discipline (do this or nothing else matters)

The critic flagged this as systemic. Five concrete leakage vectors, each with the fix:

| # | Leakage vector | Why it copies the label | Fix |
|---|---|---|---|
| L1 | **Disjoint fraud/legit bounds** | Most proposals specify e.g. `legit 0.7–1.0` vs `fraud 0.0–0.5`. If the simulator draws attack values from the fraud range and controls from the legit range, the feature *is* the label. | For every attribute, define a **control-emission spec** with **overlapping** distributions and matched base rates (legit CoP mismatches happen; some genuine rushed payments hug thresholds). |
| L2 | **Presence/absence leakage** | New attributes get added only to attack stage templates; `simulation.py::_control_shape` controls wouldn't carry the keys, so **NaN-vs-present** alone separates classes. | Emit **the same attribute keys** on controls (and on the benign baseline), populated from the overlapping legit distribution. |
| L3 | **Chain-length leakage** | Attacks are 4–6 stages; controls are 1–2 events. Stage count/type alone separates classes. | Expand `simulate_legitimate_controls` / `_control_shape` into **full-length benign chains** mirroring each family's stage structure. |
| L4 | **Baseline-amount leak** | `simulation.py:41` hard-codes `sender_baseline_amount_inr = BASELINE_AMOUNT_INR (2500)` for all attack value events while controls use varied literals (1500/2500/4000/42000). Any amount-vs-baseline feature trivially separates. | Give **each synthetic account a population-drawn baseline** and jitter multipliers for **both** legit and fraud; feed **ratios**, not raw baselines. |
| L5 | **Model-score-as-feature circularity** | `behavioral_biometric_anomaly_score`, `..._deviation_score`, `automation_cadence_score`, `session_navigation_familiarity/fluency_score`, `account_entry_paste_ratio` are hand-authored "fraud-likeness" scores → a simulator draws them from disjoint ranges = trivial leakage. | Keep **at most one** consolidated behavioural score, generated from an overlapping distribution, and treat it as low-trust; prefer *observable* behavioural primitives (hesitation counts, paste ratio) with heavy overlap. |

**Two mechanical prerequisites the critic surfaced:**
- **`parameter_bounds` wiring:** every `is_mutable_param=true` attribute **must** get a matching `parameter_bounds` entry in `attack_cards.json`, or `safety.py`'s fail-closed gate rejects the scenario (`unbounded_parameters`, line ~143; out-of-bounds, lines ~124–130).
- **Simulator materialization gap:** `_materialize_attributes` only converts **5 named** `*_probability` fields (`new_device`, `new_network`, `shared_device`, `shared_network`, `device_reuse`) into booleans. Any new `X_probability` attribute needs to be added to that materializer or it will pass through un-rolled.

Also honour the existing **forbidden key fragments** (`safety.py`, case-insensitive): `password, credential, otp_value, pin_value, cvv, pan, real_name, phone_number, email_address, message_content, phishing_url, malware, exploit_code`; any `*_id` must use the `syn_` prefix; `amount_inr` ∈ [1, 250000]. This is why several email/domain/content attributes were cut (§4, cut list).

---

## 3. Canonical cross-family attributes (consolidate the duplicates)

The research agents proposed the same concept under different names/dtypes across families. Standardise **one** canonical attribute each (one dtype, one enum, one `parameter_bounds`), emitted with the L1/L2 overlap discipline. These are the highest-leverage additions because they are reusable and graph/behaviour-grounded.

| Canonical name | dtype | Replaces (per-family drafts) | Signal class | Legit vs fraud (with required overlap) | Typical stage |
|---|---|---|---|---|---|
| `receiver_inbound_sender_count` | int | APP `beneficiary_inbound_sender_count_24h`, ATO `receiver_inbound_sender_count_24h`, MULE `beneficiary_indegree_24h`, EVADE `beneficiary_indegree_7d`, BEC `destination_fan_in_sender_count`, existing `sender_count` | graph | Legit payees + popular merchants have stable/high but *explained* fan-in; mules spike from many unrelated senders. **Overlap:** popular-merchant control must also show high fan-in. | CONTAIN (receipt) |
| `funds_dwell_time_seconds` | int | APP `beneficiary_dispersal_latency_seconds`, ATO `receiver_outflow_latency_seconds`, BEC `funds_dwell_time_seconds`, MULE `dispersal_latency_seconds`, EVADE `dispersal_delay_seconds` | temporal/graph | Mules pass funds through in seconds–minutes; legit hold hours–days. **Overlap:** some legit sweep-accounts move fast. | CONTAIN |
| `dormancy_days_before_activity` | int | ATO `dormancy_days_before_session`, MULE `account_dormancy_days`, EVADE `beneficiary_dormancy_gap_days` | temporal | Long dormancy → sudden burst = sleeper mule/ATO. **Overlap:** seasonal legit accounts also reactivate. | PREVENT/warmup |
| `device_binding_age_days` | int | ATO `device_binding_age_days`, APP/MULE `sim_binding_age_days` | device/identity | Fresh binding + high-value txn = takeover/bulk provisioning. **Overlap:** legit new-phone/travel produces low binding age too. | PREVENT |
| `beneficiary_name_match_score` | float 0–1 | APP + BEC `beneficiary_name_match_score` (Confirmation-of-Payee / RBI name lookup) | identity | Mismatch = funds to a mule under a different name. **Overlap:** legit CoP near-misses (abbreviations); fraud partial matches. | PREVENT (beneficiary add) |
| `beneficiary_kyc_tier` | categorical enum `{min_kyc, full_kyc, business_kyc}` | APP/ATO/BEC/MULE/EVADE variants (had bool/int/categorical mix) | identity/KYC | Mule endpoints skew min-KYC. **Overlap:** legit thin-file/wallet users are also low-tier. | PREVENT/CONTAIN |
| `add_to_payment_seconds` | int | APP `new_beneficiary_to_payment_latency_seconds`, ATO `beneficiary_add_to_transfer_seconds` | temporal | Cooling-period compression: add payee → pay within seconds. **Overlap:** some genuine urgent payments are quick. | DECIDE |
| `balance_drain_ratio` | float 0–1 | APP + ATO `balance_drain_ratio` | monetary | Near-full sweep = cash-out. **Overlap:** many legit payments are a large fraction of balance. | DECIDE |
| `data_entry_hesitation_ratio` | float 0–1 | APP `data_entry_hesitation_count`, ATO `data_entry_hesitation_ratio`, BEC `account_entry_paste_ratio`, EVADE `form_field_hesitation_ms` | behavioural | Unfamiliar/dictated payee data → paste/hesitation. **Overlap:** legit users paste/typo too. *(Low-trust; see L5.)* | PREVENT/DECIDE |
| `round_amount_ratio` | float 0–1 | BEC `amount_roundness_score`, MULE `round_amount_ratio` | monetary | Scripted transfers over-use round values. **Overlap:** many legit payments are round. | value stage |
| `behavioural_deviation_score` | float 0–1 | APP/ATO/MULE/EVADE biometric & automation scores | behavioural (low-trust, L5) | Operator ≠ enrolled owner / scripted cadence. **Must** be drawn from overlapping ranges. Keep **one only**. | PREVENT |

---

## 4. Per-scenario attribute expansion

Each table lists the **surviving family-specific** attributes (canonical ones from §3 also apply). All are additive to the existing card keys; all require the §2 control-emission spec. `mut` = should be a red-team-tunable `parameter` (needs a `parameter_bounds` entry); `obs` = derived observable (not tuned).

Inspiration from the card repo is *conceptual* (velocity/graph/first-seen/impossible-travel patterns) — **no card feature name is reused** (those belong to card-present fraud: `entry_mode`, `avs_result`, `cvv_result`, `otp_status`, `dist_from_home_km`, `mcc`, `bin`, etc.).

### 4.1 ATO-01 (account takeover → rapid transfer)
| Attribute | dtype | class | stage | mut/bounds |
|---|---|---|---|---|
| `sim_swap_recency_hours` | int | identity | novel_session | mut 0–8760 |
| `ip_geo_velocity_kmph` | float | velocity/geo | novel_session | mut 0–5000 |
| `security_settings_change_count` | int | behavioural | authentication_change | mut 0–10 |
| `transaction_limit_increase_ratio` | float | monetary/auth | authentication_change | mut 1.0–50.0 |
| `step_up_auth_failure_count` | int | auth | authentication_change | mut 0–10 |
| `auth_change_to_payment_seconds` | int | temporal | test_transfer | mut 0–86400 |
| `test_to_primary_amount_ratio` | float | monetary | primary_transfer | **fix**: reconcile with existing `test_payment_multiplier` knob rather than adding a parallel amount |
| + canonical: `device_binding_age_days`, `dormancy_days_before_activity`, `balance_drain_ratio`, `add_to_payment_seconds`, `receiver_inbound_sender_count`, `funds_dwell_time_seconds`, `beneficiary_name_match_score` |||||

*Cut:* `cross_rail_switch_flag` (no rail field in the event schema), `beneficiary_prior_fraud_link_count` (label-in-disguise), `remote_access_tool_detected` → use canonical `remote_access_tool_active`.

### 4.2 APP-01 (authorised push-payment scam)
| Attribute | dtype | class | stage | mut/bounds |
|---|---|---|---|---|
| `remote_access_tool_active` | bool via `remote_access_probability` | device | persuasion_context | mut 0–1 (add to materializer) |
| `accessibility_service_enabled` | bool via prob | device | persuasion_context | mut 0–1 |
| `beneficiary_entry_method` | categorical `{saved,typed,paste,qr,link}` | channel | new_beneficiary | mut (weight legit toward paste/qr too) |
| `upi_collect_request_flag` | bool via prob | channel/auth | authorized_payment | mut 0–1 |
| `payment_amount_escalation_ratio` | float | temporal/monetary | payment_follow_up | mut 0.5–20 |
| + canonical: `beneficiary_name_match_score`, `add_to_payment_seconds`, `balance_drain_ratio`, `data_entry_hesitation_ratio`, `receiver_inbound_sender_count`, `funds_dwell_time_seconds`, `beneficiary_kyc_tier` |||||

*Cut:* `beneficiary_prior_scam_report_count` (label-in-disguise). *Fix/flag:* `concurrent_call_active` — a sender-bank app generally cannot observe a separate voice call; keep **only** if modelled as a coarse device-telephony risk flag with heavy overlap, else drop. `behavioral_biometric_anomaly_score` → canonical `behavioural_deviation_score` (one only).

### 4.3 BEC-01 (supplier-impersonation / invoice diversion)
| Attribute | dtype | class | stage | mut/bounds |
|---|---|---|---|---|
| `beneficiary_bank_change_count` | int | vendor/mandate | supplier_profile_change | mut 0–10 |
| `beneficiary_bank_region_mismatch_flag` | bool via prob | geo | replacement_beneficiary | mut 0–1 |
| `beneficiary_prior_inbound_count` | int | graph/familiarity | replacement_beneficiary | mut 0–500 |
| `invoice_amount_zscore` | float | monetary | diverted_invoice_payment | **fix** (depends on L4 baseline fix) mut −3–6 |
| `payment_offhours_flag` | bool via prob | temporal | diverted_invoice_payment | mut 0–1 |
| `maker_checker_interval_seconds` | float | control/temporal | diverted_invoice_payment | mut 1–86400 |
| `out_of_band_verification_complete` | bool | **legit control** (obs, not tuned) | diverted_invoice_payment | rename to match existing `evidence.py` marker |
| `destination_shared_onboarding_count` | int | graph | destination_receipt | mut 0–30 |
| + canonical: `beneficiary_name_match_score`, `funds_dwell_time_seconds`, `receiver_inbound_sender_count`, `beneficiary_kyc_tier`, `round_amount_ratio`, `data_entry_hesitation_ratio` |||||

*Cut (content-free lab):* `sender_domain_similarity_score`, `sender_domain_age_days`, `thread_reply_to_changed_flag` — require inspecting email domains/threads/content, which `safety.py` forbids and `content_available=false` precludes.

### 4.4 MULE-01 (AI-coordinated fan-in / dispersal)
| Attribute | dtype | class | stage | mut/bounds |
|---|---|---|---|---|
| `device_account_linkage_count` | int | graph/device | account_warmup | mut 1–200 |
| `emulator_or_rooted_flag` | bool via prob | device | account_warmup | mut 0–1 (enforce small nonzero legit) |
| `hosting_network_ratio` | float | network | fan_in | mut 0–1 |
| `fan_in_burst_velocity` | float | velocity | fan_in | mut 0.1–500/min (reuse `evidence.py` events_per_minute) |
| `inbound_sender_novelty_ratio` | float | graph | fan_in | mut 0–1 |
| `sender_geo_dispersion_index` | float | geo/graph | fan_in | mut 0–1 |
| `amount_structuring_ratio` | float | monetary | fan_in | mut 0–1 |
| `dispersal_channel_diversity` | int | channel | rapid_dispersal | mut 1–6 |
| `crypto_offramp_probability` | bool via prob | channel | rapid_dispersal | mut 0–1 |
| + canonical: `receiver_inbound_sender_count`, `funds_dwell_time_seconds`, `dormancy_days_before_activity`, `device_binding_age_days`, `beneficiary_kyc_tier`, `round_amount_ratio` |||||

*Cut:* `cashout_atm_ratio` (no ATM/cash event type in `ALLOWED_EVENT_TYPES`), `graph_community_reuse_score` (label-in-disguise: overlap with "known mule communities" ≈ `is_attack`), `automation_cadence_score` → fold into single `behavioural_deviation_score` (L5).

### 4.5 SYNID-01 (synthetic-identity onboarding → activation)
Strongest, most leakage-resistant set (identity-fabrication signals are genuinely observable at onboarding). Keep essentially all:
| Attribute | dtype | class | stage | mut/bounds |
|---|---|---|---|---|
| `synthetic_face_match_count` | int | identity/graph | verification_attempt | mut 0–10 (legit control = 0, but keep small nonzero rate) |
| `shared_kyc_artifact_count` | int | identity/graph | verification_attempt | mut 0–10 |
| `pii_reuse_velocity` | int | identity/graph | identity_application | mut 0–25 |
| `digital_footprint_age_days` | int | identity | identity_application | mut 0–3650 |
| `disposable_contact_probability` | float | channel/identity | identity_application | mut 0–1 |
| `presentation_attack_score` | float | liveness | verification_attempt | mut 0–1 |
| `geo_mismatch_score` | float | geo | verification_attempt | mut 0–1 |
| `onboarding_prompting_probability` | float | behavioural | verification_attempt | mut 0–1 |
| `device_emulator_probability` | float | device | account_creation | mut 0–1 |
| `identity_cluster_size` | int | graph | account_creation | mut 1–500 |
| `onboarding_attempt_velocity` | int | velocity | identity_application | mut 1–50 |
| `authorized_user_tradeline_count` | int | credit/graph | history_building | mut 0–15 |
| `warmup_counterparty_diversity` | float | behavioural/graph | history_building | mut 0–1 |
| `dormancy_to_activation_days` | int | temporal | payment_activation | mut 0–720 |
| `bust_out_amount_ratio` | float | monetary | payment_activation | mut 1–50 |
| `form_fill_fluency_score` | float | behavioural (L5, low-trust) | identity_application | mut 0–1 |

> Note: the critic's "SYNID/EVADE got nothing" remark was an artifact of input truncation in the critic prompt — both families **do** have full proposals (above and §4.6). Apply the same §2 overlap discipline; the disjoint `legit=0 / fraud>0` framing (e.g. `synthetic_face_match_count`) is the main risk to soften.

### 4.6 EVADE-01 (feedback-guided low-and-slow) — the hardest family
Cross-event signals matter most here; per-event views are blind by design.
| Attribute | dtype | class | stage | mut/bounds |
|---|---|---|---|---|
| `amount_to_alert_threshold_ratio` | float | monetary (structuring) | aged_beneficiary_payment | mut 0.5–0.99 — *strongest single tell* |
| `step_up_threshold_margin_ratio` | float | auth/monetary | aged_beneficiary_payment | mut 0.5–0.99 |
| `payment_amount_cov` | float | temporal/monetary | spaced_payments | mut 0.0–0.60 |
| `inter_payment_interval_cov` | float | temporal | spaced_payments | mut 0.0–1.50 |
| `inter_payment_jitter_seconds` | int | temporal (the evasion knob) | spaced_payments | mut 0–86400 |
| `daily_cumulative_amount_ratio` | float | monetary (rolling) | spaced_payments | mut 0.5–3.0 |
| `input_automation_likelihood` | float | behavioural | bounded_replay | mut 0–1 |
| `device_graph_reuse_count` | int | graph/device | familiarized_session | mut 1–40 |
| `network_asn_stability_score` | float | network | familiarized_session | mut 0–1 |
| `beneficiary_sender_overlap_ratio` | float | graph (obs) | aged_beneficiary_payment | derived |
| `onward_fanout_count` | int | graph | delayed_dispersal | mut 1–50 |
| `cross_channel_consistency_score` | float | channel | familiarized_session | mut 0–1 |
| + canonical: `beneficiary_kyc_tier`, `dormancy_days_before_activity`, `receiver_inbound_sender_count`, `funds_dwell_time_seconds`, `data_entry_hesitation_ratio` |||||

*Fold:* `session_navigation_fluency_score` → `behavioural_deviation_score` (L5).

---

## 5. ML classifier integration — fire at every stage

**Why it fits the repo unchanged.** `blue_agent.py::GenAIBlueAgent.run_case` already streams one event at a time, appending each to a growing `visible` list and calling `investigate_event(event, visible_history=list(visible), prior_turns=turns)`, breaking on a decisive `hold`/`block`. That `visible_history` (strictly-prior events + current) **is** the point-in-time cursor — the streaming analog of the sibling repo's single causal velocity function. `evidence.py` tools (`_velocity_profile`, `_entity_linkage`, `_payment_context`, `_timeline_summary`, `_legitimate_alternatives`) already compute causal aggregates over that window.

**Design:**
- **Detector** (`sentinelloop/blue_ml/detector.py`): `HistGradientBoostingClassifier` over an allowlisted feature set — categoricals via pandas `category`, NaN left as learnable signal, `sample_weight = N_neg/N_pos` on positives (no SMOTE), params ~`max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2=1.0, early_stopping`. Wrap in `CalibratedClassifierCV` (isotonic ≥~1k samples, else Platt) so per-event `p_t` are true probabilities.
- **One feature function** (`blue_ml/features.py`): `build_features(current_event, visible_history) -> dict`, used **identically** at train and inference time (the "one causal function" discipline). Combines current-event allowlisted attributes + running aggregates (elapsed, events/min, fan-in/out, amount/baseline **ratio**, supporting-context count) + sequence position. **Allowlist with `audit_leakage()`** banning: `ANSWER_KEY` (all `TruthRecord` fields — `is_attack`, `stage_id`, `attack_family`, `scenario_id`, `value_at_risk_inr`, `intervention_point`), `GROUPING_KEYS` (every `*_id`), `FINGERPRINTS` (`event_id`, raw `occurred_at`/`offset_seconds`, `sequence`, and the hand-authored `observable_signals` strings — these are near-perfect stage labels). Derive `hour_of_day`/`day_of_week` only.
- **Per-event → cumulative** (`blue_ml/session.py`): cumulative session risk `S_t = running_max(p_1..p_t)` (or short-window noisy-OR). Running-max matches the repo's existing **risk-continuity** policy ("don't downgrade an unresolved alert").
- **Stage-aware thresholds:** map to `intervention_point`. LOW threshold at **PREVENT** stages triggering only cheap `step_up`/`monitor` (favour recall on thin early evidence); HIGHER thresholds gate money-stopping `hold`/`block` at **DECIDE**/**CONTAIN**. Calibrate against an **alert budget** (~1% of scored rows), reusing `referee.py` chain semantics — **not** per-event AUC.
- **Fusion with the LLM (critical design choice):** inject the score as a deterministic `EvidencePacket(tool_name="ml_risk_score", facts={p_t, S_t, thresholds})` into `investigate_event`'s evidence, so the LLM decider **consumes** it but the number stays deterministic and comparable round-over-round. 

> **Fix flagged by critic:** an earlier draft wanted a hard ML "risk floor" injected into `_decision_errors`/`_normalize_policy_labels` to *force* `block`/`hold`. That **contradicts** the current design — the policy gate only validates internal consistency; it never injects an external score or escalates the chosen action. If a hard floor is desired, add it as an **explicit new, documented policy rule** (a design change with its own test), not a silent mutation of the consistency gate. Default recommendation: start with evidence-only fusion; add an explicit floor later behind a config flag.

---

## 6. Retraining strategy — "frozen-scorer inference, generational batch retrain"

Terminology: a **round** = one `SentinelLoopOrchestrator.run` iteration. A **generation** = N consecutive rounds during which classifier weights are **frozen**. Inference runs every round; training runs **once per generation**.

- **Why batch, not online:** per-round label yield is tiny (~5–6 attack stages + a few controls). Per-event SGD updates would be high-variance and forget. Accumulate to an append-only JSONL and retrain in batch. (River-style online learning is reserved for cheap drift stats, not the classifier.)
- **What each round logs** (`sentinelloop/labeling.py`, called after `referee.score`): one row per `attack_case.events` row **and** per control event → `data/loop/training_log.jsonl`, joined to its `TruthRecord` by `event_id`. **Log all materialised stages, not just the turns Blue processed** — `simulate_attack` emits every stage up front, so early-stop chains are still fully labelled. This avoids selective-labels / feedback-loop bias.
- **Retrain (`sentinelloop/retrain.py`):** **full retrain from scratch** each generation (cheap at this scale; warm-start on HistGBC only adds trees on the same data and would ossify dead attack modes). Handle imbalance with in-fold `sample_weight` only. Blend in a **down-weighted synthetic legit baseline** so the model learns the legit manifold.
- **Replay buffer:** class- and family-stratified reservoir of past positives, always mixed in, so the model retains families/params Red abandons then revisits (e.g. `EVADE-01`).
- **Validation:** `StratifiedGroupKFold` grouped on `scenario_id`/`case_id` (account_id for baseline), stratified by `attack_family`, `test_share≈0.4`. Evaluate **prequentially** (train gen 1..k, hold out gen k+1) with a label embargo to mimic verification latency.
- **Cadence + drift trigger:** fixed **N = 5–10 rounds** as schedule, **plus** a drift detector (ADWIN `delta≈0.002`, or Page-Hinkley) on per-round score-separation / `value_prevented_ratio` to trigger early retrain. Red is an adaptive adversary, so drift is guaranteed.
- **Champion/challenger gate:** promote the retrained challenger to `data/loop/models/champion/` **only if** it beats the incumbent on held-out chain recall at fixed budget **and** does not increase hard-FP/friction on controls. Version each model.

---

## 7. Row simulation & legit data

**Source of truth = the event stream, not a parallel generator.** Do **not** extend `src/simulator/generate_dataset.py` — the critic confirmed it is schema-incompatible: it only knows ATO-01/MULE-01, invents its own attribute space (`payment_rail`, `mcc`, `ip_prefix`, `dist_from_home_km`…) that does **not** match the catalog's real attribute keys, and uses a chronological split that leaks kill-chain stages. Instead:

- **Derive one tabular row per `ObservedEvent`** already produced by `simulation.py::simulate_attack` and `simulate_legitimate_controls`, joined to its sealed `TruthRecord` by `event_id`. This guarantees rows and the LLM-Blue event space share one label-separated compiler + one safety gate, and automatically covers all 6 families.
- **Bulk batch builder** (`sentinelloop/dataset.py`): the live loop emits ~50% positive rows — useless for training. The builder (a) synthesises a large legit account population + ordinary events **in the catalog attribute space** with population-drawn per-account baseline amounts, (b) loops the 6 families × {easy,medium,hard} × many seeds through the existing `planner → safety → simulate_attack` path, (c) emits each family's `legitimate_controls` look-alikes, and (d) sizes the legit baseline so injected fraud lands at **`TARGET_FRAUD_SHARE ≈ 0.75%`**.
- **`fraud_contributing` flag:** extend `TruthRecord` (`contracts.py`) and stage templates with a per-stage `fraud_contributing:bool`. Genuinely benign stages (MULE `account_warmup`, SYNID `history_building`, ATO `novel_session`, victim-authorised context) should be **label=legit but keep `kill_chain_id`** — the sibling's "compromise rows" rule — so chains stay analysable without poisoning labels.
- **Coverage audit:** assert every `event_type` and every feature column carries **both** classes; expand `_control_shape` so every attack `event_type` (e.g. `SUPPLIER_PROFILE_CHANGED`, `IDENTITY_*`) has a benign counterpart, closing the disjoint-event-type leak.
- **One causal aggregator:** refactor `evidence.py` aggregate logic into a single causal velocity/graph function called by `dataset.py` over the merged per-account/per-chain stream, so legit and fraud rows get identical arithmetic.

---

## 8. Novel-attack generation — two-loop Red ("propose-then-admit")

Keep the current bounded path as the **inner loop** ("exploit known cards"): the LLM picks a catalog family, emphasises stages, and mutates numeric knobs inside `card.allowed_mutations`/`parameter_bounds` (`red_agent.py::_validate_plan`), all downstream closed to `attack_cards.json` + `safety.ALLOWED_EVENT_TYPES`.

Add an **outer loop** — a **Scenario Architect** (`sentinelloop/architect.py`) that emits a *candidate* full `AttackCard` (new `attack_family`, new `stage_templates`, `parameter_profiles`+`parameter_bounds`, `legitimate_controls`, mandatory `source_refs`). Novelty is admitted through an **expandable allowlist + gate**, never by relaxing `safety.py`:

1. **RAG grounding** (`sentinelloop/threat_intel.py`): index a fraud-typology corpus — the repo already cites the FinCEN deepfake alert + Mastercard GenAI report in each card's `source_refs`; extend with FATF, Europol IOCTA, UK-Finance APP-fraud, the Fed synthetic-identity white paper, and **MITRE ATLAS**. The Architect must retrieve ≥1 passage and attach `source_refs` (which `safety.py` already hard-requires).
2. **Quarantine + gate:** add `red_team_agent/candidate_cards.json`; `AttackCatalog` loads both, tagging `status='live'|'candidate'`. Only `live` cards are selectable by the inner loop. `safety.py` keeps `ALLOWED_EVENT_TYPES` as the **live** allowlist and adds a **signed `candidate_event_types`** set; a candidate must pass all existing checks **plus** every new parameter bounded, every new attribute key clearing forbidden fragments, `syn_` IDs, amount bounds.
3. **Promotion** (`promote.py` or `cli.py`): a **human or stricter deterministic gate** moves a candidate into `attack_cards.json` and its event types into the live allowlist (mirrors Microsoft PyRIT's human-in-the-loop; NIST AI 600-1 controlled-testing guidance).
4. **Quality-diversity search** (`sentinelloop/archive.py`): a MAP-Elites-style archive keyed on descriptors `(attack_family, targeted intervention_point, evasion class, novelty type)`, storing the best `ScenarioSpec` per cell by `referee.blue_score`. Add a **novelty bonus** (distance from existing cells) to avoid RL mode-collapse (Rainbow Teaming, Curiosity-driven Red-teaming).
5. **Autocurriculum** (`orchestrator.py`): treat Red as an exploiter population, Blue as the main agent (AlphaStar-style league). Use `RefereeReport.blue_score` / `value_prevented_ratio` / `hard_false_positive_rate` to pick next-round difficulty and to target Blue's mined weaknesses (`sentinelloop/weakness.py`: missed/contained outcomes, low-prevented intervention points, never-fired reason codes). Blue then adapts through the §6 retraining loop.

---

## 9. Phased implementation roadmap

**Phase 0 — Fix the simulator so any ML is trainable (blocking):**
1. Randomise `sender_baseline_amount_inr` (per-account, population-drawn); expose amount/baseline **ratio** (fixes L4).
2. Expand `simulate_legitimate_controls`/`_control_shape` into **full-length benign chains** per family; ensure every attack `event_type` has a benign counterpart (fixes L2/L3).
3. Add `fraud_contributing` to `TruthRecord` + stage templates; relabel benign stages legit-with-`kill_chain_id`.
4. Extend `_materialize_attributes` to roll any new `*_probability` field.

**Phase 1 — Rows + detector:**
5. `sentinelloop/dataset.py` (bulk builder from the event stream, ~0.75% imbalance).
6. `sentinelloop/blue_ml/features.py` (one causal `build_features`, allowlist + `audit_leakage`).
7. `sentinelloop/blue_ml/detector.py` (HistGBC + calibration) and `scripts/train_detector.py` (grouped split, stage-aware thresholds).
8. `sentinelloop/blue_ml/metrics.py` (chain recall @ budget, `caught_at_stage`, `value_prevented`, hard-FP/friction), reusing `referee.py` semantics.
9. Add **`scikit-learn` to `requirements.txt`** (it is currently in `requirements-legacy.txt` only — the deployed runtime has just pandas/Flask/gunicorn; the "no new dependency" assumption is false for v2).

**Phase 2 — Per-stage hybrid decisioning:**
10. Wire `ml_risk_score` into `blue_agent.investigate_event` as an `EvidencePacket`; add `session.py` cumulative `S_t`; `config.py` toggle `ML_DETECTOR_ENABLED` for LLM-only vs hybrid A/B.
11. Extend `contracts.BlueTurn.to_dict` to log `per_event_risk`, `cumulative_session_risk`, thresholds, model-artifact hash.

**Phase 3 — Retraining loop:**
12. `labeling.py` (append-only training log, all stages), `retrain.py` (full retrain + replay + champion/challenger gate), generation bookkeeping + drift trigger; **lift the `rounds` cap** in `orchestrator.py`.

**Phase 4 — Attribute expansion (incremental):** add §3 canonicals first, then §4 family-specifics **one at a time**, each shipped with its control-emission spec and a leakage test.

**Phase 5 — Novel-attack generation:** `threat_intel.py`, `architect.py`, `candidate_cards.json` + gate, `archive.py`, autocurriculum + `weakness.py`.

**Cross-cutting tests (`tests/`):** leakage audit (no forbidden key reaches FEATURES), determinism (same seed → same scores), training-log schema (every attack event has a matching `TruthRecord`; controls labelled), coverage audit (both classes per event_type/column).

---

## 10. Open questions / risks

- **Cross-institution observability:** many high-value CONTAIN signals (`receiver_inbound_sender_count`, `funds_dwell_time_seconds`, receiver KYC) are only visible to the **beneficiary's** bank, not the sender-side Blue. Decide whether the lab models shared-intelligence / network-level visibility, or restricts these to CONTAIN with an explicit "network data" assumption.
- **Calibration data volume:** each family has only ~3 `legitimate_controls` (1–2 events) per round — far below the ~100+ chains needed for a stable 1% FP budget. The Phase-0 control expansion + bulk builder is a **hard prerequisite**, not a nice-to-have.
- **Behavioural signals (L5):** hesitation/biometric/automation scores are the most leakage-prone and the least genuinely simulable. Prefer observable primitives; keep at most one consolidated score; consider dropping them until a principled generator exists.
- **Hybrid fusion vs hard floor:** does the ML score merely inform the LLM (recommended default) or can it deterministically force an action? The latter is a real change to the safety/policy model and needs its own review.
- **Determinism:** `Date.now`/RNG discipline — the bulk builder and any new randomness must be seed-threaded to preserve the repo's byte-identical-replay guarantee.

---

## 11. What we deliberately did **not** copy from the card repo
Card-present features (`entry_mode`, `avs_result`, `cvv_result`, `otp_status`, `dist_from_home_km`, `card_age_days`, `bin`, `mcc`, `merchant_country`, `vpn_flag`, `is_hosting_asn`, the `vf_*` card-velocity set, etc.) are **not** reused — they belong to a different fraud modality. What we import is the **method**: allowlist-not-denylist, `audit_leakage`, one causal post-hoc feature function, grouped split on the chain entity, chain-level metrics at an alert budget, sample-weight imbalance handling, and the discipline of deliberately decorrelating any signal (channel↔category, ASN↔fraud, amount↔label) that would otherwise be a "perfect marker created by an absence rather than an attack."

---

## 12. Sources (consolidated)

**Regulatory / industry:** RBI Master Direction on Digital Payment Security Controls (Id=12032); RBI V-CIP / KYC Master Direction (id=11566); NPCI UPI product overview; FinCEN advisories — money mules (FIN-2019-A003 / 2019 mule advisory), BEC (FIN-2019-A005), synthetic-identity white paper (2020), deepfake/GenAI alert; FBI IC3 BEC PSA (PSA220504); FATF digital-identity guidance; Europol IOCTA / money-muling; UK Finance Annual Fraud Report; UK PSR APP-scams work; US Federal Reserve synthetic-identity payments initiative; FFIEC authentication guidance; NIST AI 600-1 Generative AI Profile; MITRE ATLAS (atlas.mitre.org); Mastercard "Generative AI: Preparing Your Fraud Organization."

**Vendor / practitioner:** Feedzai (ATO), FICO Falcon, BioCatch (behavioural biometrics), SEON, Jumio, Experian (synthetic identity), Unit21, ComplyAdvantage.

**Academic / technical:** Rainbow Teaming (arXiv:2402.16822); Curiosity-driven Red-teaming (arXiv:2402.19464); Red Teaming LMs with LMs, Perez et al. (arXiv:2202.03286); graph-based mule detection (arXiv:2306.16424, arXiv:2205.13426); Microsoft PyRIT; AlphaStar league self-play; the Fraud-Detection Handbook (validation strategies, baseline features); scikit-learn docs (HistGradientBoostingClassifier, calibration, permutation importance, StratifiedGroupKFold); imbalanced-learn common pitfalls; River (online learning / ADWIN); Google Cloud MLOps continuous-delivery guidance; Feast point-in-time joins.

*(Per-attribute source URLs are retained in the workflow's structured research output.)*
