# Merge Execution Plan — `Saral-redteam` (ML) into `review-masterguard` (MasterGuard)

**Strategy: theirs-as-trunk.** MasterGuard's changes are pervasive (a `lifecycle_phase` reframing
across contracts/referee/blue/red + 3 new families + evaluation/validation/population modules).
Your ML work is mostly **additive** (self-contained `blue_ml/`, `dataset.py`, `scripts/`). So we
take MasterGuard as the base and graft the ML layer on top.

Trial `git merge-tree review-masterguard Saral-redteam` result: **only 5 content conflicts**
(`config.py`, `red_agent.py`, `simulation.py`, `blue_agent.py`, `orchestrator.py`). Everything else
— including `contracts.py`, `attack_cards.json`, `referee.py`, `requirements.txt` — **auto-merges**.
But auto-merge ≠ correct: two semantic breakages must be fixed by hand (see Phase D/E).

Scratch branch: `merge-attempt` (already created from `review-masterguard`).

---

## Phase A — Prep (unblock the merge)
1. **Locked CSVs block the working-tree merge.** `data/loop/big*.csv` are untracked and locked by
   OneDrive (`Invalid argument` on unlink). Before `git merge`: close Excel/OneDrive, move
   `data/loop/big*.csv` out of the tree, or `git rm --cached` any tracked ones.
2. Add `.gitignore` for `data/loop/**` binaries (`*.csv`, `*.joblib`, `*.jsonl`) so model/data
   artifacts never enter the merge again. Both branches committed such artifacts; keep them out.
3. `git checkout merge-attempt` and run `git merge --no-ff Saral-redteam`.

## Phase B — Mechanical conflicts (keep both sides)
**`sentinelloop/config.py`** (2 blocks): union the fields.
- dataclass: keep `case_parallelism` **and** `ml_detector_enabled`, `ml_model_dir`.
- `from_env`: keep `case_parallelism=_int_env(...)` **and** `ml_detector_enabled=...`, `ml_model_dir=...`.
- `validate()` (their `case_parallelism` check) already auto-merged — leave it.

**`sentinelloop/red_agent.py`** (1 block) + caller: combine both validators.
- Keep your 3-tuple return `(card, overrides, dropped)`.
- Fold their focus-tie rule (mutations must belong to focus stages) into the **soft-drop**:
  append `f"{p} (unrelated to focus)"` to `dropped` instead of raising — preserves your
  no-crash behavior while honoring their constraint.
- **Fix the caller in `plan()`**: ensure it unpacks 3 values (`_, overrides, dropped = self._validate_plan(...)`)
  and keep your `red.plan.mutations_dropped` trace. Verify no leftover 2-tuple unpack.

## Phase C — Structural conflicts (hand-rewrite)
**`sentinelloop/simulation.py`** (3 blocks): merge the event taxonomies + your leakage logic.
- Keep their `PRE_/POST_/PAYMENT_EVENT_TYPES` sets (with agentic/payout/dispute types),
  `event_lifecycle_phase()`, `event_delivery_profile()`, and the 3-new-family handling.
- Re-add your `VALUE_EVENT_TYPES`, `BENIGN_STAGE_IDS`, `account_baseline_inr()`, the generic
  `*_probability`→flag rolling + auth-field stamping in `_materialize_attributes`, the canonical
  overlap draws in `_benignize`, and the **multi-event ambient sessions** + traps.
- In `simulate_attack`: stamp **both** `lifecycle_phase` on each `ObservedEvent` (theirs) **and**
  `fraud_contributing` on each `TruthRecord` (yours).

**`sentinelloop/blue_agent.py`** (2 blocks) — the hardest: their Blue is a **single combined
event-response** (`BLUE_EVENT_RESPONSE_SCHEMA`, one LLM call), yours is a **two-call** flow
(investigator → workbench → decider) and your `_ml_evidence` injection assumes two calls.
- Adopt **their** single-response flow as the base.
- Re-implement the ML graft for it: after evidence gathering and **before** their decision call,
  call `_ml_evidence(event, visible_history, prior_turns)`, append the `ml_risk_score`
  `EvidencePacket` to `evidence`, and add the ml-risk note to the payload/policy.
- Imports: keep **both** `DefensePlaybook` and `EvidencePacket`.
- Keep `_load_detector`, `reload_detector`, `_ml_evidence`, and set `ml_risk=ml_risk_info` on their
  `BlueTurn` (the auto-merged `contracts.BlueTurn` already carries `ml_risk`).
- Adapt `_ml_evidence` to their richer `EvidencePacket(source, as_of_event_id, confidence)` signature.

**`sentinelloop/orchestrator.py`** (5 blocks): reconcile parallelism + playbook + ambient + retrain.
- Keep their `import time`, `ThreadPoolExecutor`, and `RoundResult` fields
  (`active_blue_playbook`, `blue_adaptation`, `submission_evaluation`, `duration_ms`).
- Add back your `ambient_results` field, the ambient/trap evaluation pass, the `training_log_path`
  logging, and the `retrain_every` generation trigger + `self.blue.reload_detector()` hot-reload.
- Ensure the `RoundResult(...)` constructor passes **all** fields from both sides.
- Merge the control loop: keep their (possibly parallel) control execution, then run your ambient pass.

## Phase D — Propagate the new required `ObservedEvent.lifecycle_phase` (HIDDEN BREAKAGE)
Their `ObservedEvent` gained a **required** `lifecycle_phase: str` field. Every `ObservedEvent(...)`
constructed in your code omits it and will raise `TypeError`. Fix all sites:
- `simulation.py`: `_standalone_case`, ambient session builder → pass
  `lifecycle_phase=event_lifecycle_phase(event_type)`.
- `dataset.py`: any direct `ObservedEvent(...)` (ambient/trap) → same.
- `blue_ml/feature_frame.py::_event_dict`: include a `"lifecycle_phase"` key in the reconstructed
  event dict (read from the row; default via `event_lifecycle_phase`).
- Grep the whole tree: `ObservedEvent(` — every call must supply `lifecycle_phase`.

## Phase E — Adapt the ML feature pipeline to the merged contracts
- **`blue_ml/features.py`**: decide `lifecycle_phase`'s role — add it to `CATEGORICAL_FEATURES`
  (likely useful, both classes span all phases so not a presence-leak) **or** add to `FINGERPRINTS`
  to exclude. Re-add `fraud_contributing` to the structural/forbidden set.
- **`blue_ml/feature_frame.py`**: add `lifecycle_phase` + `fraud_contributing` to the `STRUCTURAL`
  set so they're not treated as model attributes; keep `label = is_attack AND fraud_contributing`.
- **`dataset.py` vs `population.py`**: both survive the merge. Decide the training source:
  - *Low-risk:* keep `dataset.py` (already leak-audited), just fix its `ObservedEvent` calls.
  - *Cleaner (later):* retarget `feature_frame` to consume `population.py` output.
- Re-run `scripts/train_detector.py` to regenerate a champion under the merged contracts.

## Phase F — Tests (make every test green)
- **Restore/keep yours** (mine-only additions, survive merge): `tests/test_detector.py`,
  `tests/test_canonical_attributes.py`, `tests/test_retrain.py` — update for `lifecycle_phase` +
  `fraud_contributing`.
- **Keep theirs**: `tests/test_external_validation.py`, `tests/test_population_benchmark.py`.
- **`tests/test_agentic_loop.py`** auto-merged but both sides changed it — run it; it is authoritative
  for the new Blue. Make your ML additions (default OFF) satisfy it; verify the fake-gateway stub
  matches their `BLUE_EVENT_RESPONSE_SCHEMA` shape.
- **`tests/test_dashboard.py`** needs `flask` + `data/processed/*` (pre-existing failure on both
  branches) — install flask or skip; not a merge regression.
- Iterate: `python -m unittest discover -s tests` and `... -s red_team_agent/tests` until green.

## Phase G — Housekeeping
- `requirements.txt`: confirm the union — your `scikit-learn`, `joblib` + their audit/validation
  requirement files (auto-merged; verify).
- Docs: keep both — their `benchmark-methodology`/`external-validation`/`whitepaper` + your
  `blue-team-ml-evolution-research.md` (was deleted on their side; re-add it).
- `.gitignore` the data/model artifacts (Phase A).

## Phase H — Final validation (safe + complete)
1. `python -m unittest discover -s tests` → all green (except the known flask/dashboard).
2. `python -m unittest discover -s red_team_agent/tests` → green.
3. `python scripts/train_detector.py --seed 42 --seeds-per-cell 8` → champion trains under merged contracts.
4. Fake-gateway agentic-loop smoke (ML off) → unchanged behavior; then ML on → `ml_risk` logged.
5. Their `evaluation`/`external_validation`/`benchmark` entrypoints run without import errors.
6. `git grep "ObservedEvent(" ` → every call supplies `lifecycle_phase`.

---

## Risk register (highest first)
1. **`blue_agent.py` single-vs-two-call flow** — the ML injection must be rewritten for their
   combined-response Blue; the biggest hand-merge.
2. **`ObservedEvent.lifecycle_phase` required field** — breaks every ML-side `ObservedEvent(...)`
   until propagated (Phase D). Silent `TypeError` at dataset/ambient build time.
3. **`orchestrator.py`** — parallel execution + playbook/adaptation (theirs) vs ambient + retrain
   (yours); `RoundResult` must carry all fields.
4. **`dataset.py` vs `population.py` overlap** — pick one training source; don't run both blindly.
5. **`attack_cards.json` auto-merged** — verify JSON validity + that your canonicals landed on the
   right (of 9) cards and `allowed_mutations` is intact; safety gate must pass all family×difficulty.
6. **Data/model artifacts** — keep out of git (Phase A) to avoid repeat lock/merge failures.

## Bottom line
Auto-merge does ~90% of the file set. Real work = **5 conflicts + 1 propagation (lifecycle_phase)
+ 1 structural re-graft (Blue ML injection)**. Order: A → B → C → D → E → F → G → H. Estimated
focused effort ~½–1 day; the Blue re-graft and lifecycle propagation are the critical path.
