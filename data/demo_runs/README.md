# Precomputed public-demo runs

Each JSON file in this directory is a complete recorded synthetic Agent Arena run.
With `DEMO_MODE=precomputed`, the app selects an artifact matching the chosen attack
family and difficulty. It never calls an LLM or external API. A recorded multi-round
artifact serves its complete run and shorter prefixes. If only a one-round artifact
is available, the UI can show up to five explicitly non-adaptive replay cycles.

Only commit synthetic, safe-to-disclose artifacts. Preserve `model_configuration` so
the UI accurately identifies the model that generated each recorded trace.

Use `python scripts/curate_demo_run.py SOURCE.json DESTINATION.json` to validate and
copy a recorded run into this catalog.

The committed `*-5r-claude-opus-48.json` catalog contains true recorded cross-round
adaptation for all nine attack families at easy, medium and hard difficulty: 27 runs
and 135 rounds. The selector automatically prefers these recordings for requests of
one to five rounds. Public demo mode needs no model credentials or inference service.

To regenerate the catalog from a Claude-compatible endpoint, supply credentials only
through the environment and run:

```bash
MODEL_BASE_URL="https://provider.example/v1" \
MODEL_API_KEY="..." \
RED_MODEL_ID="Claude-Opus-4.8" \
BLUE_MODEL_ID="Claude-Opus-4.8" \
MODEL_STRUCTURED_OUTPUT_MODE=prompt \
MODEL_REASONING_EFFORT=high \
MODEL_MAX_OUTPUT_TOKENS=4096 \
CASE_PARALLELISM=4 \
python scripts/generate_demo_catalog.py \
  --rounds 5 \
  --prompt-profile claude \
  --artifact-label claude-opus-48
```

The generator resumes safely by skipping valid existing files. Its optional
`--families`, `--difficulties` and `--mars-auth-source` arguments support partitioned
generation and local MARS authentication; generated artifacts never contain the
provider token.
