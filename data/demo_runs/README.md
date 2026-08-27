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

To replace replay cycles with true recorded cross-round adaptation, generate five-round
artifacts with `python scripts/generate_demo_catalog.py --rounds 5`; the selector will
prefer them automatically for requests of two to five rounds.
