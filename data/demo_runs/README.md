# Precomputed public-demo runs

Each JSON file in this directory is a complete recorded synthetic Agent Arena run.
With `DEMO_MODE=precomputed`, the app selects an artifact matching the chosen attack
family, difficulty, and number of rounds. It never calls an LLM or external API.

Only commit synthetic, safe-to-disclose artifacts. Preserve `model_configuration` so
the UI accurately identifies the model that generated each recorded trace.

Use `python scripts/curate_demo_run.py SOURCE.json DESTINATION.json` to validate and
copy a recorded run into this catalog.
