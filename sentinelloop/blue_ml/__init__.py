"""Blue-only, leakage-controlled payment detector and champion/challenger learning loop.

The detector supplies causal risk evidence to the Blue GenAI investigator. It never receives
sealed truth at inference and no detector details are exposed to the Red agent.
"""
