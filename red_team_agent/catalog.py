"""Load and query the curated, source-backed attack-card catalogue."""

from __future__ import annotations

import json
from pathlib import Path

from .models import AttackCard


DEFAULT_CATALOG_PATH = Path(__file__).with_name("attack_cards.json")


class AttackCatalog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CATALOG_PATH
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        cards = [AttackCard(**item) for item in payload["cards"]]
        self._cards = {card.attack_family: card for card in cards}
        if len(self._cards) != len(cards):
            raise ValueError("Attack catalogue contains duplicate attack_family values.")

    def list(self) -> list[AttackCard]:
        return sorted(self._cards.values(), key=lambda card: card.attack_family)

    def get(self, attack_family: str) -> AttackCard:
        try:
            return self._cards[attack_family]
        except KeyError as exc:
            available = ", ".join(sorted(self._cards))
            raise KeyError(f"Unknown attack family {attack_family!r}. Available: {available}") from exc

    @property
    def families(self) -> list[str]:
        return sorted(self._cards)
