"""Source-backed threat-landscape catalogue for the Identify pillar."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from red_team_agent.catalog import AttackCatalog


DEFAULT_ATLAS_PATH = Path(__file__).resolve().parents[1] / "red_team_agent" / "threat_atlas.json"
LIFECYCLE_PHASES = {"pre_transaction", "transaction", "post_transaction"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}


@dataclass(frozen=True)
class ThreatSource:
    id: str
    publisher: str
    title: str
    url: str
    evidence: str


@dataclass(frozen=True)
class ThreatVector:
    id: str
    name: str
    simulation_family: str
    category: str
    rails: list[str]
    channels: list[str]
    social_surfaces: list[str]
    lifecycle_phases: list[str]
    genai_capabilities: list[str]
    defensive_observables: list[str]
    source_ids: list[str]
    plausibility: int
    novelty: int
    research_confidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThreatAtlas:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_ATLAS_PATH
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.schema_version = str(payload["schema_version"])
        self.reviewed_at = str(payload["reviewed_at"])
        self.scope_note = str(payload["scope_note"])
        self.sources = [ThreatSource(**item) for item in payload["sources"]]
        self.vectors = [ThreatVector(**item) for item in payload["vectors"]]
        self.novel_holdout_vector_ids = set(payload.get("novel_holdout_vector_ids", []))
        self._validate()

    def _validate(self) -> None:
        source_ids = [source.id for source in self.sources]
        vector_ids = [vector.id for vector in self.vectors]
        errors: list[str] = []
        if len(source_ids) != len(set(source_ids)):
            errors.append("Threat Atlas source identifiers must be unique.")
        if len(vector_ids) != len(set(vector_ids)):
            errors.append("Threat vector identifiers must be unique.")
        known_sources = set(source_ids)
        known_families = set(AttackCatalog().families)
        for vector in self.vectors:
            if vector.simulation_family not in known_families:
                errors.append(
                    f"{vector.id} maps to unknown simulation family {vector.simulation_family}."
                )
            if not vector.source_ids or set(vector.source_ids).difference(known_sources):
                errors.append(f"{vector.id} has missing or unknown sources.")
            if not 1 <= vector.plausibility <= 5 or not 1 <= vector.novelty <= 5:
                errors.append(f"{vector.id} plausibility and novelty must be between 1 and 5.")
            if vector.research_confidence not in CONFIDENCE_LEVELS:
                errors.append(f"{vector.id} has invalid research confidence.")
            if not vector.rails or not vector.channels or not vector.genai_capabilities:
                errors.append(f"{vector.id} lacks required coverage dimensions.")
            if set(vector.lifecycle_phases).difference(LIFECYCLE_PHASES):
                errors.append(f"{vector.id} contains an unknown lifecycle phase.")
        unknown_holdouts = self.novel_holdout_vector_ids.difference(vector_ids)
        if unknown_holdouts:
            errors.append(f"Novel holdout vectors are unknown: {sorted(unknown_holdouts)}")
        families_without_vectors = known_families.difference(
            vector.simulation_family for vector in self.vectors
        )
        if families_without_vectors:
            errors.append(
                f"Implemented attack families lack Threat Atlas vectors: {sorted(families_without_vectors)}"
            )
        if errors:
            raise ValueError(f"Threat Atlas validation failed: {errors}")

    def source_by_id(self) -> dict[str, ThreatSource]:
        return {source.id: source for source in self.sources}

    def vectors_for_family(self, family: str) -> list[ThreatVector]:
        return [vector for vector in self.vectors if vector.simulation_family == family]

    def summary(self) -> dict[str, Any]:
        families = sorted({vector.simulation_family for vector in self.vectors})
        rails = sorted({item for vector in self.vectors for item in vector.rails})
        channels = sorted({item for vector in self.vectors for item in vector.channels})
        surfaces = sorted(
            {item for vector in self.vectors for item in vector.social_surfaces if item != "none"}
        )
        capabilities = sorted(
            {item for vector in self.vectors for item in vector.genai_capabilities}
        )
        phases = sorted({item for vector in self.vectors for item in vector.lifecycle_phases})
        high_confidence = sum(
            vector.research_confidence == "high" for vector in self.vectors
        )
        return {
            "schema_version": self.schema_version,
            "reviewed_at": self.reviewed_at,
            "vector_count": len(self.vectors),
            "simulation_ready_vector_count": len(self.vectors),
            "attack_family_count": len(families),
            "source_count": len(self.sources),
            "rail_count": len(rails),
            "channel_count": len(channels),
            "social_surface_count": len(surfaces),
            "genai_capability_count": len(capabilities),
            "lifecycle_phase_count": len(phases),
            "high_confidence_vector_rate": round(
                high_confidence / len(self.vectors) if self.vectors else 0.0, 4
            ),
            "novel_holdout_vector_count": len(self.novel_holdout_vector_ids),
            "attack_families": families,
            "rails": rails,
            "channels": channels,
            "social_surfaces": surfaces,
            "genai_capabilities": capabilities,
            "lifecycle_phases": phases,
        }

    def to_dict(self) -> dict[str, Any]:
        source_map = self.source_by_id()
        vectors = []
        for vector in self.vectors:
            item = vector.to_dict()
            item["novel_holdout"] = vector.id in self.novel_holdout_vector_ids
            item["sources"] = [asdict(source_map[source_id]) for source_id in vector.source_ids]
            vectors.append(item)
        return {
            "summary": self.summary(),
            "scope_note": self.scope_note,
            "sources": [asdict(source) for source in self.sources],
            "vectors": vectors,
        }
