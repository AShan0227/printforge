"""Competitor Monitor: Track pricing, features, and versions of competing services."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default storage path
DEFAULT_DATA_PATH = Path.home() / ".printforge" / "competitors.json"


@dataclass
class CompetitorUpdate:
    """A single change detected for a competitor."""
    competitor: str
    category: str    # "pricing", "features", "version", "status"
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    detected_at: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()


@dataclass
class CompetitorInfo:
    """Snapshot of a competitor's current state."""
    name: str
    url: str
    pricing: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    version: str = ""
    last_checked: str = ""
    notes: str = ""


# Hardcoded competitor data (structure for future web scraping)
_COMPETITOR_DATA: Dict[str, CompetitorInfo] = {
    "meshy": CompetitorInfo(
        name="Meshy.ai",
        url="https://www.meshy.ai",
        pricing={
            "free": "200 credits/month",
            "pro": "$20/month — 1000 credits",
            "max": "$60/month — unlimited",
        },
        features=[
            "Text to 3D",
            "Image to 3D",
            "AI texturing",
            "PBR materials",
            "GLB/FBX/OBJ export",
            "API access",
        ],
        version="3.0",
        notes="Strongest consumer brand. Fast iteration on features.",
    ),
    "tripo": CompetitorInfo(
        name="Tripo AI",
        url="https://www.tripo3d.ai",
        pricing={
            "free": "Limited generations",
            "pro": "$10/month",
            "enterprise": "Custom pricing",
        },
        features=[
            "Image to 3D",
            "Text to 3D",
            "Multi-view reconstruction",
            "High-poly output",
            "API access",
        ],
        version="2.0",
        notes="Competitive pricing. Good quality for the price.",
    ),
    "hitem3d": CompetitorInfo(
        name="Hitem3D",
        url="https://hitem3d.com",
        pricing={
            "free": "5 generations/day",
            "pro": "$15/month",
        },
        features=[
            "Image to 3D",
            "Batch processing",
            "STL/OBJ export",
            "Print optimization",
        ],
        version="1.5",
        notes="Niche player focused on 3D printing output.",
    ),
    "makerworld": CompetitorInfo(
        name="MakerWorld",
        url="https://makerworld.com",
        pricing={
            "free": "Free community platform",
            "designer_rewards": "Revenue sharing for popular models",
        },
        features=[
            "Model marketplace",
            "Community sharing",
            "Bambu Lab integration",
            "Direct print from browser",
            "AI model suggestions",
        ],
        version="N/A",
        notes="Bambu Lab's model platform. Not direct competitor but adjacent.",
    ),
}


class CompetitorMonitor:
    """Track competitor updates and store results locally.

    Currently uses hardcoded data. Structure is ready for future
    web scraping integration.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DEFAULT_DATA_PATH
        self._previous = self._load_previous()

    def check_updates(self) -> List[CompetitorUpdate]:
        """Check for changes compared to last stored state.

        Returns:
            List of CompetitorUpdate objects describing changes.
        """
        updates: List[CompetitorUpdate] = []
        current = _COMPETITOR_DATA

        for key, competitor in current.items():
            prev = self._previous.get(key)
            if prev is None:
                updates.append(CompetitorUpdate(
                    competitor=competitor.name,
                    category="status",
                    description=f"New competitor tracked: {competitor.name}",
                    new_value=competitor.url,
                ))
                continue

            # Check pricing changes
            if competitor.pricing != prev.get("pricing", {}):
                updates.append(CompetitorUpdate(
                    competitor=competitor.name,
                    category="pricing",
                    description=f"Pricing changed for {competitor.name}",
                    old_value=json.dumps(prev.get("pricing", {})),
                    new_value=json.dumps(competitor.pricing),
                ))

            # Check feature changes
            prev_features = set(prev.get("features", []))
            curr_features = set(competitor.features)
            new_features = curr_features - prev_features
            if new_features:
                updates.append(CompetitorUpdate(
                    competitor=competitor.name,
                    category="features",
                    description=f"New features: {', '.join(new_features)}",
                    new_value=str(list(new_features)),
                ))

            # Check version changes
            if competitor.version != prev.get("version", ""):
                updates.append(CompetitorUpdate(
                    competitor=competitor.name,
                    category="version",
                    description=f"Version update: {prev.get('version', '?')} → {competitor.version}",
                    old_value=prev.get("version", ""),
                    new_value=competitor.version,
                ))

        # Save current state
        self._save_current(current)

        logger.info("Competitor check: %d updates found", len(updates))
        return updates

    def get_competitors(self) -> Dict[str, CompetitorInfo]:
        """Return current competitor data."""
        return dict(_COMPETITOR_DATA)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for CLI/API output."""
        competitors = []
        for key, info in _COMPETITOR_DATA.items():
            competitors.append({
                "name": info.name,
                "url": info.url,
                "pricing": info.pricing,
                "features": info.features,
                "version": info.version,
                "notes": info.notes,
            })
        return {
            "competitors": competitors,
            "total": len(competitors),
            "last_checked": datetime.now().isoformat(),
        }

    def _load_previous(self) -> Dict[str, Any]:
        """Load previously stored competitor data."""
        if not self.data_path.exists():
            return {}
        try:
            with open(self.data_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_current(self, data: Dict[str, CompetitorInfo]):
        """Save current competitor data to JSON."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for key, info in data.items():
            serializable[key] = {
                "name": info.name,
                "url": info.url,
                "pricing": info.pricing,
                "features": info.features,
                "version": info.version,
                "last_checked": datetime.now().isoformat(),
                "notes": info.notes,
            }
        with open(self.data_path, "w") as f:
            json.dump(serializable, f, indent=2)
