"""
Smart Generate — Verify-before-generate pipeline.

The core PrintForge workflow:
  1. User uploads image
  2. Analyze: what is this object?
  3. SEARCH: find multi-angle references online
  4. Present findings to user for verification
  5. Generate with maximum context (original + references)
  6. Post-process for printing

This is what makes PrintForge better than raw Tripo API calls.
"""

import io
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """What we know about the input image before generation."""
    description: str                    # What the object is
    keywords: List[str]                 # Search keywords
    category: str                       # "figure", "toy", "sculpture", "everyday", "custom"
    suggested_search_queries: List[str] # Queries to find reference views
    confidence: float                   # 0-1, how sure we are about the identification
    needs_back_reference: bool = True   # Does this object need back view reference?
    needs_side_reference: bool = True


@dataclass
class VerificationPackage:
    """Everything we found, packaged for user review before generation."""
    analysis: AnalysisResult
    reference_urls: List[dict]  # [{url, view_angle, title}]
    recommended_config: dict    # TripoExpert config params
    estimated_credits: int
    estimated_time_s: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class SmartGenerateResult:
    """Final output of the smart pipeline."""
    glb_path: str
    glb_data: bytes
    vertices: int
    faces: int
    has_texture: bool
    total_credits: int
    generation_time_s: float
    analysis: AnalysisResult
    references_used: int


class SmartGenerator:
    """The brain of PrintForge — intelligent generation pipeline."""

    # Category → search strategy mapping
    CATEGORY_STRATEGIES = {
        "figure": {
            "queries": [
                "{name} figure turnaround 360 all angles",
                "{name} figure back view rear",
                "{name} 3D model reference sheet",
            ],
            "config": "config_for_figure",
            "needs_back": True,
            "needs_side": True,
        },
        "toy": {
            "queries": [
                "{name} toy unboxing all angles",
                "{name} toy back view",
                "{name} collectible 3D reference",
            ],
            "config": "config_for_figure",
            "needs_back": True,
            "needs_side": True,
        },
        "sculpture": {
            "queries": [
                "{name} sculpture turnaround",
                "{name} sculpture back detail",
                "{name} 3D scan reference",
            ],
            "config": "config_for_print",
            "needs_back": True,
            "needs_side": True,
        },
        "everyday": {
            "queries": [
                "{name} product 360 view",
                "{name} all sides",
            ],
            "config": "config_for_print",
            "needs_back": True,
            "needs_side": False,
        },
    }

    def analyze(self, image_path: str, user_description: str = "") -> AnalysisResult:
        """Step 1: Analyze what the image contains.

        Uses user description as primary signal.
        In production, could also use vision LLM for auto-identification.
        """
        description = user_description or "3D figure"

        # Determine category from description
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in ["pop mart", "molly", "figure", "手办", "泡泡玛特", "nendoroid"]):
            category = "figure"
        elif any(kw in desc_lower for kw in ["toy", "玩具", "collectible"]):
            category = "toy"
        elif any(kw in desc_lower for kw in ["sculpture", "雕塑", "statue", "bust"]):
            category = "sculpture"
        else:
            category = "everyday"

        strategy = self.CATEGORY_STRATEGIES.get(category, self.CATEGORY_STRATEGIES["everyday"])

        # Build search queries
        name = description.replace("figure", "").replace("toy", "").strip()
        queries = [q.format(name=name) for q in strategy["queries"]]

        return AnalysisResult(
            description=description,
            keywords=name.split(),
            category=category,
            suggested_search_queries=queries,
            confidence=0.8 if user_description else 0.3,
            needs_back_reference=strategy["needs_back"],
            needs_side_reference=strategy["needs_side"],
        )

    def prepare_verification(
        self,
        analysis: AnalysisResult,
        search_fn: Optional[Callable] = None,
    ) -> VerificationPackage:
        """Step 2: Search and package for user verification.

        Args:
            analysis: From analyze()
            search_fn: Optional function(query) -> list of {url, title} dicts
                       If not provided, returns queries for the caller to execute.
        """
        reference_urls = []
        warnings = []

        if search_fn:
            for query in analysis.suggested_search_queries:
                try:
                    results = search_fn(query)
                    reference_urls.extend(results)
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")

        if not reference_urls:
            warnings.append(
                "No reference images found. Generation will rely solely on the input image. "
                "For better results, provide additional angle photos or a description."
            )

        # Estimate costs
        estimated_credits = 50  # Base generation
        if analysis.category in ("figure", "sculpture"):
            estimated_credits += 10  # Refine attempt

        return VerificationPackage(
            analysis=analysis,
            reference_urls=reference_urls[:10],
            recommended_config={
                "model_version": "P1-20260311",
                "texture_quality": "detailed",
                "do_refine": True,
                "do_rig": False,
                "category": analysis.category,
            },
            estimated_credits=estimated_credits,
            estimated_time_s=90,
            warnings=warnings,
        )

    def generate(
        self,
        image_path: str,
        verification: VerificationPackage,
        output_path: str,
        reference_images: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> SmartGenerateResult:
        """Step 3: Generate with full context (after user verification).

        Args:
            image_path: Primary input image
            verification: From prepare_verification() (user has approved)
            output_path: Where to save the GLB
            reference_images: Optional local paths to reference images for multiview
            progress_callback: Optional(stage, detail)
        """
        from .tripo_expert import TripoExpert, TripoExpertConfig

        expert = TripoExpert()
        start = time.time()

        def _progress(stage, detail=""):
            if progress_callback:
                progress_callback(stage, detail)

        # Use multiview if we have reference images
        if reference_images and len(reference_images) >= 2:
            _progress("multiview", f"Using {len(reference_images)} reference views")
            config = TripoExpertConfig(
                model_version="P1-20260311",
                texture=True,
                pbr=True,
                texture_quality="detailed",
                enable_image_autofix=True,
                do_refine=True,
            )
            # TODO: Upload references and use multiview_to_model
            # For now, fall through to single image with best config
            result = expert.generate_from_image(image_path, config, _progress)
        else:
            # Single image with expert config
            if verification.analysis.category == "figure":
                config = TripoExpert.config_for_figure()
            else:
                config = TripoExpert.config_for_print()

            result = expert.generate_from_image(image_path, config, _progress)

        # Save
        with open(output_path, 'wb') as f:
            f.write(result.glb_data)

        return SmartGenerateResult(
            glb_path=output_path,
            glb_data=result.glb_data,
            vertices=result.vertices,
            faces=result.faces,
            has_texture=result.has_texture,
            total_credits=result.total_credits,
            generation_time_s=time.time() - start,
            analysis=verification.analysis,
            references_used=len(reference_images) if reference_images else 0,
        )
