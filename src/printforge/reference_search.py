"""
Reference Image Search — Find multi-angle views of objects before 3D generation.

Core principle: don't guess what the back looks like, SEARCH for it.

Pipeline:
  1. Analyze input image → extract keywords (object type, brand, character name)
  2. Search web for multi-angle / turnaround / 360 views
  3. Filter & rank results by relevance
  4. Return reference images for user verification before generation
"""

import io
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class ReferenceImage:
    """A reference image found via search."""
    url: str
    title: str
    source: str  # "google", "bing", "searxng"
    relevance_score: float  # 0-1
    view_angle: str = "unknown"  # "front", "back", "side", "3/4", "turnaround"
    local_path: Optional[str] = None  # Downloaded path


@dataclass
class SearchResult:
    """Complete search result with analysis."""
    query_used: str
    object_description: str
    references: List[ReferenceImage] = field(default_factory=list)
    turnaround_found: bool = False
    back_view_found: bool = False
    side_views_found: int = 0


class ReferenceSearcher:
    """Search for reference images to improve 3D generation quality."""

    # View angle keywords for classification
    BACK_KEYWORDS = ["back", "rear", "behind", "背面", "背后", "后面", "backside"]
    SIDE_KEYWORDS = ["side", "left", "right", "lateral", "侧面", "左侧", "右侧", "profile"]
    TURNAROUND_KEYWORDS = ["turnaround", "360", "all angles", "all sides", "reference sheet",
                           "多角度", "全方位", "转面", "turn around", "model sheet"]
    THREE_QUARTER_KEYWORDS = ["3/4", "three quarter", "角度", "perspective"]

    def __init__(self, searxng_url: str = "http://localhost:8888"):
        self.searxng_url = searxng_url

    def analyze_and_search(self, image, description: str = "") -> SearchResult:
        """Analyze image content and search for reference views.

        Args:
            image: PIL.Image or file path
            description: Optional user-provided description to improve search

        Returns:
            SearchResult with found reference images
        """
        # Step 1: Build search query
        keywords = self._analyze_image_content(image, description)
        logger.info(f"Image analysis keywords: {keywords}")

        # Step 2: Search for multi-angle references
        references = []

        # Search 1: Turnaround / reference sheet
        turnaround_query = f"{keywords} turnaround 360 reference sheet all angles"
        refs_turnaround = self._search_images(turnaround_query, num=5)
        for ref in refs_turnaround:
            ref.relevance_score *= 1.2  # Boost turnarounds
            ref.view_angle = self._classify_view(ref.title, ref.url)
        references.extend(refs_turnaround)

        # Search 2: Back view specifically
        back_query = f"{keywords} back view rear behind 背面"
        refs_back = self._search_images(back_query, num=5)
        for ref in refs_back:
            ref.view_angle = "back"
        references.extend(refs_back)

        # Search 3: Side views
        side_query = f"{keywords} side view profile 侧面"
        refs_side = self._search_images(side_query, num=3)
        for ref in refs_side:
            ref.view_angle = "side"
        references.extend(refs_side)

        # Deduplicate and sort by relevance
        seen_urls = set()
        unique_refs = []
        for ref in sorted(references, key=lambda r: r.relevance_score, reverse=True):
            if ref.url not in seen_urls:
                seen_urls.add(ref.url)
                unique_refs.append(ref)

        result = SearchResult(
            query_used=keywords,
            object_description=keywords,
            references=unique_refs[:10],  # Top 10
            turnaround_found=any(r.view_angle == "turnaround" for r in unique_refs),
            back_view_found=any(r.view_angle == "back" for r in unique_refs),
            side_views_found=sum(1 for r in unique_refs if r.view_angle == "side"),
        )

        logger.info(
            f"Search complete: {len(result.references)} refs, "
            f"turnaround={result.turnaround_found}, "
            f"back={result.back_view_found}, "
            f"sides={result.side_views_found}"
        )
        return result

    def download_references(
        self, result: SearchResult, output_dir: str, max_images: int = 5
    ) -> List[str]:
        """Download top reference images to local directory.

        Returns list of local file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        for i, ref in enumerate(result.references[:max_images]):
            try:
                resp = requests.get(ref.url, timeout=15, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                })
                if resp.status_code != 200:
                    continue

                # Detect format
                content_type = resp.headers.get("content-type", "")
                ext = ".jpg"
                if "png" in content_type:
                    ext = ".png"
                elif "webp" in content_type:
                    ext = ".webp"

                filename = f"ref_{i:02d}_{ref.view_angle}{ext}"
                path = os.path.join(output_dir, filename)
                with open(path, "wb") as f:
                    f.write(resp.content)

                ref.local_path = path
                paths.append(path)
                logger.info(f"Downloaded: {filename} ({ref.view_angle})")

            except Exception as e:
                logger.warning(f"Failed to download {ref.url}: {e}")
                continue

        return paths

    def _analyze_image_content(self, image, description: str = "") -> str:
        """Extract searchable keywords from image + user description.

        Uses description if provided, otherwise falls back to basic analysis.
        """
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            image = PILImage.open(str(image))

        keywords = []

        # User description is primary signal
        if description:
            keywords.append(description)

        # Try to use LLM for image understanding via web search
        # (This is the "smart" part — we describe what we see)
        if not description:
            # Basic heuristics from image properties
            w, h = image.size

            # Check dominant colors for hints
            colors = image.resize((1, 1)).getpixel((0, 0))
            if isinstance(colors, tuple) and len(colors) >= 3:
                r, g, b = colors[:3]
                # Very rough content hints
                if r > 200 and g < 100 and b < 100:
                    keywords.append("red figure")
                elif g > 200:
                    keywords.append("green object")

            # Default: generic 3D figure search
            if not keywords:
                keywords.append("3D figure character")

        # Add standard suffixes for 3D reference search
        base = " ".join(keywords)
        return base

    def _search_images(self, query: str, num: int = 5) -> List[ReferenceImage]:
        """Search for images using SearXNG (local) or web APIs."""
        results = []

        # Try SearXNG first (local, no rate limits)
        try:
            results = self._search_searxng(query, num)
            if results:
                return results
        except Exception as e:
            logger.info(f"SearXNG unavailable: {e}")

        # Fallback: Google Custom Search or Bing (if API keys available)
        # For now, try a simple DuckDuckGo scrape approach
        try:
            results = self._search_duckduckgo(query, num)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        return results

    def _search_searxng(self, query: str, num: int) -> List[ReferenceImage]:
        """Search via local SearXNG instance."""
        resp = requests.get(
            f"{self.searxng_url}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "images",
                "engines": "google images,bing images",
                "language": "auto",
                "safesearch": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("results", [])[:num]:
            img_url = item.get("img_src") or item.get("url", "")
            if not img_url or not img_url.startswith("http"):
                continue

            title = item.get("title", "")
            source = item.get("engine", "searxng")
            score = 0.5

            # Boost score based on title relevance
            title_lower = title.lower()
            if any(kw in title_lower for kw in self.TURNAROUND_KEYWORDS):
                score = 0.9
            elif any(kw in title_lower for kw in self.BACK_KEYWORDS):
                score = 0.85
            elif any(kw in title_lower for kw in self.SIDE_KEYWORDS):
                score = 0.8

            results.append(ReferenceImage(
                url=img_url,
                title=title,
                source=source,
                relevance_score=score,
            ))

        return results

    def _search_duckduckgo(self, query: str, num: int) -> List[ReferenceImage]:
        """Fallback search via DuckDuckGo."""
        from urllib.parse import quote_plus

        # DuckDuckGo image search API (unofficial)
        url = f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images"

        # Use a simple approach — search for image URLs in the HTML
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })

        # Extract image URLs from response (basic pattern matching)
        img_urls = re.findall(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)', resp.text)
        img_urls = list(dict.fromkeys(img_urls))[:num]  # Dedupe

        return [
            ReferenceImage(url=u, title=query, source="duckduckgo", relevance_score=0.5)
            for u in img_urls
        ]

    def _classify_view(self, title: str, url: str) -> str:
        """Classify the view angle of an image based on title/URL."""
        text = (title + " " + url).lower()

        if any(kw in text for kw in self.TURNAROUND_KEYWORDS):
            return "turnaround"
        elif any(kw in text for kw in self.BACK_KEYWORDS):
            return "back"
        elif any(kw in text for kw in self.SIDE_KEYWORDS):
            return "side"
        elif any(kw in text for kw in self.THREE_QUARTER_KEYWORDS):
            return "3/4"
        else:
            return "unknown"


def search_references(image_path: str, description: str = "") -> SearchResult:
    """Convenience function for quick reference search."""
    searcher = ReferenceSearcher()
    return searcher.analyze_and_search(image_path, description)
