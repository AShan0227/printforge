"""
Community — Model sharing, print results, and community templates.

Users can:
  - Share generated models with print settings
  - Rate and review other users' prints
  - Browse templates (pre-configured generation + print params)
  - Share "print recipes" (settings that work for specific models)

Data stored locally in SQLite, with optional sync to GitHub/cloud.
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.expanduser("~/.printforge/community.db")


@dataclass
class SharedModel:
    """A model shared by a user."""
    id: str
    title: str
    description: str
    creator: str
    image_url: str = ""
    glb_path: str = ""
    stl_path: str = ""
    vertices: int = 0
    faces: int = 0
    engine_used: str = ""           # "tripo_p1", "trellis", etc.
    generation_params: Dict = field(default_factory=dict)
    # Print info
    material: str = ""              # PLA, PETG, etc.
    print_time_min: int = 0
    printer_model: str = ""         # "Bambu X1C"
    print_settings: Dict = field(default_factory=dict)  # layer_height, infill, etc.
    # Community
    likes: int = 0
    downloads: int = 0
    rating: float = 0
    rating_count: int = 0
    tags: List[str] = field(default_factory=list)
    created_at: float = 0


@dataclass
class PrintRecipe:
    """Proven print settings for a specific type of model."""
    id: str
    name: str
    description: str
    material: str
    layer_height_mm: float = 0.2
    infill_percent: int = 15
    supports: bool = False
    printer_model: str = ""
    tips: List[str] = field(default_factory=list)
    success_rate: float = 0  # 0-1
    uses: int = 0
    # Applicable to
    model_category: str = ""  # "figure", "functional", "decorative"
    max_size_mm: float = 0
    min_wall_mm: float = 0


class CommunityHub:
    """Local-first community model sharing."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                creator TEXT DEFAULT 'anonymous',
                image_url TEXT DEFAULT '',
                glb_path TEXT DEFAULT '',
                stl_path TEXT DEFAULT '',
                vertices INTEGER DEFAULT 0,
                faces INTEGER DEFAULT 0,
                engine_used TEXT DEFAULT '',
                generation_params TEXT DEFAULT '{}',
                material TEXT DEFAULT '',
                print_time_min INTEGER DEFAULT 0,
                printer_model TEXT DEFAULT '',
                print_settings TEXT DEFAULT '{}',
                likes INTEGER DEFAULT 0,
                downloads INTEGER DEFAULT 0,
                rating REAL DEFAULT 0,
                rating_count INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]',
                created_at REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS recipes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                material TEXT DEFAULT 'PLA',
                layer_height_mm REAL DEFAULT 0.2,
                infill_percent INTEGER DEFAULT 15,
                supports INTEGER DEFAULT 0,
                printer_model TEXT DEFAULT '',
                tips TEXT DEFAULT '[]',
                success_rate REAL DEFAULT 0,
                uses INTEGER DEFAULT 0,
                model_category TEXT DEFAULT '',
                max_size_mm REAL DEFAULT 0,
                min_wall_mm REAL DEFAULT 0,
                created_at REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS ratings (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                user_id TEXT DEFAULT 'anonymous',
                score INTEGER NOT NULL,
                comment TEXT DEFAULT '',
                print_success INTEGER DEFAULT 1,
                created_at REAL DEFAULT 0,
                FOREIGN KEY (model_id) REFERENCES models(id)
            );
        """)
        self.db.commit()

    # ── Models ───────────────────────────────────────────────────────

    def share_model(self, model: SharedModel) -> str:
        """Share a model to the community."""
        if not model.id:
            model.id = f"model_{uuid.uuid4().hex[:12]}"
        if not model.created_at:
            model.created_at = time.time()

        self.db.execute("""
            INSERT OR REPLACE INTO models
            (id, title, description, creator, image_url, glb_path, stl_path,
             vertices, faces, engine_used, generation_params,
             material, print_time_min, printer_model, print_settings,
             likes, downloads, rating, rating_count, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model.id, model.title, model.description, model.creator,
            model.image_url, model.glb_path, model.stl_path,
            model.vertices, model.faces, model.engine_used,
            json.dumps(model.generation_params),
            model.material, model.print_time_min, model.printer_model,
            json.dumps(model.print_settings),
            model.likes, model.downloads, model.rating, model.rating_count,
            json.dumps(model.tags), model.created_at,
        ))
        self.db.commit()
        return model.id

    def get_model(self, model_id: str) -> Optional[SharedModel]:
        """Get a shared model by ID."""
        row = self.db.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        if not row:
            return None
        return self._row_to_model(row)

    def browse_models(
        self,
        category: str = "",
        sort_by: str = "created_at",
        limit: int = 20,
    ) -> List[SharedModel]:
        """Browse community models."""
        query = "SELECT * FROM models"
        params = []
        if category:
            query += " WHERE tags LIKE ?"
            params.append(f"%{category}%")
        query += f" ORDER BY {sort_by} DESC LIMIT ?"
        params.append(limit)

        rows = self.db.execute(query, params).fetchall()
        return [self._row_to_model(r) for r in rows]

    def like_model(self, model_id: str):
        """Like a model."""
        self.db.execute("UPDATE models SET likes = likes + 1 WHERE id=?", (model_id,))
        self.db.commit()

    def download_model(self, model_id: str):
        """Increment download count."""
        self.db.execute("UPDATE models SET downloads = downloads + 1 WHERE id=?", (model_id,))
        self.db.commit()

    # ── Recipes ──────────────────────────────────────────────────────

    def add_recipe(self, recipe: PrintRecipe) -> str:
        """Share a print recipe."""
        if not recipe.id:
            recipe.id = f"recipe_{uuid.uuid4().hex[:8]}"

        self.db.execute("""
            INSERT OR REPLACE INTO recipes
            (id, name, description, material, layer_height_mm, infill_percent,
             supports, printer_model, tips, success_rate, uses,
             model_category, max_size_mm, min_wall_mm, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            recipe.id, recipe.name, recipe.description, recipe.material,
            recipe.layer_height_mm, recipe.infill_percent,
            1 if recipe.supports else 0, recipe.printer_model,
            json.dumps(recipe.tips), recipe.success_rate, recipe.uses,
            recipe.model_category, recipe.max_size_mm, recipe.min_wall_mm,
            time.time(),
        ))
        self.db.commit()
        return recipe.id

    def get_recipe_for_model(self, category: str, size_mm: float, material: str = "") -> List[PrintRecipe]:
        """Find best recipes for a model category and size."""
        query = "SELECT * FROM recipes WHERE model_category=?"
        params = [category]
        if material:
            query += " AND material=?"
            params.append(material)
        query += " ORDER BY success_rate DESC, uses DESC LIMIT 5"

        rows = self.db.execute(query, params).fetchall()
        return [self._row_to_recipe(r) for r in rows]

    # ── Ratings ──────────────────────────────────────────────────────

    def rate_model(self, model_id: str, score: int, comment: str = "", print_success: bool = True):
        """Rate a model (1-5 stars)."""
        rating_id = f"rating_{uuid.uuid4().hex[:8]}"
        self.db.execute("""
            INSERT INTO ratings (id, model_id, score, comment, print_success, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (rating_id, model_id, min(max(score, 1), 5), comment,
              1 if print_success else 0, time.time()))

        # Update model's average rating
        avg = self.db.execute(
            "SELECT AVG(score), COUNT(*) FROM ratings WHERE model_id=?", (model_id,)
        ).fetchone()
        if avg[0]:
            self.db.execute(
                "UPDATE models SET rating=?, rating_count=? WHERE id=?",
                (avg[0], avg[1], model_id)
            )
        self.db.commit()

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Community statistics."""
        models = self.db.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        recipes = self.db.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
        ratings = self.db.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
        total_likes = self.db.execute("SELECT SUM(likes) FROM models").fetchone()[0] or 0
        total_downloads = self.db.execute("SELECT SUM(downloads) FROM models").fetchone()[0] or 0
        return {
            "models": models,
            "recipes": recipes,
            "ratings": ratings,
            "total_likes": total_likes,
            "total_downloads": total_downloads,
        }

    # ── Seed Data ────────────────────────────────────────────────────

    def seed_default_recipes(self):
        """Add default print recipes for common scenarios."""
        defaults = [
            PrintRecipe(
                id="recipe_figure_pla", name="PLA Figure (Standard)",
                description="Best for detailed figurines under 100mm",
                material="PLA", layer_height_mm=0.12, infill_percent=15,
                supports=True, model_category="figure",
                tips=["Use tree supports", "0.4mm nozzle", "Print at 200°C"],
                success_rate=0.9, max_size_mm=100, min_wall_mm=0.8,
            ),
            PrintRecipe(
                id="recipe_figure_resin", name="Resin Figure (High Detail)",
                description="Maximum detail for small figurines",
                material="Resin", layer_height_mm=0.05, infill_percent=100,
                supports=True, model_category="figure",
                tips=["Use 8K resin printer", "Hollow > 20mm models", "Cure 5min under UV"],
                success_rate=0.85, max_size_mm=80, min_wall_mm=0.3,
            ),
            PrintRecipe(
                id="recipe_functional_petg", name="PETG Functional Part",
                description="Strong parts that need heat/chemical resistance",
                material="PETG", layer_height_mm=0.2, infill_percent=40,
                supports=False, model_category="functional",
                tips=["Print at 230°C", "60°C bed", "Slower first layer"],
                success_rate=0.88, max_size_mm=200, min_wall_mm=1.2,
            ),
        ]
        for recipe in defaults:
            self.add_recipe(recipe)

    # ── Internal ─────────────────────────────────────────────────────

    def _row_to_model(self, row) -> SharedModel:
        return SharedModel(
            id=row["id"], title=row["title"], description=row["description"],
            creator=row["creator"], image_url=row["image_url"],
            glb_path=row["glb_path"], stl_path=row["stl_path"],
            vertices=row["vertices"], faces=row["faces"],
            engine_used=row["engine_used"],
            generation_params=json.loads(row["generation_params"]),
            material=row["material"], print_time_min=row["print_time_min"],
            printer_model=row["printer_model"],
            print_settings=json.loads(row["print_settings"]),
            likes=row["likes"], downloads=row["downloads"],
            rating=row["rating"], rating_count=row["rating_count"],
            tags=json.loads(row["tags"]), created_at=row["created_at"],
        )

    def _row_to_recipe(self, row) -> PrintRecipe:
        return PrintRecipe(
            id=row["id"], name=row["name"], description=row["description"],
            material=row["material"], layer_height_mm=row["layer_height_mm"],
            infill_percent=row["infill_percent"],
            supports=bool(row["supports"]), printer_model=row["printer_model"],
            tips=json.loads(row["tips"]), success_rate=row["success_rate"],
            uses=row["uses"], model_category=row["model_category"],
            max_size_mm=row["max_size_mm"], min_wall_mm=row["min_wall_mm"],
        )
