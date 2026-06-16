"""
SQLite-backed ConfigManager for Brain-RAG Qdrant settings.
Provides hardware-aware dynamic configuration at startup.
"""

import os
import sqlite3
from typing import Optional

from loguru import logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed — hardware detection disabled")


class ConfigManager:
    """Manages Qdrant configuration via SQLite with hardware-aware profiles."""

    def __init__(self, db_path: str = "config/brain.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self):
        """Create DB and schema if missing."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_defaults()

    def _init_schema(self):
        """Create tables if they don't exist."""
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS qdrant_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS quantization_params (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                enabled BOOLEAN DEFAULT 0,
                scalar_type TEXT DEFAULT 'INT8',
                quantile REAL DEFAULT 0.99,
                always_ram BOOLEAN DEFAULT 1,
                hardware_profile TEXT DEFAULT 'medium'
            );
            CREATE TABLE IF NOT EXISTS hnsw_params (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                m INTEGER DEFAULT 16,
                ef_construct INTEGER DEFAULT 100,
                full_scan_threshold INTEGER DEFAULT 10000,
                on_disk BOOLEAN DEFAULT 0
            );
        """)
        self.conn.commit()

    def _seed_defaults(self):
        """Insert default configs if tables are empty."""
        cur = self.conn.cursor()
        # Seed qdrant_settings if empty
        cur.execute("SELECT COUNT(*) FROM qdrant_settings")
        if cur.fetchone()[0] == 0:
            defaults = [
                ("host", "localhost"),
                ("port", "6333"),
                ("grpc_port", "6334"),
                ("api_key", None),
                ("timeout", "60"),
                ("default_collection", "brain_rag_collection"),
                ("dense_vector_size", "384"),
                ("sparse_vector_size", "768"),
                ("prefer_grpc", "false"),
                ("sparse_strategy", "bm25"),
            ]
            cur.executemany("INSERT INTO qdrant_settings (key, value) VALUES (?, ?)", defaults)
        # Seed quantization_params if empty
        cur.execute("SELECT COUNT(*) FROM quantization_params")
        if cur.fetchone()[0] == 0:
            quant_defaults = [
                ("turboquant_low", 1, "INT8", 0.99, 0, "low"),
                ("turboquant_medium", 1, "INT8", 0.99, 1, "medium"),
                ("turboquant_high", 1, "INT8", 0.99, 1, "high"),
            ]
            cur.executemany(
                "INSERT INTO quantization_params (name, enabled, scalar_type, quantile, always_ram, hardware_profile) VALUES (?, ?, ?, ?, ?, ?)",
                quant_defaults
            )
        # Seed hnsw_params if empty
        cur.execute("SELECT COUNT(*) FROM hnsw_params")
        if cur.fetchone()[0] == 0:
            hnsw_defaults = [
                ("fast", 8, 64, 50000, 0),
                ("default", 16, 100, 10000, 0),
                ("high_recall", 32, 200, 1000, 0),
            ]
            cur.executemany(
                "INSERT INTO hnsw_params (name, m, ef_construct, full_scan_threshold, on_disk) VALUES (?, ?, ?, ?, ?)",
                hnsw_defaults
            )
        self.conn.commit()

    def get_qdrant_config(self) -> dict:
        """Return Qdrant connection config as dict."""
        cur = self.conn.cursor()
        cur.execute("SELECT key, value FROM qdrant_settings")
        rows = cur.fetchall()
        config = {}
        for key, value in rows:
            if value is None:
                config[key] = None
            elif key in ("port", "grpc_port", "timeout", "dense_vector_size", "sparse_vector_size"):
                config[key] = int(value)
            elif key in ("prefer_grpc",):
                config[key] = value.lower() == "true"
            else:
                config[key] = value
        return config

    def get_quantization_params(self, hw_profile: str = "medium") -> dict:
        """Get quantization params for hardware profile."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM quantization_params WHERE hardware_profile = ? AND enabled = 1 LIMIT 1",
            (hw_profile,)
        )
        row = cur.fetchone()
        if row:
            return dict(row)
        # Fallback to medium if profile not found
        cur.execute("SELECT * FROM quantization_params WHERE hardware_profile = 'medium' AND enabled = 1 LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else {}

    def get_hnsw_params(self, profile: str = "default") -> dict:
        """Get HNSW params by profile name."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM hnsw_params WHERE name = ?", (profile,))
        row = cur.fetchone()
        if row:
            return dict(row)
        # Fallback to default
        cur.execute("SELECT * FROM hnsw_params WHERE name = 'default'")
        row = cur.fetchone()
        return dict(row) if row else {"m": 16, "ef_construct": 100, "full_scan_threshold": 10000, "on_disk": False}

    def update_setting(self, key: str, value: str):
        """Update a Qdrant setting at runtime."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO qdrant_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, value)
        )
        self.conn.commit()

    def detect_hardware_profile(self) -> str:
        """Detect hardware profile based on available RAM."""
        if not PSUTIL_AVAILABLE:
            return "medium"
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        if total_gb < 4:
            return "low"
        elif total_gb < 8:
            return "medium"
        else:
            return "high"

    def close(self):
        """Close DB connection (idempotent)."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False