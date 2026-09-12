"""
妖股生命周期数据湖存储层（DuckDB）。

三张表：
  episode        : 妖股炒作周期
  turning_point  : 变盘节点
  segment        : 切分后的阶段片段（含阶段特征向量）

PIT：segment.valid_from = end_date，查询 WHERE valid_from <= as_of。
"""

import os
from datetime import datetime

import duckdb
import pandas as pd


class LifecycleStore:
    def __init__(self, db_path="data/lifecycle.duckdb"):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._create_tables()
        self._point_counter = self._next_id("turning_point", "point_id")
        self._segment_counter = self._next_id("segment", "segment_id")

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episode (
                episode_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                start_date DATE,
                end_date DATE,
                n_segments INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS turning_point (
                point_id INTEGER PRIMARY KEY,
                episode_id VARCHAR,
                date DATE,
                type VARCHAR,
                confidence DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS segment (
                segment_id INTEGER PRIMARY KEY,
                episode_id VARCHAR,
                symbol VARCHAR,
                stage VARCHAR,
                start_date DATE,
                end_date DATE,
                start_idx INTEGER,
                end_idx INTEGER,
                feature_vec DOUBLE[],
                valid_from DATE
            )
        """)

    @staticmethod
    def _to_date(v):
        if v is None:
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.strftime("%Y-%m-%d")
        s = str(v).strip()
        return s[:10] if len(s) >= 10 else s

    @staticmethod
    def _to_float_list(vec):
        if vec is None:
            return []
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return [float(x) for x in vec]

    def _next_id(self, table, col):
        try:
            row = self.conn.execute(f"SELECT COALESCE(MAX({col}), 0) FROM {table}").fetchone()
            return int(row[0]) + 1
        except Exception:
            return 1

    # ---------- 写入 ----------

    def insert_episode(self, episode_id, symbol, start_date, end_date, n_segments):
        self.conn.execute(
            """
            INSERT INTO episode (episode_id, symbol, start_date, end_date, n_segments)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (episode_id) DO UPDATE SET
                symbol = excluded.symbol,
                start_date = excluded.start_date,
                end_date = excluded.end_date,
                n_segments = excluded.n_segments
            """,
            [episode_id, symbol, self._to_date(start_date), self._to_date(end_date), int(n_segments)],
        )

    def insert_turning_points(self, episode_id, pivots):
        for p in (pivots or []):
            self.conn.execute(
                "INSERT INTO turning_point (point_id, episode_id, date, type, confidence) VALUES (?, ?, ?, ?, ?)",
                [self._point_counter, episode_id, self._to_date(p["date"]),
                 p.get("type"), float(p.get("confidence", 0.0) or 0.0)],
            )
            self._point_counter += 1

    def insert_segments(self, episode_id, segments):
        for s in (segments or []):
            feature_vec = self._to_float_list(s.get("feature_vec"))
            valid_from = s.get("valid_from", s.get("end_date"))
            self.conn.execute(
                """
                INSERT INTO segment
                    (segment_id, episode_id, symbol, stage, start_date, end_date,
                     start_idx, end_idx, feature_vec, valid_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::DOUBLE[], ?)
                """,
                [self._segment_counter, episode_id, s["symbol"], s.get("stage"),
                 self._to_date(s["start_date"]), self._to_date(s["end_date"]),
                 int(s["start_idx"]), int(s["end_idx"]), feature_vec, self._to_date(valid_from)],
            )
            self._segment_counter += 1

    def delete_episode(self, episode_id):
        """删除一个 episode 及其所有子记录（用于幂等重建）。"""
        self.conn.execute("DELETE FROM segment WHERE episode_id = ?", [episode_id])
        self.conn.execute("DELETE FROM turning_point WHERE episode_id = ?", [episode_id])
        self.conn.execute("DELETE FROM episode WHERE episode_id = ?", [episode_id])

    # ---------- 查询 ----------

    def query_segments(self, as_of, stage=None):
        """PIT 查询：只返回 valid_from <= as_of 的片段，可选按 stage 过滤。"""
        sql = "SELECT * FROM segment WHERE valid_from <= ?"
        params = [self._to_date(as_of)]
        if stage:
            sql += " AND stage = ?"
            params.append(stage)
        sql += " ORDER BY start_date"
        return self.conn.execute(sql, params).fetchdf()

    def get_episode(self, episode_id):
        ep = self.conn.execute("SELECT * FROM episode WHERE episode_id = ?", [episode_id]).fetchdf()
        pts = self.conn.execute(
            "SELECT * FROM turning_point WHERE episode_id = ? ORDER BY date", [episode_id]
        ).fetchdf()
        segs = self.conn.execute(
            "SELECT * FROM segment WHERE episode_id = ? ORDER BY start_date", [episode_id]
        ).fetchdf()
        return {"episode": ep, "turning_points": pts, "segments": segs}

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()