"""Model registry handling model tracking and promotion."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.calibration import calibration_curve


def plot_calibration(prob_pred: np.ndarray, prob_true: np.ndarray, *, path: Path) -> None:
    """Plot and save a calibration curve."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Empirical")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical probability")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

try:  # optional MLflow integration
    import mlflow  # type: ignore
except Exception:  # pragma: no cover - mlflow is optional
    mlflow = None  # type: ignore


@dataclass
class ModelRecord:
    id: int
    ts: int
    type: str
    genes_json: str
    artifact_path: str
    calib_path: str
    lstm_path: Optional[str]
    scaler_path: Optional[str]
    features_path: Optional[str]
    thresholds_path: Optional[str]
    risk_rules_path: Optional[str]
    ga_version: Optional[str]
    seed: Optional[int]
    data_hash: Optional[str]
    status: str


class ModelRegistry:
    """SQLite backed registry for models and their performance."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS models(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER,
                type TEXT,
                genes_json TEXT,
                artifact_path TEXT,
                calib_path TEXT,
                lstm_path TEXT,
                scaler_path TEXT,
                features_path TEXT,
                thresholds_path TEXT,
                risk_rules_path TEXT,
                ga_version TEXT,
                seed INTEGER,
                data_hash TEXT,
                status TEXT
            )
            """
        )
        # for backwards compatibility add missing columns
        for col, typ in [
            ("lstm_path", "TEXT"),
            ("scaler_path", "TEXT"),
            ("features_path", "TEXT"),
            ("thresholds_path", "TEXT"),
            ("risk_rules_path", "TEXT"),
            ("ga_version", "TEXT"),
            ("seed", "INTEGER"),
            ("data_hash", "TEXT"),
        ]:
            try:
                cur.execute(f"ALTER TABLE models ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_perf(
                model_id INTEGER,
                ts INTEGER,
                ret REAL,
                sharpe REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_oos(
                model_id INTEGER,
                ts INTEGER,
                params_json TEXT,
                metrics_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_calibration(
                model_id INTEGER,
                ts INTEGER,
                prob_true TEXT,
                prob_pred TEXT
            )
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Registry operations
    # ------------------------------------------------------------------
    def register_model(
        self,
        *,
        model_type: str,
        genes_json: str,
        artifact_path: str,
        calib_path: str,
        lstm_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        features_path: Optional[str] = None,
        thresholds_path: Optional[str] = None,
        risk_rules_path: Optional[str] = None,
        ga_version: Optional[str] = None,
        seed: Optional[int] = None,
        data_hash: Optional[str] = None,
        status: str = "challenger",
        ts: Optional[int] = None,
    ) -> int:
        """Register a model and return its id."""

        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """INSERT INTO models(ts, type, genes_json, artifact_path, calib_path,
                lstm_path, scaler_path, features_path, thresholds_path, risk_rules_path,
                ga_version, seed, data_hash, status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts,
                    model_type,
                    genes_json,
                    artifact_path,
                    calib_path,
                    lstm_path,
                    scaler_path,
                    features_path,
                    thresholds_path,
                    risk_rules_path,
                    ga_version,
                    seed,
                    data_hash,
                    status,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def list_models(self, status: Optional[str] = None) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            if status:
                cur.execute("SELECT * FROM models WHERE status=?", (status,))
            else:
                cur.execute("SELECT * FROM models")
            return [dict(r) for r in cur.fetchall()]

    def log_perf(
        self, model_id: int, *, ret: float, sharpe: float, ts: Optional[int] = None
    ) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO model_perf(model_id, ts, ret, sharpe) VALUES(?,?,?,?)",
                (model_id, ts, ret, sharpe),
            )
            self.conn.commit()
        if mlflow is not None:
            if ts is None:
                mlflow.log_metric(f"pnl_model_{model_id}", ret)
            else:
                mlflow.log_metric(f"pnl_model_{model_id}", ret, step=ts)

    def log_oos_metrics(
        self,
        model_id: int,
        *,
        params: Dict,
        metrics: Dict,
        y_true: Optional[np.ndarray] = None,
        y_prob: Optional[np.ndarray] = None,
        ts: Optional[int] = None,
    ) -> None:
        """Record out-of-sample metrics and parameters for a model."""

        ts = ts or int(time.time())

        if y_true is not None and y_prob is not None:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            data = [
                {"prob": float(p), "empirical": float(t)}
                for p, t in zip(prob_pred, prob_true)
            ]
            calib_dir = self.db_path.parent / "calibration"
            calib_dir.mkdir(parents=True, exist_ok=True)
            base = calib_dir / f"model_{model_id}_{ts}"
            json_path = base.with_suffix(".json")
            csv_path = base.with_suffix(".csv")
            png_path = base.with_suffix(".png")
            with open(json_path, "w") as f:
                json.dump(data, f)
            with open(csv_path, "w") as f:
                f.write("prob,empirical\n")
                for row in data:
                    f.write(f"{row['prob']},{row['empirical']}\n")
            plot_calibration(np.array([d["prob"] for d in data]), np.array([d["empirical"] for d in data]), path=png_path)
            metrics = dict(metrics)
            metrics["calibration_curve"] = data
            metrics["calibration_json"] = str(json_path)
            metrics["calibration_csv"] = str(csv_path)
            metrics["calibration_png"] = str(png_path)

        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO model_oos(model_id, ts, params_json, metrics_json)
                VALUES(?,?,?,?)
                """,
                (model_id, ts, json.dumps(params), json.dumps(metrics)),
            )
            self.conn.commit()

    def list_oos_metrics(self, model_id: int) -> List[Dict]:
        """List logged out-of-sample metrics for a model."""

        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT * FROM model_oos WHERE model_id=? ORDER BY ts",
                (model_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def log_calibration_curve(
        self,
        model_id: int,
        *,
        prob_true: List[float],
        prob_pred: List[float],
        ts: Optional[int] = None,
    ) -> None:
        """Store calibration curve points for a model."""

        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO model_calibration(model_id, ts, prob_true, prob_pred)
                VALUES(?,?,?,?)
                """,
                (model_id, ts, json.dumps(prob_true), json.dumps(prob_pred)),
            )
            self.conn.commit()

    def list_calibration_curves(self, model_id: int) -> List[Dict]:
        """Retrieve stored calibration curves for a model."""

        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT * FROM model_calibration WHERE model_id=? ORDER BY ts",
                (model_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def promote_model(self, model_id: int, export_dir: str) -> None:
        """Promote a model to champion and export its artifacts."""

        with self._lock:
            cur = self.conn.cursor()
            cur.execute("UPDATE models SET status='archived' WHERE status='champion'")
            cur.execute("UPDATE models SET status='champion' WHERE id=?", (model_id,))
            self.conn.commit()
        self._export_champion(model_id, export_dir)

    # ------------------------------------------------------------------
    # Performance evaluation
    # ------------------------------------------------------------------
    def evaluate_challengers(
        self,
        *,
        eval_window_bars: int,
        uplift_min: float,
        min_bars_to_compare: int,
        export_dir: str,
        sharpe_min: float,
        max_drawdown: float,
    ) -> Optional[int]:
        """Evaluate challengers and promote if performance uplift achieved."""

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with self._lock:
            champ = self._current_champion()
            champ_perf = (
                self._recent_perf(champ["id"], eval_window_bars) if champ else []
            )

        def _metrics(perf: List[Dict]) -> Dict[str, float]:
            rets = [p["ret"] for p in perf]
            sharpes = [p["sharpe"] for p in perf]
            cum = np.cumsum(rets)
            dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0
            return {
                "ret": sum(rets),
                "sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
                "dd": dd,
                "bars": len(perf),
            }

        champ_metrics = _metrics(champ_perf) if champ_perf else {
            "ret": 0.0,
            "sharpe": float("-inf"),
            "dd": float("inf"),
            "bars": 0,
        }
        champ_ret = champ_metrics["ret"]

        challengers = self.list_models(status="challenger")

        def _eval(chal: Dict) -> Optional[int]:
            chal_perf = self._recent_perf(chal["id"], eval_window_bars)
            metrics = _metrics(chal_perf)
            bars = min(metrics["bars"], len(champ_perf))
            if bars < min_bars_to_compare:
                return None
            if champ_ret == 0:
                uplift = float("inf") if metrics["ret"] > 0 else 0.0
            else:
                uplift = (metrics["ret"] - champ_ret) / abs(champ_ret)
            if (
                uplift >= uplift_min
                and metrics["sharpe"] >= sharpe_min
                and metrics["dd"] <= max_drawdown
                and (
                    not champ_perf
                    or (
                        metrics["sharpe"] > champ_metrics["sharpe"]
                        and metrics["dd"] < champ_metrics["dd"]
                    )
                )
            ):
                self.promote_model(chal["id"], export_dir)
                return chal["id"]
            return None

        with ThreadPoolExecutor() as ex:
            futures = {ex.submit(_eval, c): c for c in challengers}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    return res
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _current_champion(self) -> Optional[Dict]:
        champs = self.list_models(status="champion")
        return champs[0] if champs else None

    def _recent_perf(self, model_id: int, window: int) -> List[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT * FROM model_perf WHERE model_id=? ORDER BY ts DESC LIMIT ?
                """,
                (model_id, window),
            )
            rows = cur.fetchall()
        rows.reverse()
        return [dict(r) for r in rows]

    def get_model(self, model_id: int) -> Dict:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM models WHERE id=?", (model_id,))
            row = cur.fetchone()
        if not row:
            raise ValueError("model not found")
        return dict(row)

    def delete_model(self, model_id: int) -> None:
        """Delete a model and its performance records."""

        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM models WHERE id=?", (model_id,))
            cur.execute("DELETE FROM model_perf WHERE model_id=?", (model_id,))
            self.conn.commit()

    def prune_challengers(self, max_challengers: int) -> None:
        """Keep only the most recent challengers."""

        challengers = self.list_models(status="challenger")
        if len(challengers) <= max_challengers:
            return
        challengers.sort(key=lambda m: m.get("ts") or 0)
        for m in challengers[:-max_challengers]:
            self.delete_model(m["id"])

    def _export_champion(self, model_id: int, export_dir: str) -> None:
        model = self.get_model(model_id)
        d = Path(export_dir)
        d.mkdir(parents=True, exist_ok=True)
        art_dest = d / "current_artifact"
        calib_dest = d / "current_calib"
        shutil.copyfile(model["artifact_path"], art_dest)
        shutil.copyfile(model["calib_path"], calib_dest)
        lstm_dest = None
        if model.get("lstm_path"):
            lstm_dest = d / "current_lstm"
            shutil.copyfile(model["lstm_path"], lstm_dest)
        scaler_dest = None
        if model.get("scaler_path"):
            scaler_dest = d / "current_scaler"
            shutil.copyfile(model["scaler_path"], scaler_dest)
        feat_dest = None
        if model.get("features_path"):
            feat_dest = d / "current_features"
            shutil.copyfile(model["features_path"], feat_dest)
        thresh_dest = None
        if model.get("thresholds_path"):
            thresh_dest = d / "current_thresholds"
            shutil.copyfile(model["thresholds_path"], thresh_dest)
        risk_dest = None
        if model.get("risk_rules_path"):
            risk_dest = d / "current_risk_rules"
            shutil.copyfile(model["risk_rules_path"], risk_dest)
        meta = {
            "id": model_id,
            "type": model["type"],
            "genes_json": model["genes_json"],
            "artifact": str(art_dest),
            "calib": str(calib_dest),
            "lstm": str(lstm_dest) if lstm_dest else None,
            "scaler": str(scaler_dest) if scaler_dest else None,
            "features": str(feat_dest) if feat_dest else None,
            "thresholds": str(thresh_dest) if thresh_dest else None,
            "risk_rules": str(risk_dest) if risk_dest else None,
            "ga_version": model.get("ga_version"),
            "seed": model.get("seed"),
            "data_hash": model.get("data_hash"),
        }
        with open(d / "current_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)


class ChampionReloader:
    """Hot-reloads champion when metadata file changes."""

    def __init__(self, meta_path: str | Path, loader: Callable[[Path], object]) -> None:
        self.meta_path = Path(meta_path)
        self.loader = loader
        self._last_mtime = 0.0
        self.model = None

    def poll(self) -> object:
        try:
            mtime = self.meta_path.stat().st_mtime
        except FileNotFoundError:
            return self.model
        if mtime > self._last_mtime:
            self._last_mtime = mtime
            self.model = self.loader(self.meta_path)
        return self.model


__all__ = ["ModelRegistry", "ChampionReloader", "ModelRecord", "plot_calibration"]
