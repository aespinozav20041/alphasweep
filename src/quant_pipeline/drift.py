"""Simple data/concept drift detection utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    ks: float
    psi: float


class DriftDetector:
    """Detects data drift using KS-test and Population Stability Index."""

    def __init__(self, *, ks_threshold: float = 0.1, psi_threshold: float = 0.1, bins: int = 10) -> None:
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.bins = bins

    # ------------------------------------------------------------------
    def _ks_stat(self, ref: np.ndarray, cur: np.ndarray) -> float:
        ref_sorted = np.sort(ref)
        cur_sorted = np.sort(cur)
        all_vals = np.concatenate([ref_sorted, cur_sorted])
        cdf_ref = np.searchsorted(ref_sorted, all_vals, side="right") / len(ref_sorted)
        cdf_cur = np.searchsorted(cur_sorted, all_vals, side="right") / len(cur_sorted)
        return float(np.max(np.abs(cdf_ref - cdf_cur)))

    def _psi(self, ref: np.ndarray, cur: np.ndarray) -> float:
        quantiles = np.linspace(0, 100, self.bins + 1)
        bins = np.percentile(ref, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        ref_hist, _ = np.histogram(ref, bins=bins)
        cur_hist, _ = np.histogram(cur, bins=bins)
        ref_pct = ref_hist / max(len(ref), 1)
        cur_pct = cur_hist / max(len(cur), 1)
        # avoid division by zero
        mask = (ref_pct > 0) & (cur_pct > 0)
        psi = np.sum((ref_pct[mask] - cur_pct[mask]) * np.log(ref_pct[mask] / cur_pct[mask]))
        return float(psi)

    # ------------------------------------------------------------------
    def check(self, reference: np.ndarray, current: np.ndarray) -> DriftResult:
        ks = self._ks_stat(reference, current)
        psi = self._psi(reference, current)
        if ks > self.ks_threshold or psi > self.psi_threshold:
            logger.warning("data drift detected ks=%.3f psi=%.3f", ks, psi)
        return DriftResult(ks=ks, psi=psi)


__all__ = ["DriftDetector", "DriftResult"]
