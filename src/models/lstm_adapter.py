from __future__ import annotations

import pickle
from typing import Any, Dict


class LSTMAdapter:
    """Adapter que mantiene estado oculto por símbolo para modelos LSTM."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.hidden_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any:
        """Ajusta el modelo subyacente."""
        return self.model.fit(X, y, **kwargs)

    # ------------------------------------------------------------------
    def predict(self, bar: Any, *, symbol: str, **kwargs: Any) -> Any:
        """Genera predicción manteniendo estado oculto por símbolo.

        Resetea el estado en rollovers o gaps si el objeto ``bar`` expone
        los atributos ``rollover`` o ``gap``.
        """

        if getattr(bar, "rollover", False) or getattr(bar, "gap", False):
            self.hidden_state.pop(symbol, None)

        state = self.hidden_state.get(symbol)
        result = self.model.predict(bar, state, **kwargs)

        # Se espera que ``model.predict`` devuelva (salida, nuevo_estado)
        if isinstance(result, tuple) and len(result) == 2:
            output, new_state = result
        else:  # pragma: no cover - fallback
            output, new_state = result, state

        self.hidden_state[symbol] = new_state
        return output

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persiste el modelo y los estados ocultos."""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "hidden_state": self.hidden_state}, f)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "LSTMAdapter":
        """Carga un modelo previamente guardado."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        adapter = cls(data["model"])
        adapter.hidden_state = data.get("hidden_state", {})
        return adapter
