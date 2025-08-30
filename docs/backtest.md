# Backtesting

The backtesting utilities allow plugging custom cost models to better
approximate real-world trading.

## Custom slippage function

`CostModel` accepts a `slippage_fn` with signature
`slippage_fn(spread, vol, volume)` returning slippage in basis points. The
function receives the bar's spread, volatility and traded volume.

```python
from execution import CostModel

# Simple example: widen slippage with spread and volatility

def slippage_fn(spread: float, vol: float, volume: float) -> float:
    return spread * 10_000 + vol * 100

cost = CostModel(fee_bps=1.0, slippage_fn=slippage_fn)
```

The simulator will pass the current bar's values when computing trade
costs.
