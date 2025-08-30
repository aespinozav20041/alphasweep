import pandas as pd

from quant_pipeline.simple_lstm import SimpleLSTM
from quant_pipeline.moving_average import MovingAverageStrategy


def test_strategies_share_predict_interface():
    data = pd.DataFrame({"ret": [0.1, -0.2, 0.05, 0.1, -0.05]})
    lstm = SimpleLSTM(input_size=1, hidden_size=2)
    lstm.fit(data, epochs=1, lr=0.01)
    ma = MovingAverageStrategy()
    lstm_signal = lstm.predict(data)
    ma_signal = ma.predict(data["ret"])
    assert len(lstm_signal) == 1
    assert ma_signal.ndim == 1
