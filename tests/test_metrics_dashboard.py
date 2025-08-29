from trading import metrics


def test_order_queue_gauge():
    metrics.ORDER_QUEUE.set(7)
    assert metrics.ORDER_QUEUE._value.get() == 7
