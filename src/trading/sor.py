"""Simple smart order router (SOR).

This module exposes a :func:`route_order` helper that inspects quotes
across multiple venues and submits the order through the venue offering
best execution.  It expects each venue to be represented by an
``ExecutionClient`` that optionally implements a ``quote`` method
returning ``(price, volume)`` for a given symbol.

If only a single execution client is provided, the function falls back to
sending the order directly without routing.
"""

from __future__ import annotations

from typing import Dict

from execution import ExecutionClient, Order


def _pick_venue(order: Order, venues: Dict[str, ExecutionClient]) -> ExecutionClient:
    """Select the best venue based on price and available volume."""

    best: ExecutionClient | None = None
    best_price: float | None = None
    required_vol = abs(order.qty)

    for client in venues.values():
        quote_fn = getattr(client, "quote", None)
        if quote_fn is None:
            continue
        price, volume = quote_fn(order.symbol)
        if volume < required_vol:
            continue
        if best is None:
            best = client
            best_price = price
            continue
        if order.qty > 0:  # buy -> prefer lower price
            if price < (best_price if best_price is not None else float("inf")):
                best = client
                best_price = price
        else:  # sell -> prefer higher price
            if price > (best_price if best_price is not None else float("-inf")):
                best = client
                best_price = price

    if best is None:
        # No venue had enough liquidity or quotes were unavailable; fall back
        # to the first client in the mapping.
        best = next(iter(venues.values()))
    return best


def route_order(order: Order, venues: Dict[str, ExecutionClient] | ExecutionClient) -> str:
    """Route ``order`` to the optimal venue and return its id.

    Parameters
    ----------
    order:
        Order to be submitted.
    venues:
        Either a mapping of venue names to ``ExecutionClient`` instances or a
        single ``ExecutionClient``.  When a mapping is provided the router will
        attempt to query each venue for quotes via a ``quote`` method.

    Returns
    -------
    str
        Identifier of the submitted order.
    """

    if isinstance(venues, dict):
        client = _pick_venue(order, venues)
        return client.send(order)

    # Fallback for single client – no routing required
    return venues.send(order)
