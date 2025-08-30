# Adding New Edges

Copy one of the adapters in `adapters/` and implement the model specific logic
inside the `predict` method. The adapter should expose a uniform API so that it
can be plugged into the ensemble and risk-parity allocator without changes.

## Steps
1. Implement an adapter for the new model or strategy.
2. Register the edge in the ensemble configuration.
3. Ensure the risk manager knows about the new horizon if applicable.
4. Add hooks via `src.engine_hooks` to persist weights or extra logs for attribution.
