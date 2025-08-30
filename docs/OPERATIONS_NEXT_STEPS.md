# Operations Next Steps

## Stress Tests
Run stress scenarios:

```bash
python -m src.stress.test_harness --config config/examples/config_stress.yaml
```

Outputs are written to `reports/` including CSV, SQLite and a Markdown summary.

## Champion–Challenger
Evaluate models and handle promotions:

```bash
python -m src.cc.champion_challenger --config config/examples/config_cc.yaml
```

A promotion report and SQLite log are produced in `reports/`.

## P&L Attribution
Generate factor and component attribution reports:

```bash
python -m src.analytics.pnl_attribution --start 2025-07-01 --end 2025-08-01
```

CSV, SQLite tables and simple PNG charts are created in `reports/`.

## Adding New Edges
See `src/templates/new_edge_template/docs/ADDING_EDGES.md` for guidance on
integrating new models or strategies into the ensemble and risk parity flow.
