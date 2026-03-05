---
name: asset-manager
description: Manage portfolio holdings in Asset-Management/assets.xlsx (view, upsert quantity, remove symbol).
metadata: {"nanobot":{"emoji":"📊","requires":{"bins":["python3"]}}}
---

# Asset Manager

Use this skill when the user asks to view or edit holdings.

Supported operations:

1. View holdings
```bash
python3 nanobot/skills/asset-manager/scripts/view_holdings.py
```

2. Add/update holding (symbol upsert)
```bash
python3 nanobot/skills/asset-manager/scripts/update_holding.py \
  --symbol AAPL.US --quantity 10 --name "Apple Inc." --account uSmart --price-usd 180
```

3. Remove holding (hard delete)
```bash
python3 nanobot/skills/asset-manager/scripts/remove_holding.py --symbol AAPL.US
```

## Workbook resolution

By default, scripts use:

1. `<repo>/Asset-Management/assets.xlsx`
2. `./assets.xlsx`

You can override with `--workbook /absolute/or/relative/path.xlsx`.

## Safety

- `update_holding.py` and `remove_holding.py` create timestamped backups under `<workbook_dir>/backups/`.
- Backups are rotated to keep the latest 10 files.
