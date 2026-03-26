# src/tools — Utility Tools

Standalone tools for checkpoint manipulation and other maintenance tasks.

## Files

### `checkpoint_surgeon.py` — Checkpoint Surgery

Modifies checkpoint state dicts: renames keys, removes keys, adds prefixes/suffixes.

Used when a model's architecture changes (e.g., module renamed) but you want to reuse a previous checkpoint without retraining.

```bash
uv run python src/tools/checkpoint_surgeon.py --config-name rename_transformer_autoencoder
```

Configs in `configs/checkpoint_surgery/` define the rename operations.

Example config:
```yaml
# configs/checkpoint_surgery/rename_transformer_autoencoder.yaml
input_path: results/autoencoder/run_name/best.pt
output_path: results/autoencoder/run_name/best_renamed.pt
rename_rules:
  - old: "encoder.transformer"
    new: "encoder.backbone"
```
