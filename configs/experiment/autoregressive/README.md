# Autoregressive Language Model Experiments

Language model experiments for text generation.

## Available Experiments

| Config | Description | Dataset | Model |
|--------|-------------|---------|-------|
| `wikitext_lm.yaml` | GPT-style LM on WikiText | WikiText | Transformer |
| `tiny_shakespeare.yaml` | Character-level LM | Tiny Shakespeare | Small Transformer |

## Usage

```bash
# WikiText language model
python src/trainer.py experiment/autoregressive/wikitext_lm

# Tiny Shakespeare (character-level)
python src/trainer.py experiment/autoregressive/tiny_shakespeare
```

## Key Differences

### WikiText LM
- Token-level using GPT-2 tokenizer
- Larger model (768 dim, 6 layers, 6 heads)
- Longer context (1024 tokens)

### Tiny Shakespeare
- Character-level (no pretrained tokenizer)
- Smaller model (384 dim, 1 layer, 1 head)
- Shorter context (256 chars)
- Useful for quick experimentation

## Common Overrides

```bash
# Increase model size
python src/trainer.py experiment/autoregressive/wikitext_lm num_layers=12 num_heads=12

# Change sequence length
python src/trainer.py experiment/autoregressive/wikitext_lm max_length=512
```
