# Dataset Configs

Dataset and dataloader configurations for different training tasks.

## Files

| File | Description | Task |
|------|-------------|------|
| `div2k.yaml` | DIV2K high-resolution images | Image compression/autoencoder |
| `div2k_diffusion.yaml` | DIV2K wrapped for diffusion | Diffusion model training |
| `cityscapes_depth.yaml` | Cityscapes with depth maps | Depth estimation |
| `cityscapes_seg.yaml` | Cityscapes with segmentation | Semantic segmentation |
| `cityscapes_autoencoder.yaml` | Cityscapes for reconstruction | Image autoencoder |
| `imagenet.yaml` | TinyImageNet | Classification |
| `wikitext.yaml` | WikiText paragraphs | Language modeling |
| `tiny_shakespeare.yaml` | Tiny Shakespeare | Character-level LM |

## Usage

### In Experiment Configs

```yaml
defaults:
  - /dataset: div2k
```

### Overriding Dataset Parameters

```yaml
# Override target image shape
target_shape: [512, 768]

# Override batch size (uses ${batch_size} interpolation)
batch_size: 32
```

## Dataset Structure

Each dataset config defines:

```yaml
dataset_name: div2k
target_shape: [640, 1024]  # Dataset-specific params

dataloader:
  train:
    _target_: torch.utils.data.DataLoader
    dataset:
      _target_: dataset.div2k_dataset.DIV2KDataset
      root_dir: data/div2k/train
      target_shape: ${target_shape}
    batch_size: ${batch_size}
    shuffle: true
    num_workers: 4
    # ... other dataloader params
  val:
    # Similar structure for validation
  test: null
```

## Adding New Datasets

1. Create `dataset/your_dataset.yaml`
2. Define the dataloader structure with train/val/test splits
3. Use `${batch_size}` for batch size interpolation
4. Reference in experiments: `- /dataset: your_dataset`
