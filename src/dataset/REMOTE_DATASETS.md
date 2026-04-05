# Remote Dataset Architecture

This document describes the remote dataset loading system for Elemental. It solves the problem of dataset availability on ephemeral/interruptible GPU servers (vast.ai, runpod) without requiring full dataset downloads before training.

## Overview

Two complementary approaches are provided:

| Approach | Best For | Latency | Dataset Transfer |
|----------|----------|---------|-----------------|
| **Approach 1: Remote File + Cache** | Small-medium datasets (<50GB) | First epoch slower, subsequent epochs at local speed | Full dataset cached locally |
| **Approach 2: Dataset Server** | Large datasets (100s GB+) | Consistent per-sample latency | Only processed tensors transferred |

Both approaches are transparent to the training pipeline — they produce standard `torch.utils.data.Dataset` objects compatible with `DataLoader`, `CUDAPrefetcher`, and `DistributedSampler`.

---

## Approach 1: Remote File Loading with On-the-Fly Caching

### Architecture

```
Training Server                          Storage Server
┌──────────────────────┐                ┌──────────────────┐
│  DIV2KDataset        │                │  /datasets/      │
│  (extends BaseDataset)│                │    div2k/        │
│                      │  SSH/S3/HTTP   │      train/      │
│  resolve_path(file) ─┼───────────────>│        0001.png  │
│       │              │                │        0002.png  │
│       ▼              │                │        ...       │
│  Local Cache         │                └──────────────────┘
│  /tmp/elemental_cache│
│    div2k/train/      │
│      0001.png (cached)│
│      0002.png (cached)│
└──────────────────────┘
```

### Components

#### BaseDataset (`src/dataset/base_dataset.py`)

Parent class for all datasets. Replaces `torch.utils.data.Dataset` as the base. Provides:

- **`is_remote`** — property indicating whether remote storage is configured
- **`remote_mode`** — `"stream"`, `"sync"`, or `None`
- **`resolve_path(path)`** — downloads file if remote, returns local path (cached)
- **`open_file(path, mode)`** — opens file transparently
- **`start_prefetch(paths)`** — background-downloads files in parallel
- **File discovery methods** — see below

When no `remote_config` is provided, all methods are transparent pass-throughs with zero overhead.

#### RemoteFileSystem (`src/dataset/remote_fs.py`)

Wraps `fsspec` to provide unified access to SSH, S3, HTTP, and local filesystems.

**Protocols:**
- `ssh` — via `sshfs` backend, supports key-based auth
- `s3` — via `s3fs` backend, credentials from env vars
- `http` / `https` — via fsspec HTTP backend
- `local` — passthrough (no-op)

**Modes:**
- `stream` — per-file on-the-fly caching with background prefetching
- `sync` — bulk directory sync via rsync/scp/aws-cli before dataset init

### File Discovery Strategies

Remote directory listings (glob over network) are expensive. `BaseDataset` provides multiple discovery strategies. Each dataset chooses the most efficient one based on its known structure:

| Method | Network Calls | Best For |
|--------|--------------|----------|
| `discover_files_from_range()` | 0 | Numbered files (DIV2K: `0001.png`–`0900.png`) |
| `discover_files_from_manifest()` | 1 (manifest file) | Any dataset with a pre-generated file list |
| `discover_files_from_structure()` | N (one per subdir) | Known directory layout (ImageNet class dirs) |
| `list_files()` | 1 (glob) | Fallback — works everywhere but slower remote |

#### Manifest Files

For maximum efficiency with remote datasets, pre-generate a `manifest.json` on the storage server:

```json
["image_0001.png", "image_0002.png", "image_0003.png"]
```

Or a plain text file with one path per line:

```
image_0001.png
image_0002.png
image_0003.png
```

The dataset downloads only this single manifest file to discover all available samples.

### Prefetching

After file discovery, `start_prefetch(paths)` launches a `ThreadPoolExecutor` that downloads files in the background. The training loop can start immediately — `resolve_path()` in `__getitem__` blocks only if the file hasn't been downloaded yet.

```
Timeline:
  Prefetcher:  [download 0001] [download 0002] [download 0003] ...
  Training:    [wait] [process 0001] [process 0002] [process 0003] ...
                 ↑ only waits if prefetch hasn't finished this file yet
```

Second epoch and beyond: all files are cached locally, zero network overhead.

### Cache Persistence

The cache directory (`cache_dir`) persists across training runs. If a training job is interrupted and restarted:
- **Stream mode**: already-cached files are reused; only missing files are downloaded
- **Sync mode**: rsync performs incremental sync (only new/changed files)

### Configuration

Remote config is added as `remote_config` in the dataset's Hydra YAML:

```yaml
dataloader:
  train:
    dataset:
      remote_config:
        mode: stream          # or "sync"
        protocol: ssh         # or "s3", "http"
        remote_root: "user@host:/datasets/div2k/train"
        cache_dir: /tmp/elemental_cache/div2k/train
        ssh_key_path: ~/.ssh/id_rsa
        prefetch_workers: 4
```

See `configs/dataset/div2k_remote_stream.yaml` and `configs/dataset/div2k_remote_sync.yaml` for complete examples.

---

## Approach 2: Live Dataset Server

### Architecture

```
Training Server                              Dataset Server
┌───────────────────────┐                   ┌───────────────────────┐
│  RemoteDataset        │                   │  DatasetServer        │
│  (torch Dataset)      │                   │                       │
│                       │   ZeroMQ          │  DIV2KDataset         │
│  __getitem__(idx) ────┼──── GETITEM(42) ──>│  __getitem__(42)     │
│       │               │                   │       │               │
│       │               │<── [meta, data] ──┼───────┘               │
│       ▼               │   lz4-compressed  │  serialize + compress │
│  deserialize_sample() │                   │                       │
│  return dict[Tensor]  │                   │  ThreadPoolExecutor   │
│                       │                   │  (parallel workers)   │
└───────────────────────┘                   └───────────────────────┘
```

### Components

#### DatasetServer (`src/dataset/dataset_server.py`)

Runs on the machine with the dataset. Instantiates the real dataset from Hydra config, processes `__getitem__` in worker threads, and serves compressed tensors over ZeroMQ ROUTER socket.

**Features:**
- Multi-client support via ZeroMQ ROUTER/DEALER pattern
- Parallel sample processing with `ThreadPoolExecutor`
- lz4 or zstd compression
- HMAC-SHA256 authentication
- Bandwidth monitoring and statistics logging
- Graceful shutdown via SIGINT/SIGTERM

#### RemoteDataset (`src/dataset/remote_dataset.py`)

Client-side `torch.utils.data.Dataset`. Drop-in replacement in training configs. Connects to a `DatasetServer` and fetches samples by index.

**Features:**
- Automatic retry with exponential backoff
- Timeout handling
- Compression auto-negotiation from server handshake
- Multi-worker DataLoader compatible (each worker opens its own socket)
- Health check via `ping()` method

#### Serialization (`src/dataset/serialization.py`)

Shared between server and client. Uses `torch.save`/`torch.load` with `BytesIO` for exact tensor preservation (dtypes, shapes, strides), then applies lz4 or zstd compression.

### Protocol

Communication uses ZeroMQ ROUTER/DEALER pattern with JSON metadata and binary payloads:

| Message | Client Sends | Server Replies |
|---------|-------------|----------------|
| `HANDSHAKE` | `{type, timestamp, token}` | `{status, length, compression}` |
| `GETITEM` | `{type, idx}` | `[metadata_json, compressed_data]` |
| `GETBATCH` | `{type, indices}` | `[metadata_json, compressed_data]` |
| `PING` | `{type}` | `{status, uptime_s, samples_served}` |

### Authentication

1. Server reads shared secret from `$ELEMENTAL_DATASET_SERVER_TOKEN`
2. Client computes `HMAC-SHA256(secret, timestamp)` and sends in handshake
3. Server verifies HMAC and checks timestamp freshness (5-minute window)
4. One-time per connection — not repeated per request

If no token is set, auth is disabled (warning logged).

### Multi-Worker DataLoader Compatibility

When `DataLoader(num_workers=N)` is used, each worker process (fork/spawn) creates its own `RemoteDataset` instance with its own ZeroMQ DEALER socket. The server's ROUTER socket handles all clients concurrently. No special configuration needed.

### Configuration

Replace the dataset `_target_` with `RemoteDataset`:

```yaml
dataloader:
  train:
    _target_: torch.utils.data.DataLoader
    dataset:
      _target_: dataset.remote_dataset.RemoteDataset
      server_host: "192.168.1.100"
      server_port: 5555
      compression: lz4
      prefetch_size: 32
```

See `configs/dataset/div2k_served.yaml` for a complete example.

### Server CLI

```bash
# Start server for training split
python scripts/start_dataset_server.py \
    experiment/autoencoder/compression_ae \
    --host 0.0.0.0 --port 5555 --split train --num_workers 4

# Health check
python scripts/start_dataset_server.py --check --host 192.168.1.100 --port 5555

# Benchmark compression
python scripts/start_dataset_server.py \
    experiment/autoencoder/compression_ae --benchmark --split train
```

---

## Class Hierarchy

```
torch.utils.data.Dataset
    └── BaseDataset (base_dataset.py)
            ├── DIV2KDataset (div2k_dataset.py)
            ├── CityscapesDataset (cityscapes_dataset.py)
            ├── ImagenetDataset (imagenet_dataset.py)
            ├── WikiTextDataset (wikitext_dataset.py)
            └── TinyShakespeareDataset (tiny_shakespeare_dataset.py)

    └── RemoteDataset (remote_dataset.py)  [Approach 2 client, standalone]

DiffusionDataset — wraps any of the above, unchanged
```

---

## File Reference

| File | Purpose |
|------|---------|
| `src/dataset/base_dataset.py` | Base class with remote support and file discovery |
| `src/dataset/remote_fs.py` | fsspec wrapper — file caching, prefetch, directory sync |
| `src/dataset/remote_dataset.py` | Approach 2 client (fetches from DatasetServer) |
| `src/dataset/dataset_server.py` | Approach 2 server (serves processed samples) |
| `src/dataset/serialization.py` | Tensor serialization with lz4/zstd compression |
| `scripts/start_dataset_server.py` | CLI to start dataset server from Hydra config |
| `configs/dataset/div2k_remote_stream.yaml` | Example: Approach 1 stream mode |
| `configs/dataset/div2k_remote_sync.yaml` | Example: Approach 1 sync mode |
| `configs/dataset/div2k_served.yaml` | Example: Approach 2 dataset server |

---

## Dependencies

| Package | Purpose | Required By |
|---------|---------|-------------|
| `pyzmq` | ZeroMQ bindings | Approach 2 |
| `fsspec` | Filesystem abstraction | Approach 1 |
| `sshfs` | fsspec SSH backend | Approach 1 (SSH) |
| `s3fs` | fsspec S3 backend | Approach 1 (S3) |
| `aiohttp` | fsspec HTTP backend | Approach 1 (HTTP) |
| `zstandard` | zstd compression (optional) | Both (alternative to lz4) |
| `lz4` | lz4 compression (already installed) | Both (default) |
| `fire` | CLI argument parsing | Server CLI |
