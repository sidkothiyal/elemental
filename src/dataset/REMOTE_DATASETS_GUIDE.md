# Remote Dataset Deployment Guide

Step-by-step guides for setting up remote dataset loading with Elemental.

---

## Table of Contents

1. [Quick Start: SSH Remote Files](#1-quick-start-ssh-remote-files)
2. [Quick Start: Dataset Server](#2-quick-start-dataset-server)
3. [SSH Setup (Approach 1)](#3-ssh-setup-approach-1)
4. [S3 Setup (Approach 1)](#4-s3-setup-approach-1)
5. [Dataset Server on AWS/Cloud (Approach 2)](#5-dataset-server-on-awscloud-approach-2)
6. [Training on vast.ai with Remote Dataset](#6-training-on-vastai-with-remote-dataset)
7. [Training on RunPod with Remote Dataset](#7-training-on-runpod-with-remote-dataset)
8. [Choosing Between Approaches](#8-choosing-between-approaches)
9. [Performance Tuning](#9-performance-tuning)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Quick Start: SSH Remote Files

**Scenario:** Dataset lives on a persistent server, training runs on ephemeral GPU instances.

### On the storage server

```bash
# Ensure the dataset is accessible via SSH
ls /data/div2k/train/  # should contain .png files
```

### On the training server

```bash
# Ensure SSH access works (key-based auth recommended)
ssh user@storage-server "ls /data/div2k/train/ | head -5"

# Run training with remote dataset config override
python -m trainer \
    experiment=autoencoder/compression_ae \
    dataloader.train.dataset.remote_config.protocol=ssh \
    dataloader.train.dataset.remote_config.remote_root="user@storage-server:/data/div2k/train" \
    dataloader.train.dataset.remote_config.cache_dir=/tmp/cache/div2k/train \
    dataloader.val.dataset.remote_config.protocol=ssh \
    dataloader.val.dataset.remote_config.remote_root="user@storage-server:/data/div2k/val" \
    dataloader.val.dataset.remote_config.cache_dir=/tmp/cache/div2k/val
```

Or use a YAML config — see `configs/dataset/div2k_remote_stream.yaml`.

---

## 2. Quick Start: Dataset Server

**Scenario:** Very large dataset that shouldn't be transferred. Server processes samples and sends only tensors.

### On the dataset server

```bash
# Set auth token
export ELEMENTAL_DATASET_SERVER_TOKEN="your-secret-token"

# Start server for training split
python scripts/start_dataset_server.py \
    experiment/autoencoder/compression_ae \
    --host 0.0.0.0 --port 5555 --split train --num_workers 4

# In another terminal: start server for validation split
python scripts/start_dataset_server.py \
    experiment/autoencoder/compression_ae \
    --host 0.0.0.0 --port 5556 --split val --num_workers 2
```

### On the training server

```bash
# Set the same auth token
export ELEMENTAL_DATASET_SERVER_TOKEN="your-secret-token"

# Verify server is reachable
python scripts/start_dataset_server.py --check --host <server-ip> --port 5555

# Run training with dataset server config
python -m trainer \
    experiment=autoencoder/compression_ae \
    dataset=div2k_served \
    dataloader.train.dataset.server_host=<server-ip> \
    dataloader.val.dataset.server_host=<server-ip>
```

---

## 3. SSH Setup (Approach 1)

### Prerequisites

- SSH key-based authentication between training and storage servers
- `rsync` installed on both machines (recommended for sync mode)
- `sshfs` Python package installed (`pip install sshfs`)

### Setting up SSH keys

```bash
# On the training server (or your local machine)
ssh-keygen -t ed25519 -f ~/.ssh/elemental_data -N ""

# Copy public key to storage server
ssh-copy-id -i ~/.ssh/elemental_data.pub user@storage-server

# Test connection
ssh -i ~/.ssh/elemental_data user@storage-server "echo OK"
```

### Stream mode config

Best for small-medium datasets. Training starts immediately.

```yaml
# In your experiment YAML or as overrides
dataloader:
  train:
    dataset:
      remote_config:
        mode: stream
        protocol: ssh
        remote_root: "user@storage-server:/data/div2k/train"
        cache_dir: /tmp/elemental_cache/div2k/train
        ssh_key_path: ~/.ssh/elemental_data
        prefetch_workers: 4  # parallel download threads
```

### Sync mode config

Best when you want the simplicity of local files after an initial sync.

```yaml
dataloader:
  train:
    dataset:
      remote_config:
        mode: sync
        protocol: ssh
        remote_root: "user@storage-server:/data/div2k/train"
        cache_dir: /tmp/elemental_cache/div2k/train
        ssh_key_path: ~/.ssh/elemental_data
```

### Non-standard SSH port

```yaml
remote_config:
  protocol: ssh
  remote_root: "user@storage-server:/data/div2k/train"
  port: 2222  # custom SSH port
  ssh_key_path: ~/.ssh/elemental_data
```

---

## 4. S3 Setup (Approach 1)

### Prerequisites

- AWS credentials configured (`~/.aws/credentials` or environment variables)
- `s3fs` Python package installed (`pip install s3fs`)
- Dataset uploaded to an S3 bucket

### Upload dataset to S3

```bash
aws s3 sync /data/div2k/ s3://my-datasets/div2k/ --exclude "*.DS_Store"
```

### Config

```yaml
dataloader:
  train:
    dataset:
      remote_config:
        mode: stream  # or sync
        protocol: s3
        remote_root: "s3://my-datasets/div2k/train"
        cache_dir: /tmp/elemental_cache/div2k/train
        # Credentials from env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        # Or for public buckets:
        # anon: true
```

### On the training server

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# Run training
python -m trainer experiment=autoencoder/compression_ae dataset=div2k_remote_stream
```

---

## 5. Dataset Server on AWS/Cloud (Approach 2)

### Setting up the server

1. **Launch a persistent instance** (e.g., AWS EC2 t3.xlarge — no GPU needed):
   ```bash
   # SSH into the instance
   ssh ubuntu@<server-ip>

   # Clone the repo and set up
   git clone <repo-url> elemental && cd elemental
   pip install -e .

   # Place your dataset
   ln -s /mnt/data/div2k data/div2k  # or wherever your data lives
   ```

2. **Open the required ports** in the security group / firewall:
   - Port 5555 (training split)
   - Port 5556 (validation split)

3. **Start the servers:**
   ```bash
   export ELEMENTAL_DATASET_SERVER_TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")
   echo "Token: $ELEMENTAL_DATASET_SERVER_TOKEN"  # save this for the training server

   # Use tmux or screen for persistence
   tmux new-session -d -s train_server \
       "python scripts/start_dataset_server.py \
           experiment/autoencoder/compression_ae \
           --host 0.0.0.0 --port 5555 --split train --num_workers 4 \
           --compression lz4"

   tmux new-session -d -s val_server \
       "python scripts/start_dataset_server.py \
           experiment/autoencoder/compression_ae \
           --host 0.0.0.0 --port 5556 --split val --num_workers 2 \
           --compression lz4"
   ```

4. **Verify from training machine:**
   ```bash
   export ELEMENTAL_DATASET_SERVER_TOKEN="<same-token>"
   python scripts/start_dataset_server.py --check --host <server-ip> --port 5555
   ```

### Benchmark before training

```bash
# On the server, test compression performance
python scripts/start_dataset_server.py \
    experiment/autoencoder/compression_ae --benchmark --split train
```

Output:
```
  lz4:  45.2 MB raw -> 12.3 MB compressed (27.2% ratio), 850.0 samples/s
  zstd: 45.2 MB raw -> 8.1 MB compressed (17.9% ratio), 620.0 samples/s
```

Choose `lz4` for speed, `zstd` for bandwidth-constrained networks.

---

## 6. Training on vast.ai with Remote Dataset

### Using Approach 1 (SSH remote files)

Add to your `run_remote.sh` command or experiment config:

```bash
# In deploy/run_remote.sh or via Hydra overrides
python -m trainer \
    experiment=autoencoder/compression_ae \
    dataloader.train.dataset.remote_config.protocol=ssh \
    dataloader.train.dataset.remote_config.remote_root="user@persistent-server:/data/div2k/train" \
    dataloader.train.dataset.remote_config.cache_dir=/tmp/cache/div2k/train \
    dataloader.train.dataset.remote_config.ssh_key_path=~/.ssh/id_rsa
```

**Tip:** Use `/tmp/` for `cache_dir` — it's fast local storage on vast.ai.

### Using Approach 2 (Dataset Server)

1. Start the dataset server on your persistent server (see section 5)
2. Use `div2k_served.yaml` config with server IP overrides:

```bash
python -m trainer \
    experiment=autoencoder/compression_ae \
    dataset=div2k_served \
    dataloader.train.dataset.server_host=<server-ip> \
    dataloader.val.dataset.server_host=<server-ip>
```

### SSH key setup for vast.ai

```bash
# In deploy/setup_creds.sh or manually:
scp -P <vast-ssh-port> ~/.ssh/elemental_data root@<vast-ip>:~/.ssh/
ssh -p <vast-ssh-port> root@<vast-ip> "chmod 600 ~/.ssh/elemental_data"
```

---

## 7. Training on RunPod with Remote Dataset

Similar to vast.ai. Key differences:

- RunPod pods have network volumes (`NETWORK_VOLUME_ID`) that persist across restarts
- Use the network volume as `cache_dir` for persistent caching:

```yaml
remote_config:
  mode: stream
  protocol: ssh
  remote_root: "user@storage-server:/data/div2k/train"
  cache_dir: /runpod-volume/elemental_cache/div2k/train  # persists across restarts
```

---

## 8. Choosing Between Approaches

| Factor | Approach 1 (Remote Files) | Approach 2 (Dataset Server) |
|--------|--------------------------|---------------------------|
| **Dataset size** | < 50 GB | Any size (100s GB+) |
| **Network speed** | Needs fast connection for first epoch | Works on any connection (only tensors) |
| **Server requirements** | Just storage + SSH/S3 | Needs compute (CPU + RAM for dataset) |
| **Resumption** | Cache survives restarts | Must reconnect (no state loss) |
| **Complexity** | Simpler setup | Requires running server process |
| **Data processing** | Done on training server | Done on dataset server |
| **Recommended for** | Standard image datasets | ImageNet-scale, custom preprocessing |

### Decision flowchart

```
Dataset < 50GB?
├── Yes → Can you wait for first-epoch download?
│   ├── Yes → Approach 1, sync mode
│   └── No  → Approach 1, stream mode
└── No  → Approach 2 (Dataset Server)
```

---

## 9. Performance Tuning

### Approach 1: Remote Files

- **`prefetch_workers`**: Match to available CPU cores. 4-8 is usually optimal.
- **`cache_dir`**: Use the fastest local storage. On cloud: `/tmp/` (SSD) over network volumes.
- **Stream vs sync**: Stream is better for large files (images). Sync is better for many small files (text paragraphs).
- **SSH tuning**: Add to `~/.ssh/config`:
  ```
  Host storage-server
      ControlMaster auto
      ControlPath ~/.ssh/sockets/%r@%h-%p
      ControlPersist 600
  ```
  This reuses SSH connections for massive speedup with many small files.

### Approach 2: Dataset Server

- **`num_workers` (server)**: Match to CPU cores. Each worker handles one `__getitem__` call.
- **`prefetch_size` (client)**: 16-64 per DataLoader worker. Higher = more memory but smoother throughput.
- **`compression`**: lz4 for LAN/fast networks, zstd for WAN/slow networks.
- **`num_workers` (DataLoader)**: Each worker opens its own connection. 4-8 workers × 32 prefetch = up to 256 in-flight requests.
- **Server machine**: Use a machine with fast storage (SSD) and sufficient RAM for dataset indexing.

---

## 10. Troubleshooting

### Connection refused (Approach 2)

```
ConnectionError: Server did not respond within 30000ms
```

- Check firewall/security group allows the port
- Verify server is running: `python scripts/start_dataset_server.py --check --host <ip> --port <port>`
- Check the server logs for errors

### Authentication failed (Approach 2)

```
ConnectionError: Handshake failed: Authentication failed
```

- Ensure `$ELEMENTAL_DATASET_SERVER_TOKEN` is set to the same value on both machines
- Check clock sync — timestamps must be within 5 minutes

### SSH permission denied (Approach 1)

- Verify key: `ssh -i ~/.ssh/elemental_data user@server "echo OK"`
- Check key permissions: `chmod 600 ~/.ssh/elemental_data`
- Ensure `ssh_key_path` in config points to the correct key

### Slow first epoch (Approach 1, stream mode)

- Increase `prefetch_workers` to download more files in parallel
- Consider sync mode for small datasets (downloads everything upfront)
- Check network bandwidth: `iperf3 -c storage-server`

### Out of disk space (Approach 1)

- Set `cache_dir` on a volume with sufficient space
- Old cache files from previous experiments are not auto-cleaned. Clear with:
  ```bash
  rm -rf /tmp/elemental_cache/*
  ```
