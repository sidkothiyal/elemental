"""Vector quantiser with EMA codebook updates and codebook management."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class Quantizer(nn.Module):
    """Quantizer module with improved codebook management.

    Features:
    - EMA or gradient-based codebook updates
    - Codebook diversity loss to prevent collapse
    - Smart initialization from multiple batches
    - Dead code revival mechanism
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        codebook_hit_summary_frequency: int = 100,
        codebook_revival_threshold: int = 2,
        codebook_revival_enabled: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        dictionary_loss_weight: float = 1.0,
        commitment_loss_weight: float = 0.25,
        # Diversity loss parameters
        diversity_loss_enabled: bool = True,
        diversity_loss_weight: float = 0.1,
        diversity_loss_margin: float = 0.5,
        # Smart initialization parameters
        smart_init_enabled: bool = True,
        smart_init_steps: int = 100,
        smart_init_initial_resample_pct: float = 0.5,
        smart_init_final_resample_pct: float = 0.0,
    ):
        """Initialize the Quantizer.

        Args:
            codebook_size: The size of the codebook.
            codebook_dim: The dimension of the codebook.
            codebook_hit_summary_frequency: The frequency to summarize the codebook hits.
            codebook_revival_threshold: Threshold below which a codebook entry is considered unused.
            codebook_revival_enabled: Whether to enable codebook revival.
            use_ema: Whether to use EMA to update codebook entries instead of gradients.
            ema_decay: Decay rate for EMA updates (only used if use_ema=True).
            ema_epsilon: Small constant for numerical stability in EMA updates.
            dictionary_loss_weight: Weight for the dictionary loss (default: 1.0).
            commitment_loss_weight: Weight for the commitment loss (default: 0.25).
            diversity_loss_enabled: Whether to enable codebook diversity loss.
            diversity_loss_weight: Weight for the diversity loss.
            diversity_loss_margin: Minimum margin for pairwise codebook distances.
            smart_init_enabled: Whether to enable smart initialization from batches.
            smart_init_steps: Number of steps for smart initialization.
            smart_init_initial_resample_pct: Initial percentage of codebook to resample per batch.
            smart_init_final_resample_pct: Final percentage of codebook to resample
                (end of init phase). Must be between 0.0 and 1.0.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.dictionary_loss_weight = dictionary_loss_weight
        self.commitment_loss_weight = commitment_loss_weight

        # Diversity loss parameters
        self._diversity_loss_enabled = diversity_loss_enabled
        self.diversity_loss_weight = diversity_loss_weight
        self._diversity_loss_margin = diversity_loss_margin

        # Smart initialization parameters
        self._smart_init_enabled = smart_init_enabled
        self._smart_init_steps = smart_init_steps
        self._smart_init_initial_resample_pct = smart_init_initial_resample_pct
        self._smart_init_final_resample_pct = smart_init_final_resample_pct

        # When using EMA, codebook is updated via EMA, not gradients
        if use_ema:
            self.register_buffer("codebook", torch.randn(codebook_size, codebook_dim))
            # EMA buffers for cluster statistics
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_embed", torch.randn(codebook_size, codebook_dim))
        else:
            self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim))

        # Tracking buffers — typed declarations so pylint/mypy recognise the attrs
        self.codebook_hits: torch.Tensor
        self.codebook_cumulative_hits: torch.Tensor
        self.codebook_initialized: torch.Tensor
        self.codebook_batches_survived: torch.Tensor
        self.register_buffer("codebook_hits", torch.zeros(codebook_size))
        # Cumulative hits for smart init (not reset at summary frequency)
        self.register_buffer("codebook_cumulative_hits", torch.zeros(codebook_size))
        # Track which codebook entries have been initialized from data
        self.register_buffer("codebook_initialized", torch.zeros(codebook_size, dtype=torch.bool))
        # Track how many batches each entry has survived since last initialization/resampling
        self.register_buffer("codebook_batches_survived", torch.zeros(codebook_size))

        self._codebook_hit_summary_frequency = codebook_hit_summary_frequency
        self._codebook_revival_threshold = codebook_revival_threshold
        self._codebook_revival_enabled = codebook_revival_enabled
        self._call_count = 0

        self.dictionary_loss_fn = nn.MSELoss(reduction="mean")
        self.commitment_loss_fn = nn.MSELoss(reduction="mean")
        self.dictionary_loss = None
        self.commitment_loss = None
        self.diversity_loss = None

    def _compute_diversity_loss(self) -> torch.Tensor:
        """Compute diversity loss to encourage spread-out codebook vectors.

        Uses a margin-based loss that penalizes codebook vectors that are
        closer than the specified margin. This prevents codebook collapse
        where multiple entries converge to similar values.

        Returns:
            Diversity loss tensor.
        """
        # Compute pairwise distances between all codebook vectors
        # Shape: [codebook_size, codebook_size]
        pairwise_distances = torch.cdist(self.codebook, self.codebook, p=2)

        # Create mask to exclude diagonal (self-distances)
        mask = ~torch.eye(self.codebook_size, dtype=torch.bool, device=self.codebook.device)

        # Get non-diagonal distances
        distances = pairwise_distances[mask]

        # Margin-based loss: penalize distances below the margin
        # loss = max(0, margin - distance)^2
        margin_violations = F.relu(self._diversity_loss_margin - distances)
        diversity_loss = (margin_violations**2).mean()

        return diversity_loss

    def _get_smart_init_resample_percentage(self) -> float:
        """Get the current resample percentage based on linear decay schedule.

        Returns:
            Current resample percentage (0.0 to 1.0).
        """
        if self._call_count >= self._smart_init_steps:
            return 0.0

        # Linear interpolation from initial to final percentage
        progress = self._call_count / self._smart_init_steps
        resample_pct = self._smart_init_initial_resample_pct + progress * (
            self._smart_init_final_resample_pct - self._smart_init_initial_resample_pct
        )
        return max(0.0, resample_pct)

    def _compute_resample_probabilities(self) -> torch.Tensor:
        """Compute resampling probabilities based on inverse usage frequency.

        Codebook entries that are used less frequently have higher probability
        of being resampled. This helps ensure all codebook entries get
        initialized from actual data.

        Returns:
            Probability tensor of shape [codebook_size].
        """
        # Avoid division by zero - add small epsilon
        epsilon = 1e-8

        # Use per-entry survival count (how many batches since this entry was initialized/resampled)
        # Add 1 to avoid division by zero for newly initialized entries
        batches_survived = self.codebook_batches_survived + 1

        # Compute usage rate for each codebook entry relative to its survival time
        usage_rate = self.codebook_cumulative_hits / batches_survived

        # Inverse probability: less used entries have higher probability
        # Add epsilon to avoid division by zero for unused entries
        inverse_usage = 1.0 / (usage_rate + epsilon)

        # Entries that haven't been initialized from data get highest priority
        # Give uninitialized entries 10x the weight of the least used initialized entry
        max_inverse = inverse_usage.max()
        inverse_usage = torch.where(
            self.codebook_initialized,
            inverse_usage,
            max_inverse * 10.0,
        )

        # Normalize to get probabilities
        probabilities = inverse_usage / inverse_usage.sum()

        return probabilities

    def _smart_init_resample(self, flattened_x: torch.Tensor) -> int:
        """Perform smart initialization resampling from batch.

        Args:
            flattened_x: Flattened input tensor of shape [N, codebook_dim].

        Returns:
            Number of codebook entries resampled.
        """
        if not self._smart_init_enabled:
            return 0

        resample_pct = self._get_smart_init_resample_percentage()
        if resample_pct <= 0:
            return 0

        batch_size = flattened_x.shape[0]
        if batch_size == 0:
            return 0

        # Number of entries to resample this step
        num_to_resample = max(1, int(self.codebook_size * resample_pct))

        # Get resampling probabilities
        probabilities = self._compute_resample_probabilities()

        # Sample which codebook entries to replace
        indices_to_resample = torch.multinomial(probabilities, num_to_resample, replacement=False)

        # Sample random vectors from the batch
        batch_indices = torch.randint(0, batch_size, (num_to_resample,), device=flattened_x.device)

        # Replace codebook entries
        with torch.no_grad():
            sampled = flattened_x[batch_indices].clone().to(self.codebook.dtype)
            if self.use_ema:
                self.codebook[indices_to_resample] = sampled
                self.ema_embed[indices_to_resample] = sampled
                self.ema_cluster_size[indices_to_resample] = 1.0
            else:
                self.codebook.data[indices_to_resample] = sampled

            # Mark these entries as initialized from data
            self.codebook_initialized[indices_to_resample] = True
            # Reset survival and hit counters for resampled entries
            self.codebook_batches_survived[indices_to_resample] = 0
            self.codebook_cumulative_hits[indices_to_resample] = 0

        if self._call_count % self._codebook_hit_summary_frequency == 0:
            init_progress = min(1.0, self._call_count / self._smart_init_steps)
            num_initialized = self.codebook_initialized.sum().item()
            logger.info(
                "Smart init: resampled %d entries (pct: %.2f, progress: %.1f, initialized: %d/%d)",
                num_to_resample,
                resample_pct,
                init_progress,
                num_initialized,
                self.codebook_size,
            )

        return num_to_resample

    def update_codebook_with_ema(
        self, flattened_x: torch.Tensor, closest_indices: torch.Tensor
    ) -> None:
        """Update codebook using exponential moving average.

        Args:
            flattened_x: Flattened input tensor of shape [N, codebook_dim].
            closest_indices: Indices of closest codebook entries for each input.
        """
        if not self.use_ema or not self.training:
            return

        # Count how many times each codebook entry was used in this batch
        encodings = F.one_hot(
            closest_indices, num_classes=self.codebook_size
        ).float()  # [N, codebook_size]

        # Sum of assignments for each codebook entry in this batch
        encodings_sum = encodings.sum(0)  # [codebook_size]

        # Sum of encoder outputs for each codebook entry
        embed_sum = encodings.t() @ flattened_x  # [codebook_size, codebook_dim]

        # Update EMA statistics
        with torch.no_grad():
            # Update cluster sizes with EMA
            self.ema_cluster_size.mul_(self.ema_decay).add_(encodings_sum, alpha=1 - self.ema_decay)

            # Update embeddings with EMA
            self.ema_embed.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

            # Laplace smoothing for cluster sizes
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.ema_epsilon)
                / (n + self.codebook_size * self.ema_epsilon)
                * n
            )

            # Update codebook
            self.codebook.copy_(self.ema_embed / cluster_size.unsqueeze(1))

    def revive_dead_codebook_entries(self, flattened_x: torch.Tensor) -> int:
        """Revive unused codebook entries by replacing them with random encodings from the batch.

        Args:
            flattened_x: Flattened input tensor of shape [N, codebook_dim].

        Returns:
            Number of codebook entries that were revived.
        """
        if not self._codebook_revival_enabled:
            return 0

        # Find unused codebook entries
        unused_mask = self.codebook_hits < self._codebook_revival_threshold
        unused_indices = torch.where(unused_mask)[0]
        num_unused = unused_indices.numel()

        if num_unused > 0:
            # Randomly sample encodings from the batch to replace dead codes
            batch_size = flattened_x.shape[0]
            if batch_size > 0:
                # Sample random indices from the batch
                random_batch_indices = torch.randint(
                    0, batch_size, (num_unused,), device=flattened_x.device
                )

                # Replace unused codebook entries with these random encodings
                with torch.no_grad():
                    sampled = flattened_x[random_batch_indices].clone().to(self.codebook.dtype)
                    if self.use_ema:
                        # Update both codebook and EMA statistics
                        self.codebook[unused_indices] = sampled
                        self.ema_embed[unused_indices] = sampled
                        self.ema_cluster_size[unused_indices] = 1.0
                    else:
                        self.codebook.data[unused_indices] = sampled

                    # Mark these as initialized
                    self.codebook_initialized[unused_indices] = True
                    # Reset survival and hit counters for revived entries
                    self.codebook_batches_survived[unused_indices] = 0
                    self.codebook_cumulative_hits[unused_indices] = 0

                logger.info(
                    "Revived %d unused codebook entries (threshold: %d)",
                    num_unused,
                    self._codebook_revival_threshold,
                )

        return num_unused

    def get_codebook_utilization(self) -> dict[str, float]:
        """Get codebook utilization statistics.

        Returns:
            Dictionary with utilization metrics.
        """
        used_count = (self.codebook_hits > 0).sum().item()
        total_hits = self.codebook_hits.sum().item()

        return {
            "codebook_utilization": used_count / self.codebook_size,
            "codebook_used_entries": used_count,
            "codebook_total_entries": self.codebook_size,
            "codebook_total_hits": total_hits,
            "codebook_perplexity": self._compute_perplexity().item() if total_hits > 0 else 0.0,
        }

    def _compute_perplexity(self) -> torch.Tensor:
        """Compute codebook perplexity (effective number of codes used).

        Higher perplexity indicates better codebook utilization.

        Returns:
            Perplexity value.
        """
        # Normalize hits to get probability distribution
        probs = self.codebook_hits / (self.codebook_hits.sum() + 1e-8)
        # Filter out zero probabilities for entropy calculation
        probs = probs[probs > 0]
        # Compute entropy and convert to perplexity
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        perplexity = torch.exp(entropy)
        return perplexity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Quantizer.

        Args:
            x: Input tensor of shape [B, C, H, W] (4D) or [B, N, C] (3D).

        Returns:
            Quantized tensor with same shape as input.
        """
        if x.dim() == 4:
            flattened_x = x.permute(0, 2, 3, 1).reshape(-1, self.codebook_dim)
            distances = torch.cdist(flattened_x, self.codebook)
            closest_indices = torch.argmin(distances, dim=1)
            quantized = self.codebook[closest_indices].view(
                x.shape[0], x.shape[2], x.shape[3], self.codebook_dim
            )
            quantized = quantized.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            flattened_x = x.reshape(-1, self.codebook_dim)
            distances = torch.cdist(flattened_x, self.codebook)
            closest_indices = torch.argmin(distances, dim=1)
            quantized = self.codebook[closest_indices].view(
                x.shape[0], x.shape[1], self.codebook_dim
            )
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        # Update codebook with EMA if enabled
        self.update_codebook_with_ema(flattened_x, closest_indices)

        # Update hit counts
        self._call_count += 1

        # Use bincount for efficient hit counting (handles duplicates correctly)
        batch_hits = torch.bincount(closest_indices, minlength=self.codebook_size).float()
        self.codebook_hits += batch_hits
        self.codebook_cumulative_hits += batch_hits

        # Increment survival counter for all entries (tracks batches since last init/resample)
        self.codebook_batches_survived += 1

        # Smart initialization resampling (during init phase)
        if self.training:
            self._smart_init_resample(flattened_x)

        # Periodic logging and maintenance
        if self._call_count % self._codebook_hit_summary_frequency == 0:
            utilization = self.get_codebook_utilization()
            logger.info(
                "Codebook stats: - utilization: %.2f, perplexity: %.1f, used: %d/%d",
                utilization["codebook_utilization"],
                utilization["codebook_perplexity"],
                utilization["codebook_used_entries"],
                utilization["codebook_total_entries"],
            )
            logger.info(
                "Codebook stats: - hits: %d",
                utilization["codebook_total_hits"],
            )
            # Revive unused codebook entries before resetting the hits
            self.revive_dead_codebook_entries(flattened_x)
            self.codebook_hits.zero_()

        # Compute losses
        self.dictionary_loss = self.dictionary_loss_fn(quantized, x.detach())
        self.commitment_loss = self.commitment_loss_fn(x, quantized.detach())

        # Compute diversity loss if enabled
        if self._diversity_loss_enabled:
            self.diversity_loss = self._compute_diversity_loss()

        # Straight-through estimator: gradients flow through quantized
        return x + (quantized - x).detach()
