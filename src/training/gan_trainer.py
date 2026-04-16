"""GAN trainer.

Overrides ``train_batch`` to run the three-phase GAN training step
(disc-on-fake → disc-on-real → generator), each with its own backward/step
cycle gated on the ``training`` flag. Overrides ``_build_*`` classmethods to
return dict-of-modules / dict-of-optimizers / dict-of-schedulers.
"""

import logging
import statistics
from typing import Any

import hydra.utils as hy_utils
import torch
from omegaconf import DictConfig
from torch import nn

from loss.utils import call_loss_with_epoch
from training.helpers import build_data_container, populate_predictions
from training.trainer import Trainer
from utils import distributed

logger = logging.getLogger(__name__)


class GANTrainer(Trainer):
    """GAN trainer. ``model`` is ``ModuleDict({"generator", "discriminator"})``,
    and ``optim`` / ``scheduler`` are dicts keyed the same way."""

    # ------------------------------------------------------------------
    # Override the single-batch step
    # ------------------------------------------------------------------

    def train_batch(  # pylint: disable=arguments-differ,too-many-locals,too-many-branches
        self,
        batch: dict,
        batch_idx: int,  # pylint: disable=unused-argument
        epoch: int,
        step: int,  # pylint: disable=unused-argument
        data: dict,
        *,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Three-phase GAN step: disc-on-fake → disc-on-real → generator.

        The ``training`` flag gates all three backward/step cycles. Forward
        passes and loss computation always run. Postprocessor runs INSIDE the
        autocast scope (unified with the supervised path).
        """
        config = self.config
        clip_grad_norm = config.get("clip_grad_norm", None)
        scheduler_step_per = config.get("scheduler_step_per", "epoch")
        use_scaler = self.scaler is not None and self.scaler.is_enabled()

        build_data_container(batch, data=data, model=self.model, epoch=epoch)

        # ---------------------------------------
        # Phase 1: Discriminator on fake data
        # ---------------------------------------
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=self.device.type, dtype=self.dtype),
        ):
            pred_target = self.model["generator"](batch["input"])
            discriminator_fake = self.model["discriminator"](pred_target.detach())
            disc_fake_loss = call_loss_with_epoch(
                self.loss["adversarial"],
                discriminator_fake,
                torch.zeros_like(discriminator_fake),
                epoch,
            )

        if training:
            self.optim["discriminator"].zero_grad()
            if use_scaler:
                self.scaler.scale(disc_fake_loss).backward()
                if clip_grad_norm is not None:
                    self.scaler.unscale_(self.optim["discriminator"])
                    torch.nn.utils.clip_grad_norm_(
                        self.model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                self.scaler.step(self.optim["discriminator"])
                self.scaler.update()
            else:
                disc_fake_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                self.optim["discriminator"].step()

        # ---------------------------------------
        # Phase 2: Discriminator on real data
        # ---------------------------------------
        real_target = batch["target"]
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=self.device.type, dtype=self.dtype),
        ):
            discriminator_real = self.model["discriminator"](real_target)
            disc_real_loss = call_loss_with_epoch(
                self.loss["adversarial"],
                discriminator_real,
                torch.ones_like(discriminator_real),
                epoch,
            )

        if training:
            self.optim["discriminator"].zero_grad()
            if use_scaler:
                self.scaler.scale(disc_real_loss).backward()
                if clip_grad_norm is not None:
                    self.scaler.unscale_(self.optim["discriminator"])
                    torch.nn.utils.clip_grad_norm_(
                        self.model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                self.scaler.step(self.optim["discriminator"])
                self.scaler.update()
            else:
                disc_real_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model["discriminator"].parameters(), max_norm=clip_grad_norm
                    )
                self.optim["discriminator"].step()

        # Store GAN-specific outputs in the unified container
        data["pred/discriminator_fake"] = pred_target
        data["pred/discriminator_real"] = real_target

        disc_total_loss = disc_real_loss.detach() + disc_fake_loss.detach()

        # ---------------------------------------
        # Phase 3: Generator
        # ---------------------------------------
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=self.device.type, dtype=self.dtype),
        ):
            pred_target = self.model["generator"](batch["input"])
            discriminator_fake = self.model["discriminator"](pred_target)
            disc_assumed_real_loss = call_loss_with_epoch(
                self.loss["adversarial"],
                discriminator_fake,
                torch.ones_like(discriminator_fake),
                epoch,
            )
            feature_matching_loss = call_loss_with_epoch(
                self.loss["feature_matching"], pred_target, batch["target"], epoch
            )
            gen_loss = disc_assumed_real_loss + feature_matching_loss

            populate_predictions(pred_target, data)

            # Postprocessor runs INSIDE autocast (unified with supervised path).
            if self.postprocessor_pipeline is not None:
                self.postprocessor_pipeline(data)

        if training:
            self.optim["generator"].zero_grad()
            if use_scaler:
                self.scaler.scale(gen_loss).backward()
                if clip_grad_norm is not None:
                    self.scaler.unscale_(self.optim["generator"])
                    torch.nn.utils.clip_grad_norm_(
                        self.model["generator"].parameters(), max_norm=clip_grad_norm
                    )
                self.scaler.step(self.optim["generator"])
                self.scaler.update()
            else:
                gen_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model["generator"].parameters(), max_norm=clip_grad_norm
                    )
                self.optim["generator"].step()

            if self.scheduler is not None and scheduler_step_per == "batch":
                for sched in self.scheduler.values():
                    sched.step()
                    self.callbacks.on_lr_update(
                        epoch, float(statistics.mean([float(v) for v in sched.get_lr()]))
                    )

        return {
            "generator": gen_loss.detach(),
            "discriminator": disc_total_loss,
        }

    def _format_metrics(self, running_loss: dict[str, float]) -> dict[str, float]:
        """GAN metrics use unprefixed names (matches legacy ``batch_loop_gan``)."""
        return {
            "generator_loss": running_loss.get("generator", 0.0),
            "discriminator_loss": running_loss.get("discriminator", 0.0),
        }

    # ------------------------------------------------------------------
    # Build-hook overrides (dict-of-modules)
    # ------------------------------------------------------------------

    @classmethod
    def _build_model(
        cls,
        config: DictConfig,
        device: torch.device,
        use_distributed: bool,
    ) -> nn.Module:
        """Instantiate ``ModuleDict({"generator", "discriminator"})`` from config."""
        model = hy_utils.instantiate(config.model)
        model.to(device)

        if device.type == "cuda" and config.get("channels_last", True):
            model = model.to(memory_format=torch.channels_last)
            logger.info("Model converted to channels_last memory format")

        if device.type == "cuda" and config.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")

        if use_distributed:
            find_unused = config.get("find_unused_parameters", False)
            model = distributed.wrap_model_ddp(model, device, find_unused_parameters=find_unused)
        logger.info("GAN model initialized: %s", type(model).__name__)
        return model

    @classmethod
    def _build_optimizer(
        cls,
        config: DictConfig,
        model: nn.Module,
    ) -> dict[str, torch.optim.Optimizer]:
        """Build per-network optimizers for generator and discriminator."""
        optimizer_class = getattr(torch.optim, config.get("optimizer", "AdamW"))
        optimizer_params = dict(config.get("optimizer_params", {}) or {})
        optimizer_params = {k: v for k, v in optimizer_params.items() if v is not None}
        return {
            "generator": optimizer_class(
                [p for p in model["generator"].parameters() if p.requires_grad],
                lr=config.lr,
                **optimizer_params,
            ),
            "discriminator": optimizer_class(
                [p for p in model["discriminator"].parameters() if p.requires_grad],
                lr=config.lr,
                **optimizer_params,
            ),
        }

    @classmethod
    def _build_scheduler(
        cls,
        config: DictConfig,
        optim: dict[str, torch.optim.Optimizer],
    ) -> dict[str, torch.optim.lr_scheduler.LRScheduler] | None:
        """Build per-network LR schedulers, or None."""
        sched_name = config.get("lr_scheduler", None)
        if sched_name is None:
            return None
        scheduler_class = getattr(torch.optim.lr_scheduler, sched_name)
        scheduler_params = dict(config.get("lr_scheduler_params", {}) or {})
        scheduler_params = {k: v for k, v in scheduler_params.items() if v is not None}
        return {
            "generator": scheduler_class(optim["generator"], **scheduler_params),
            "discriminator": scheduler_class(optim["discriminator"], **scheduler_params),
        }

    @classmethod
    def _build_loss(cls, config: DictConfig, device: torch.device) -> Any:
        """Instantiate the GAN loss dict (adversarial + feature_matching) and move to device."""
        loss = hy_utils.instantiate(config.loss)
        if isinstance(loss, dict):
            for value in loss.values():
                if isinstance(value, nn.Module):
                    value.to(device)
        elif isinstance(loss, nn.Module):
            loss.to(device)
        return loss
