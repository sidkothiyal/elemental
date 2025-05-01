"""Trainer."""
import collections
import os
from typing import Dict
from typing import Union

import fire
import hydra
import hydra.utils as hy_utils
import torch
import tqdm
from torch import nn
from torch.nn import functional as F

from callback import callback
from utils import initializer


def batch_loop_gan(
    phase: str,
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    model: Union[nn.Module, nn.ModuleDict],
    loss: Union[dict, nn.Module],
    optim: Union[dict, torch.optim.Optimizer],
    callbacks: callback.CallbackList,
    device: torch.device,
) -> Dict[str, float]:
    """Batch loop for GAN training."""
    running_loss: Dict[str, float] = collections.defaultdict(lambda: 0.0)

    total_batches = len(dataloader)
    num_steps = total_batches * epoch

    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        step = num_steps + batch_idx
        callbacks.on_batch_start(step, phase)
        input_data = {}
        target_data = {}
        pred_data = {}

        batch = {k: v.to(device) for k, v in batch.items()}

        # Train discriminator
        input_data["image"] = batch["image"]
        pred_seg_mask = model["generator"](batch["image"])
        pred_seg_mask = torch.softmax(pred_seg_mask, dim=1)

        discriminator_fake = model["discriminator"](pred_seg_mask)
        disc_fake_loss = loss["adversarial"](
            discriminator_fake, torch.zeros_like(discriminator_fake)
        )

        optim["discriminator"].zero_grad()
        disc_fake_loss.backward()
        optim["discriminator"].step()

        real_seg_mask = batch["seg_mask"]  # Shape -> (B, H, W)
        one_hot_encoded_seg_mask = F.one_hot(
            real_seg_mask, num_classes=model["generator"].out_channels
        )  # Shape -> (B, H, W, C)
        one_hot_encoded_seg_mask = (
            one_hot_encoded_seg_mask.permute(0, 3, 1, 2).float().to(device)
        )  # Shape -> (B, C, H, W)
        discriminator_real = model["discriminator"](one_hot_encoded_seg_mask)
        disc_real_loss = loss["adversarial"](
            discriminator_real, torch.ones_like(discriminator_real)
        )

        pred_data["discriminator_fake"] = pred_seg_mask
        target_data["discriminator_real"] = one_hot_encoded_seg_mask

        if phase == "train":
            optim["discriminator"].zero_grad()
            disc_real_loss.backward()
            optim["discriminator"].step()

        running_loss["discriminator"] += disc_real_loss.item() + disc_fake_loss.item()

        pred_seg_mask = model["generator"](batch["image"])
        pred_seg_mask = torch.softmax(pred_seg_mask, dim=1)

        discriminator_fake = model["discriminator"](pred_seg_mask)
        disc_assumed_real_loss = loss["adversarial"](
            discriminator_fake, torch.ones_like(discriminator_fake)
        )

        feature_matching_loss = loss["feature_matching"](pred_seg_mask, batch["seg_mask"])

        gen_loss = disc_assumed_real_loss + feature_matching_loss

        if phase == "train":
            optim["generator"].zero_grad()
            gen_loss.backward()
            optim["generator"].step()

        running_loss["generator"] += gen_loss.item()
        callbacks.on_batch_end(
            step,
            phase,
            input_data=input_data,
            target_data=target_data,
            pred_data=pred_data,
        )

    metrics = {
        "generator_loss": running_loss["generator"] / total_batches,
        "discriminator_loss": running_loss["discriminator"] / total_batches,
    }
    callbacks.log_metrics(num_steps + total_batches, phase, metrics)

    for key in running_loss:
        running_loss[key] /= total_batches

    return running_loss


def batch_loop(
    phase: str,
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    model: Union[nn.Module, nn.ModuleDict],
    loss: Union[dict, nn.Module],
    optim: Union[dict, torch.optim.Optimizer],
    callbacks: callback.CallbackList,
    device: torch.device,
) -> Dict[str, float]:
    """Batch loop for training."""
    running_loss: Dict[str, float] = collections.defaultdict(lambda: 0.0)

    total_batches = len(dataloader)
    num_steps = total_batches * epoch

    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        step = num_steps + batch_idx
        callbacks.on_batch_start(step, phase)

        input_data = {}
        target_data = {}
        pred_data = {}

        batch = {k: v.to(device) for k, v in batch.items()}

        pred = model(batch["input"])
        loss_dict = loss(pred, batch["target"])

        pred_data["pred"] = pred
        target_data["target"] = batch["target"]

        if phase == "train":
            optim.zero_grad()
            loss_dict["total"].backward()
            optim.step()

        for key in loss_dict:
            running_loss[key] += loss_dict[key].item()

        callbacks.on_batch_end(
            step,
            phase,
            input_data=input_data,
            target_data=target_data,
            pred_data=pred_data,
        )

    metrics = {
        "loss": running_loss["total"] / total_batches,
    }
    callbacks.log_metrics(num_steps + total_batches, phase, metrics)

    for key in running_loss:
        running_loss[key] /= total_batches

    return running_loss


def run_train(
    config_file: str,
    test: bool = False,
):
    """Run the trainer.

    Args:
        config_file: The path to the config file.
    """
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name=config_file)
    # Reproducibility
    initializer.set_seeds(config.seed)
    print("Config: ", config)
    device = initializer.initialize_device(config.device)
    print("Device: ", device)

    callbacks = hy_utils.instantiate(config.callbacks)
    callbacks = callback.CallbackList(callbacks)

    callbacks.update_config({"lr": config.lr, "num_epochs": config.num_epochs, "seed": config.seed})

    # Dataset
    train_loader, val_loader, test_loader = initializer.initialize_dataset(config.dataloader)

    model = hy_utils.instantiate(config.model)

    model.to(device)
    if config.mode == "gan":
        optim = {
            "generator": torch.optim.Adam(
                model["generator"].parameters(),
                lr=config.lr,
            ),
            "discriminator": torch.optim.Adam(model["discriminator"].parameters(), lr=config.lr),
        }
        scheduler = {
            "generator": torch.optim.lr_scheduler.StepLR(
                optim["generator"], step_size=10, gamma=0.01
            ),
            "discriminator": torch.optim.lr_scheduler.StepLR(
                optim["discriminator"], step_size=10, gamma=0.01
            ),
        }
    else:
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.01)

    loss = hy_utils.instantiate(config.loss)

    callbacks.update_config(
        {
            "scheduler": initializer.extract_class_names(scheduler),
            "criterion": initializer.extract_class_names(loss),
            "model": initializer.extract_class_names(model),
            "device": device,
        }
    )
    # # TODO: check if watch is time consuming and if we should log only the
    # best model (reference: https://docs.wandb.ai/ref/python/watch/)
    # wb_logger.watch(model, criterion, log="all")

    if not test:
        min_val_loss: Dict[str, float] = collections.defaultdict(lambda: float("inf"))
        callbacks.on_phase_start("train")
        for epoch in range(config.num_epochs):
            callbacks.on_epoch_start(epoch, config.num_epochs, "train")
            print("Epoch: ", epoch)

            model.train()

            # Train
            train_loss = batch_loop_gan(
                "train",
                epoch,
                train_loader,
                model,
                loss,
                optim,
                callbacks,
                device,
            )
            print(train_loss)

            if isinstance(scheduler, dict):
                for key in scheduler:
                    scheduler[key].step()
            else:
                scheduler.step()

            # Validation
            if epoch % 2 == 0:
                callbacks.on_phase_end("train")
                callbacks.on_phase_start("val")

                model.eval()
                callbacks.on_epoch_start(epoch, config.num_epochs, "val")
                val_loss = batch_loop_gan(
                    "val",
                    epoch + 1,
                    val_loader,
                    model,
                    loss,
                    optim,
                    callbacks,
                    device,
                )
                callbacks.on_epoch_end(epoch, config.num_epochs, "val")
                print(val_loss)

                if val_loss["generator"] < min_val_loss["generator"]:
                    min_val_loss["generator"] = val_loss["generator"]
                    torch.save(model["generator"].state_dict(), "best-model.pth")

                if val_loss["discriminator"] < min_val_loss["discriminator"]:
                    min_val_loss["discriminator"] = val_loss["discriminator"]
                    torch.save(model["discriminator"].state_dict(), "best-discriminator.pth")
                callbacks.on_phase_end("val")

    # Test
    if test:
        assert os.path.exists("outputs/best-model.pth"), "best-model.pth not found"

        print("Testing...")
        model.load_state_dict(torch.load("best-model.pth"))
        model.eval()
        test_loss = batch_loop_gan(
            "test",
            0,
            test_loader,
            model,
            loss,
            optim,
            callbacks,
            device,
        )
        print(test_loss)


if __name__ == "__main__":
    fire.Fire(run_train)
