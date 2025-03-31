"""Trainer."""
import os

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
    phase,
    epoch,
    dataloader,
    model_generator,
    model_discriminator,
    feature_matching_criterion,
    adversarial_criterion,
    optimizer_generator,
    optimizer_discriminator,
    callbacks,
    device,
):
    """Batch loop for GAN training."""
    running_generator_loss = 0.0
    running_discriminator_loss = 0.0

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
        pred_seg_mask = model_generator(batch["image"])
        pred_seg_mask = torch.softmax(pred_seg_mask, dim=1)

        discriminator_fake = model_discriminator(pred_seg_mask)
        disc_fake_loss = adversarial_criterion(
            discriminator_fake, torch.zeros_like(discriminator_fake)
        )

        optimizer_discriminator.zero_grad()
        disc_fake_loss.backward()
        optimizer_discriminator.step()

        real_seg_mask = batch["seg_mask"]  # Shape -> (B, H, W)
        one_hot_encoded_seg_mask = F.one_hot(
            real_seg_mask, num_classes=model_generator.out_channels
        )  # Shape -> (B, H, W, C)
        one_hot_encoded_seg_mask = (
            one_hot_encoded_seg_mask.permute(0, 3, 1, 2).float().to(device)
        )  # Shape -> (B, C, H, W)
        discriminator_real = model_discriminator(one_hot_encoded_seg_mask)
        disc_real_loss = adversarial_criterion(
            discriminator_real, torch.ones_like(discriminator_real)
        )

        pred_data["discriminator_fake"] = pred_seg_mask
        target_data["discriminator_real"] = one_hot_encoded_seg_mask

        if phase == "train":
            optimizer_discriminator.zero_grad()
            disc_real_loss.backward()
            optimizer_discriminator.step()

        running_discriminator_loss += disc_real_loss.item() + disc_fake_loss.item()

        pred_seg_mask = model_generator(batch["image"])
        pred_seg_mask = torch.softmax(pred_seg_mask, dim=1)

        discriminator_fake = model_discriminator(pred_seg_mask)
        disc_assumed_real_loss = adversarial_criterion(
            discriminator_fake, torch.ones_like(discriminator_fake)
        )

        feature_matching_loss = feature_matching_criterion(pred_seg_mask, batch["seg_mask"])

        gen_loss = disc_assumed_real_loss + feature_matching_loss

        if phase == "train":
            optimizer_generator.zero_grad()
            gen_loss.backward()
            optimizer_generator.step()

        running_generator_loss += gen_loss.item()
        callbacks.on_batch_end(
            step,
            phase,
            input_data=input_data,
            target_data=target_data,
            pred_data=pred_data,
        )

    metrics = {
        "generator_loss": running_generator_loss / total_batches,
        "discriminator_loss": running_discriminator_loss / total_batches,
    }
    callbacks.log_metrics(num_steps + total_batches, phase, metrics)

    return (
        running_generator_loss / total_batches,
        running_discriminator_loss / total_batches,
    )


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

    model_generator = hy_utils.instantiate(config.model_generator)
    model_discriminator = hy_utils.instantiate(config.model_discriminator)

    model_generator.to(device)
    model_discriminator.to(device)
    optim_generator = torch.optim.Adam(
        model_generator.parameters(),
        lr=config.lr,
    )
    optim_discriminator = torch.optim.Adam(model_discriminator.parameters(), lr=config.lr)
    scheduler_generator = torch.optim.lr_scheduler.StepLR(optim_generator, step_size=10, gamma=0.01)
    scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
        optim_discriminator, step_size=10, gamma=0.01
    )
    feature_matching_criterion = nn.CrossEntropyLoss()
    adversarial_criterion = nn.BCEWithLogitsLoss()

    callbacks.update_config(
        {
            "scheduler": {
                "generator": scheduler_generator.__class__.__name__,
                "discriminator": scheduler_discriminator.__class__.__name__,
            },
            "criterion": [
                feature_matching_criterion.__class__.__name__,
                adversarial_criterion.__class__.__name__,
            ],
            "model": {
                "generator": model_generator.__class__.__name__,
                "discriminator": model_discriminator.__class__.__name__,
            },
            "device": device,
        }
    )
    # # TODO: check if watch is time consuming and if we should log only the
    # best model (reference: https://docs.wandb.ai/ref/python/watch/)
    # wb_logger.watch(model, criterion, log="all")

    if not test:
        min_val_generator_loss = float("inf")
        min_val_discriminator_loss = float("inf")
        callbacks.on_phase_start("train")
        for epoch in range(config.num_epochs):
            callbacks.on_epoch_start(epoch, config.num_epochs, "train")
            print("Epoch: ", epoch)

            model_generator.train()
            model_discriminator.train()

            # Train
            train_generator_loss, train_discriminator_loss = batch_loop_gan(
                "train",
                epoch,
                train_loader,
                model_generator,
                model_discriminator,
                feature_matching_criterion,
                adversarial_criterion,
                optim_generator,
                optim_discriminator,
                callbacks,
                device,
            )
            print(
                "\t train_generator_loss:",
                round(train_generator_loss, 3),
                "\t train_discriminator_loss:",
                round(train_discriminator_loss, 3),
            )
            scheduler_generator.step()
            scheduler_discriminator.step()

            # Validation
            if epoch % 2 == 0:
                callbacks.on_phase_end("train")
                callbacks.on_phase_start("val")

                model_generator.eval()
                model_discriminator.eval()
                callbacks.on_epoch_start(epoch, config.num_epochs, "val")
                val_generator_loss, val_discriminator_loss = batch_loop_gan(
                    "val",
                    epoch + 1,
                    val_loader,
                    model_generator,
                    model_discriminator,
                    feature_matching_criterion,
                    adversarial_criterion,
                    optim_generator,
                    optim_discriminator,
                    callbacks,
                    device,
                )
                callbacks.on_epoch_end(epoch, config.num_epochs, "val")
                print(
                    "\t val_generator_loss:",
                    round(val_generator_loss, 3),
                    "\t val_discriminator_loss:",
                    round(val_discriminator_loss, 3),
                )

                if val_generator_loss < min_val_generator_loss:
                    min_val_generator_loss = val_generator_loss
                    torch.save(model_generator.state_dict(), "best-model.pth")

                if val_discriminator_loss < min_val_discriminator_loss:
                    min_val_discriminator_loss = val_discriminator_loss
                    torch.save(
                        model_discriminator.state_dict(),
                        "best-discriminator.pth",
                    )
                callbacks.on_phase_end("val")

    # Test
    if test:
        assert os.path.exists("outputs/best-model.pth"), "best-model.pth not found"

        print("Testing...")
        model_generator.load_state_dict(torch.load("best-model.pth"))
        model_discriminator.load_state_dict(torch.load("best-discriminator.pth"))
        model_generator.eval()
        model_discriminator.eval()
        generator_loss, discriminator_loss = batch_loop_gan(
            "test",
            0,
            test_loader,
            model_generator,
            model_discriminator,
            feature_matching_criterion,
            adversarial_criterion,
            optim_generator,
            optim_discriminator,
            callbacks,
            device,
        )
        print(
            "\t generator_loss:",
            round(generator_loss, 3),
            "\t discriminator_loss:",
            round(discriminator_loss, 3),
        )


if __name__ == "__main__":
    fire.Fire(run_train)
