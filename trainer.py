import os
import random
import numpy as np
import torch
import torch.nn as nn
import wandb


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def init_wandb(exp_name="EXP", user_name="elemental_user"):
    return wandb.init(
        api_key="WANDB_API_KEY",
        project="elemental",
        entity=user_name,
        name=exp_name,
    )

def batch_loop(phase, dataloader, model, criterion, optimizer, device):
    running_loss = 0.0

    for batch in dataloader:
        input_data, label1, label2, label3, label4, label5 = batch.to(device)

        predictions = model(input_data)
        labels = [label1, label2, label3, label4, label5]
        loss = criterion(predictions, labels)

        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def predict(model, dataloader, criterion, device):
    pass


def run_train(
    lr: float = 0.0001,
    num_epochs: int = 100,
    test: bool = False,
    device: str = "cuda",
    seed: int = 30,
):
    # Reproducibility and Logging
    set_seeds(seed)
    wb_logger = init_wandb("elemental_trial1")
    wb_logger.config.update({"lr": lr, "num_epochs": num_epochs, "seed": seed})

    # Dataset
    train_loader, val_loader, test_loader = None, None, None

    model = None
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.01)
    criterion = nn.CrossEntropyLoss(weight=None)

    wb_logger.config.add({"scheduler:" scheduler.__class__.__name__,
                        "criterion": criterion.__class__.__name__,
                        "model": model.__class__.__name__,
                        "device": device})
    # TODO: check if watch is time consuming and if we should log only the best model (reference: https://docs.wandb.ai/ref/python/watch/)
    wb_logger.watch(model, criterion, log="all")

    if not test:
        min_val_loss = float("inf")
        for epoch in range(num_epochs):
            print("Epoch: ", epoch)

            # Train
            model.train()
            train_loss = batch_loop(
                "train", train_loader, model, criterion, optim, device
            )
            print("\t train_loss:", round(train_loss.item(), 3))

            # Validation
            if epoch % 2 == 0:
                model.eval()
                val_loss = batch_loop(
                    "val", val_loader, model, criterion, optim, device
                )
                print("\t val_loss:", round(val_loss.item(), 3))

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), "best-model.pth")

                scheduler.step()

    # Test
    if test:
        assert os.path.exists("outputs/best-model.pth"), "best-model.pth not found"

        print("Testing...")
        model.load_state_dict(torch.load("best-model.pth"))
        model.eval()
        test_loss = predict()
        print("\t test_loss:", test_loss.item())


if __name__ == "__main__":
    main()
