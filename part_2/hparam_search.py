import os
import pickle
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import SmallCNN
from ray import train, tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
from torchvision import transforms

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/")


def load_data(data_dir=DATA_DIR):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset


def train_cifar(config, data_dir=DATA_DIR):
    """The training loop for the model over te CIFAR-100 dataset

    :param config: A dictionnary containing the hyperparameters for the trial
    :param data_dir: Path to the folder where to download the dataset
    """

    model = SmallCNN(3, 100, config["h_1"], config["h_2"], config["p_drop"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(
        start_epoch, config["epochs"] + 1
    ):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )


def test_accuracy(model, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples, max_num_epochs):
    """
    :param nu_samples: Set the number of time that the tuner should sample fro parameter
    space
    :param max_num_epochs: The maximum number of epochs
    """

    param_space = {
        "epochs": max_num_epochs,
        "h_1": tune.grid_search([64, 128]),
        "h_2": tune.grid_search([512, 1024]),
        "p_drop": tune.loguniform(0.1, 0.5),
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": 64,
    }

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        grace_period=3,
        reduction_factor=2,
    )

    trainable_with_resources = tune.with_resources(train_cifar, {"cpu": 6, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=param_space,
        run_config=train.RunConfig(
            name="ray-experiement",
            stop={"time_total_s": 3600},
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=5, checkpoint_score_attribute="accuracy"
            ),
            storage_path=os.path.abspath("./outputs/"),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler, num_samples=num_samples, metric="loss", mode="min"
        ),
    )
    result_grid = tuner.fit()

    best_result = result_grid.get_best_result("accuracy", "max", "last")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['accuracy']}")

    best_trained_model = SmallCNN(
        3,
        100,
        best_result.config["h_1"],
        best_result.config["h_2"],
        best_result.config["p_drop"],
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    best_trained_model.to(device)

    best_checkpoint = best_result.checkpoint()

    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])
        test_acc = test_accuracy(best_trained_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main(num_samples=3, max_num_epochs=20)
