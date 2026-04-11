"""Training script for the SHD spiking neural network."""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

from shd_snn.data import get_shd_dataset, sparse_data_generator
from shd_snn.model import SHDModel

# Dataset constants
NB_INPUTS = 700
NB_OUTPUTS = 20
MAX_TIME = 1.4


def train_epoch(model, x_data, y_data, optimizer, batch_size, device):
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    local_loss = []
    for x_local, y_local in sparse_data_generator(
        x_data, y_data, batch_size, model.nb_steps, NB_INPUTS, MAX_TIME, device
    ):
        output, (_, spks) = model(x_local.to_dense(), batch_size)
        m, _ = torch.max(output, 1)
        log_p_y = log_softmax_fn(m)

        reg_loss = 2e-6 * torch.sum(spks)
        reg_loss += 2e-6 * torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0) ** 2)

        loss_val = loss_fn(log_p_y, y_local) + reg_loss

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        local_loss.append(loss_val.item())

    return np.mean(local_loss)


def compute_accuracy(model, x_data, y_data, batch_size, device):
    accs = []
    for x_local, y_local in sparse_data_generator(
        x_data, y_data, batch_size, model.nb_steps, NB_INPUTS, MAX_TIME, device, shuffle=False
    ):
        output, _ = model(x_local.to_dense(), batch_size)
        m, _ = torch.max(output, 1)
        _, am = torch.max(m, 1)
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
    return np.mean(accs)


def main():
    parser = argparse.ArgumentParser(description="Train SHD spiking neural network")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adamax optimizer")
    parser.add_argument("--nb-hidden", type=int, default=200, help="Number of hidden neurons")
    parser.add_argument("--nb-steps", type=int, default=100, help="Number of simulation time steps")
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), os.pardir, "data"),
        help="Directory for cached dataset files",
    )
    parser.add_argument(
        "--save-path", type=str, default="model.pt",
        help="Path to save trained model weights",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download / locate dataset
    data_path = get_shd_dataset(args.data_dir)
    train_file = h5py.File(os.path.join(data_path, "shd_train.h5"), "r")
    test_file = h5py.File(os.path.join(data_path, "shd_test.h5"), "r")

    x_train, y_train = train_file["spikes"], train_file["labels"]
    x_test, y_test = test_file["spikes"], test_file["labels"]

    model = SHDModel(
        nb_inputs=NB_INPUTS,
        nb_hidden=args.nb_hidden,
        nb_outputs=NB_OUTPUTS,
        nb_steps=args.nb_steps,
        device=device,
    )
    optimizer = torch.optim.Adamax(model.params, lr=args.lr, betas=(0.9, 0.999))

    for epoch in range(args.epochs):
        loss = train_epoch(model, x_train, y_train, optimizer, args.batch_size, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={loss:.5f}")

    train_acc = compute_accuracy(model, x_train, y_train, args.batch_size, device)
    test_acc = compute_accuracy(model, x_test, y_test, args.batch_size, device)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    train_file.close()
    test_file.close()


if __name__ == "__main__":
    main()
