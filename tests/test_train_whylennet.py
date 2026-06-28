"""Tests that WhyLeNet matches the README design and can train on MNIST."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_whylennet import (
    POSITION_WEIGHTS,
    DigitBank,
    WhyLeNet,
    WhyNeuron,
    digit_to_image_differentiable,
    get_unit,
)


def _load_tiny_mnist(data_dir: Path) -> tuple[datasets.MNIST, DataLoader]:
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    loader = DataLoader(Subset(train_full, range(128)), batch_size=32, shuffle=True, num_workers=0)
    return train_full, loader


def test_digit_to_image_differentiable_shape() -> None:
    digit_bank = torch.randn(10, 1, 28, 28)
    probs = torch.softmax(torch.randn(4, 10), dim=-1)
    images = digit_to_image_differentiable(probs, digit_bank)
    assert images.shape == (4, 1, 28, 28)


def test_get_unit_prefers_closest_digit() -> None:
    digits = torch.arange(10, dtype=torch.float32)
    dist = get_unit(torch.tensor([42.0]), 10.0, digits)
    assert dist.argmax(dim=-1).item() == 4


def test_why_neuron_emits_five_resynthesized_channels(tmp_path: Path) -> None:
    train_full, _ = _load_tiny_mnist(tmp_path / "data")
    digit_bank = DigitBank(train_full)
    neuron = WhyNeuron(digit_bank)

    logits = torch.randn(2, 10)
    output_images, n = neuron(logits)

    assert output_images.shape == (2, len(POSITION_WEIGHTS), 28, 28)
    assert n.shape == (2, 1)
    assert torch.isfinite(output_images).all()
    assert torch.isfinite(n).all()


def test_network_trains_on_mnist(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    train_full, _ = _load_tiny_mnist(data_dir)
    test_full = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(Subset(test_full, range(64)), batch_size=32, shuffle=False, num_workers=0)

    torch.manual_seed(0)
    device = torch.device("cpu")
    digit_bank = DigitBank(train_full)
    model = WhyLeNet(digit_bank=digit_bank, hidden_neurons=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fixed_batch = DataLoader(Subset(train_full, range(64)), batch_size=64, shuffle=False, num_workers=0)
    images, labels = next(iter(fixed_batch))
    images = images.to(device)
    labels = labels.to(device)

    model.train()
    losses: list[float] = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]
    assert all(torch.isfinite(torch.tensor(losses)))

    model.eval()
    test_images, test_labels = next(iter(test_loader))
    logits = model(test_images.to(device))
    loss = F.cross_entropy(logits, test_labels.to(device))
    assert torch.isfinite(loss)
    assert logits.shape == (test_labels.size(0), 10)
