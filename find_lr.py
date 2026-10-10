import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from stage2.transport import create_transport
from utils.model_utils import instantiate_from_config
from utils.train_utils import center_crop_arr, parse_configs


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="results/lr_finder")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--start-lr", type=float, default=1e-6)
    parser.add_argument("--end-lr", type=float, default=1e-3)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if args.precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported on this device.")

    use_bf16 = args.precision == "bf16"

    os.makedirs(args.output_dir, exist_ok=True)

    config = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        _sampler_config,
        _guidance_config,
        misc_config,
        _training_config,
        _eval_config,
    ) = parse_configs(config)

    misc = OmegaConf.to_container(misc_config, resolve=True)
    transport_cfg = OmegaConf.to_container(
        transport_config,
        resolve=True,
    )

    latent_size = tuple(
        misc.get("latent_size", [768, 16, 16])
    )
    shift_dim = misc.get(
        "time_dist_shift_dim",
        math.prod(latent_size),
    )
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    rae = instantiate_from_config(rae_config).to(device).eval()
    model = instantiate_from_config(model_config).to(device).train()

    for parameter in rae.parameters():
        parameter.requires_grad_(False)

    transform = transforms.Compose([
        transforms.Lambda(
            lambda image: center_crop_arr(image, args.image_size)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(
        args.data_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    batches = infinite_loader(loader)

    transport_params = dict(
        transport_cfg.get("params", {})
    )
    transport_params.pop("time_dist_shift", None)
    transport_params["time_dist_shift"] = time_dist_shift
    transport_params["velocity_loss_type"] = "squared_l2"

    transport = create_transport(**transport_params)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.start_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    lr_multiplier = (
        args.end_lr / args.start_lr
    ) ** (1.0 / max(args.num_steps - 1, 1))

    learning_rates = []
    raw_losses = []
    smoothed_losses = []

    smoothing = 0.90
    average_loss = 0.0
    best_loss = float("inf")
    stopped_reason = "completed"

    for step in range(args.num_steps):
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)

        images, _labels = next(batches)
        images = images.to(device, non_blocking=True)

        # Keep the frozen RAE encoder in FP32.
        with torch.no_grad():
            latents = rae.encode(images)

        labels = torch.zeros(
            latents.shape[0],
            dtype=torch.long,
            device=device,
        )

        # Use BF16 only for the trainable Stage-2 forward/loss.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss_terms = transport.training_losses(
                model,
                latents,
                model_kwargs={"y": labels},
            )
            loss = loss_terms["loss"].mean()

        loss_value = float(
            loss.float().detach().cpu()
        )

        if not math.isfinite(loss_value):
            stopped_reason = (
                f"non-finite loss at lr={learning_rate:.6g}"
            )
            break

        average_loss = (
            smoothing * average_loss
            + (1.0 - smoothing) * loss_value
        )
        smoothed_loss = average_loss / (
            1.0 - smoothing ** (step + 1)
        )

        learning_rates.append(learning_rate)
        raw_losses.append(loss_value)
        smoothed_losses.append(smoothed_loss)

        if step > 10 and smoothed_loss > 4.0 * best_loss:
            stopped_reason = "loss diverged"
            break

        best_loss = min(best_loss, smoothed_loss)

        loss.float().backward()
        optimizer.step()

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] *= lr_multiplier

        if (step + 1) % 20 == 0:
            print(
                f"step {step + 1}/{args.num_steps}: "
                f"lr={learning_rate:.3e}, "
                f"smoothed_loss={smoothed_loss:.6f}",
                flush=True,
            )

    if not smoothed_losses:
        raise RuntimeError(
            f"LR finder produced no measurements: {stopped_reason}"
        )

    minimum_index = min(
        range(len(smoothed_losses)),
        key=smoothed_losses.__getitem__,
    )

    result = {
        "learning_rates": learning_rates,
        "raw_losses": raw_losses,
        "smoothed_losses": smoothed_losses,
        "minimum_loss_lr": learning_rates[minimum_index],
        "steps_completed": len(learning_rates),
        "stopped_reason": stopped_reason,
        "dataset_size": len(dataset),
        "batch_size": args.batch_size,
        "precision": args.precision,
        "loss_type": "squared_l2",
    }

    json_path = os.path.join(
        args.output_dir,
        "lr_range_test.json",
    )
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    plot_path = os.path.join(
        args.output_dir,
        "lr_range_test.png",
    )

    plt.figure(figsize=(8, 5))
    plt.semilogx(
        learning_rates,
        raw_losses,
        alpha=0.3,
        label="raw loss",
    )
    plt.semilogx(
        learning_rates,
        smoothed_losses,
        linewidth=2,
        label="smoothed loss",
    )
    plt.axvline(
        result["minimum_loss_lr"],
        color="red",
        linestyle="--",
        label="minimum loss LR",
    )
    plt.xlabel("Learning rate")
    plt.ylabel("Squared velocity loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()