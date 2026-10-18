# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch.nn as nn
import math
from time import time
import argparse
import json
from pathlib import Path
from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs
from stage1 import RAE
from torchvision.utils import save_image
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch
import sys
import os
from stage2.models import Stage2ModelProtocol
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rae_config, model_config, transport_config, sampler_config, guidance_config, misc, _, _ = parse_configs(args.config)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()  # important!
    rae.eval()
    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(
        shift_dim / shift_base)
    print(
        f"Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")
    transport = create_transport(
        **transport_config['params'],
        time_dist_shift=time_dist_shift
    )
    sampler = Sampler(transport)
    mode, sampler_params = sampler_config['mode'], sampler_config['params']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            **sampler_params
        )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            **sampler_params,
            # sampling_method=args.sampling_method,
            # diffusion_form=args.diffusion_form,
            # diffusion_norm=args.diffusion_norm,
            # last_step=args.last_step,
            # last_step_size=args.last_step_size,
            # num_steps=args.num_sampling_steps,
        )
    else:
        raise NotImplementedError(f"Invalid sampling mode {mode}.")
    
    latent_size = misc.get("latent_size", (768, 16, 16))
    # This model is trained unconditionally (num_classes: 1). The model
    # interface still requires y, so use a dummy zero label for every sample.
    model_fwd = model.forward

    # Classify generated images with an ImageNet-1K pretrained classifier.
    # These are the standard 0-based ImageNet-1K indices for the 25 dog and
    # 5 cat synsets used in the dataset.
    dog_indices = torch.tensor([
        151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
        161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
        171, 172, 173, 174, 175,
    ], device=device)
    cat_indices = torch.tensor([281, 282, 283, 284, 285], device=device)

    classifier_weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    classifier = convnext_base(weights=classifier_weights).to(device).eval()
    classifier_preprocess = classifier_weights.transforms()

    # Keep the ODE trajectory, decoded images, and classifier inputs bounded
    # by batch_size. odeint returns every requested time point, so sampling all
    # requested images in one call can consume hundreds of GiB.
    start_time = time()
    dog_count = 0
    cat_count = 0
    confidence_sum = 0.0
    min_confidence = float("inf")
    max_confidence = float("-inf")
    processed = 0
    for start in range(0, args.num_samples, args.batch_size):
        batch_n = min(args.batch_size, args.num_samples - start)
        with torch.inference_mode():
            z = torch.randn(batch_n, *latent_size, device=device)
            y = torch.zeros(batch_n, dtype=torch.long, device=device)
            latent_samples = sample_fn(z, model_fwd, y=y)[-1]
            del z, y

            # RAE reconstructions are intended for [0, 1]. Clamp before the
            # classifier's standard ImageNet preprocessing.
            samples = rae.decode(latent_samples).float().clamp(0.0, 1.0)
            del latent_samples
            classifier_images = torch.stack([
                classifier_preprocess(image) for image in samples
            ])
            del samples
            logits = classifier(classifier_images)
            del classifier_images
            dog_score = torch.logsumexp(logits[:, dog_indices], dim=1) - math.log(len(dog_indices))
            cat_score = torch.logsumexp(logits[:, cat_indices], dim=1) - math.log(len(cat_indices))
            del logits
            class_confidences = torch.softmax(torch.stack([dog_score, cat_score], dim=1), dim=1)
            dog_confidence = class_confidences[:, 0]
            cat_confidence = class_confidences[:, 1]
            winning_confidence = class_confidences.max(dim=1).values
            batch_dog_count = int((dog_confidence > cat_confidence).sum().item())
            dog_count += batch_dog_count
            cat_count += batch_n - batch_dog_count
            confidence_sum += winning_confidence.sum().item()
            min_confidence = min(min_confidence, winning_confidence.min().item())
            max_confidence = max(max_confidence, winning_confidence.max().item())
            processed += batch_n
        del class_confidences, dog_confidence, cat_confidence, winning_confidence
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Processed {processed}/{args.num_samples} samples.")

    print(f"Sampling and classification took {time() - start_time:.2f} seconds.")
    total = dog_count + cat_count
    print(
        f"Classifier results: dogs={dog_count}/{total} ({100.0 * dog_count / total:.2f}%), "
        f"cats={cat_count}/{total} ({100.0 * cat_count / total:.2f}%)."
    )
    print(
        f"Decision confidence: mean={confidence_sum / total:.4f}, "
        f"min={min_confidence:.4f}, "
        f"max={max_confidence:.4f}."
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": str(Path(args.config)),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "dogs": dog_count,
        "cats": cat_count,
        "dog_percentage": 100.0 * dog_count / total,
        "cat_percentage": 100.0 * cat_count / total,
        "mean_decision_confidence": confidence_sum / total,
        "min_decision_confidence": min_confidence,
        "max_decision_confidence": max_confidence,
    }
    with (output_dir / "classification_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved classification results to {output_dir / 'classification_results.json'}")

    # Save and display images:
    #save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=64,
                        help="Number of unconditional samples to generate.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images to sample and classify at a time.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory in which to save the JSON results.")
    args = parser.parse_known_args()[0]
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    main(args)
