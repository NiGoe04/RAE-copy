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
    n = args.num_samples
    z = torch.randn(n, *latent_size, device=device)
    y = torch.zeros(n, dtype=torch.long, device=device)
    model_kwargs = dict(y=y)
    model_fwd = model.forward
    # Sample images:
    start_time = time()
    samples: torch.Tensor = sample_fn(z, model_fwd, **model_kwargs)[-1]
    # samples = vae.decode(samples / 0.18215).sample
    samples = rae.decode(samples)
    print(f"Sampling took {time() - start_time:.2f} seconds.")

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

    # RAE reconstructions are intended for the [0, 1] image range. The
    # decoder does not hard-bound its raw output, so clamp before preprocessing.
    classifier_images = samples.float().clamp(0.0, 1.0)
    classifier_images = torch.stack([
        classifier_preprocess(image) for image in classifier_images
    ])
    with torch.no_grad():
        logits = classifier(classifier_images)
        dog_score = torch.logsumexp(logits[:, dog_indices], dim=1) - math.log(len(dog_indices))
        cat_score = torch.logsumexp(logits[:, cat_indices], dim=1) - math.log(len(cat_indices))
        class_scores = torch.stack([dog_score, cat_score], dim=1)
        class_confidences = torch.softmax(class_scores, dim=1)
        dog_confidence = class_confidences[:, 0]
        cat_confidence = class_confidences[:, 1]
        dog_mask = dog_confidence > cat_confidence
        dog_count = int(dog_mask.sum().item())
        cat_count = int((~dog_mask).sum().item())
        winning_confidence = class_confidences.max(dim=1).values

    total = dog_count + cat_count
    print(
        f"Classifier results: dogs={dog_count}/{total} ({100.0 * dog_count / total:.2f}%), "
        f"cats={cat_count}/{total} ({100.0 * cat_count / total:.2f}%)."
    )
    print(
        f"Decision confidence: mean={winning_confidence.mean().item():.4f}, "
        f"min={winning_confidence.min().item():.4f}, "
        f"max={winning_confidence.max().item():.4f}."
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": str(Path(args.config)),
        "seed": args.seed,
        "num_samples": n,
        "dogs": dog_count,
        "cats": cat_count,
        "dog_percentage": 100.0 * dog_count / total,
        "cat_percentage": 100.0 * cat_count / total,
        "mean_decision_confidence": winning_confidence.mean().item(),
        "min_decision_confidence": winning_confidence.min().item(),
        "max_decision_confidence": winning_confidence.max().item(),
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
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory in which to save the JSON results.")
    args = parser.parse_known_args()[0]
    main(args)
