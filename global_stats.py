#!/usr/bin/env python3
"""Generate feature statistics for training set.

Example:
python global_stats.py --model-type librispeech --dataset-path /home/librispeech
"""

import json
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter
import solugate_converspeech
import dataset_modules.diquest_normalspeech as diquest_normalspeech
import dataset_modules.hallym_dysarthricspeech as hallym_dysarthricspeech

import torch
from common import (
    MODEL_BASE,
    MODEL_DISABLED,
    piecewise_linear_log,
    spectrogram_transform,
)

logger = logging.getLogger()


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--model-type", type=str, choices=[MODEL_BASE, MODEL_DISABLED], required=True
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=pathlib.Path,
        help="Path to dataset.",
    )
    parser.add_argument(
        "--output-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="File to save feature statistics to. (Default: './global_stats.json')",
    )
    return parser.parse_args()


def generate_statistics(samples):
    E_x = 0
    E_x_2 = 0
    N = 0

    for idx, sample in enumerate(samples):
        mel_spec = spectrogram_transform(sample[0].squeeze()).transpose(1, 0)
        scaled_mel_spec = piecewise_linear_log(mel_spec)
        sum = scaled_mel_spec.sum(0)
        sq_sum = scaled_mel_spec.pow(2).sum(0)
        M = scaled_mel_spec.size(0)

        E_x = E_x * (N / (N + M)) + sum / (N + M)
        E_x_2 = E_x_2 * (N / (N + M)) + sq_sum / (N + M)
        N += M

        if idx % 100 == 0:
            logger.info(f"Processed {idx}")

    return E_x, (E_x_2 - E_x**2) ** 0.5


def get_dataset(args):
    if args.model_type == MODEL_BASE:
        return torch.utils.data.ConcatDataset(
            [
                solugate_converspeech.ETRISPEECH(args.dataset_path, True, "dialog"),
                diquest_normalspeech.DIQUESTSPEECH(args.dataset_path, True),
            ]
        )
    elif args.model_type == MODEL_DISABLED:
        return hallym_dysarthricspeech.KORDYSARTHRICSPEECH(args.dataset_path, True, subset_type="뇌신경장애")
    else:
        raise ValueError(f"Encountered unsupported model type {args.model_type}.")


def cli_main():
    args = parse_args()
    dataset = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=2)
    mean, stddev = generate_statistics(iter(dataloader))

    json_str = json.dumps({"mean": mean.tolist(), "invstddev": (1 / stddev).tolist()}, indent=2)

    with open(args.output_path, "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    cli_main()