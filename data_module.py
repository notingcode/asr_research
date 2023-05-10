import os
import random

import torch
import torchaudio
from pytorch_lightning import LightningDataModule
import korConverseSpeech
import korDysarthricSpeech
from train_spm_base import cleanup_transcript
from korConverseSpeech import SUBDIR_GETTER


def _batch_by_token_count(idx_target_lengths, max_tokens, batch_size=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > max_tokens or (batch_size and len(current_batch) == batch_size):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


def get_sample_lengths(korconversespeech_dataset: korConverseSpeech.KORCONVERSESPEECH):
    fileid_to_target_length = {}
    
    def _target_length(filename: str):
        if filename not in fileid_to_target_length:
            subset_type, index = filename.split("_")      

            index = int(index)-1
            parent = korconversespeech_dataset.dataset_path
            subdir = f"{subset_type}_{(index//SUBDIR_GETTER)+1:02d}"
            filename = f"{subdir}_scripts{korconversespeech_dataset._ext_txt}"

            filepath = os.path.join(parent, subdir, filename)

            with open(filepath) as f:
                for line in f:
                    audio_default_path, transcript = line.split(" :: ", 1)
                    fileid_text = audio_default_path.split("/", 6)[-1].split(".")[0]
                    transcript = cleanup_transcript(transcript.strip())
                    fileid_to_target_length[fileid_text] = len(transcript)
        
        return fileid_to_target_length[filename]

    return [_target_length(filename) for filename in korconversespeech_dataset._walker]


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        lengths,
        max_tokens,
        num_buckets,
        shuffle=False,
        batch_size=None,
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_tokens >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [(idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)]
        if shuffle:
            idx_length_buckets = random.sample(idx_length_buckets, len(idx_length_buckets))
        else:
            idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[1], reverse=True)

        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_tokens,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class korConverseSpeechDataModule(LightningDataModule):
    kor_conversespeech_cls = korConverseSpeech.KORCONVERSESPEECH

    def __init__(
        self,
        *,
        kor_conversespeech_path,
        train_transform,
        val_transform,
        test_transform,
        max_tokens=1000,
        batch_size=2,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=2,
    ):
        super().__init__()
        self.kor_conversespeech_path = kor_conversespeech_path
        self.train_dataset_lengths = None
        self.val_dataset_lengths = None
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        datasets = [
            # self.kor_conversespeech_cls(self.kor_conversespeech_path + "/Training", "hobby"),
            self.kor_conversespeech_cls(self.kor_conversespeech_path, True, "dialog"),
            # self.kor_conversespeech_cls(self.kor_conversespeech_path + "/Training", "play"),
        ]

        if not self.train_dataset_lengths:
            self.train_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_tokens,
                    self.train_num_buckets,
                    batch_size=self.batch_size,
                )
                for dataset, lengths in zip(datasets, self.train_dataset_lengths)
            ]
        )
        dataset = TransformDataset(dataset, self.train_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
        )
        return dataloader

    def val_dataloader(self):
        datasets = [
            # self.kor_conversespeech_cls(self.kor_conversespeech_path + "/Validation", "hobby"),
            self.kor_conversespeech_cls(self.kor_conversespeech_path, False, "dialog"),
            # self.kor_conversespeech_cls(self.kor_conversespeech_path + "/Validation", "play"),
        ]

        if not self.val_dataset_lengths:
            self.val_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_tokens,
                    1,
                    batch_size=self.batch_size,
                )
                for dataset, lengths in zip(datasets, self.val_dataset_lengths)
            ]
        )
        dataset = TransformDataset(dataset, self.val_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        dataset = self.kor_conversespeech_cls(self.kor_conversespeech_path, False, "life")
        dataset = TransformDataset(dataset, self.test_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
    
class korDysarthricSpeechDataModule(LightningDataModule):
    kor_dysarthricspeech_cls = korDysarthricSpeech.KORDYSARTHRICSPEECH

    def __init__(
        self,
        *,
        kor_dysarthricspeech_path,
        train_transform,
        val_transform,
        test_transform,
        max_tokens=6000,
        batch_size=2,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=2,
    ):
        super().__init__()
        self.kor_dysarthricspeech_path = kor_dysarthricspeech_path
        self.train_dataset_lengths = None
        self.val_dataset_lengths = None
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        datasets = [
            self.kor_dysarthricspeech_cls(self.kor_dysarthricspeech_path + "/Training", "hobby"),
            self.kor_dysarthricspeech_cls(self.kor_dysarthricspeech_path + "/Training", "dialog"),
            self.kor_dysarthricspeech_cls(self.kor_dysarthricspeech_path + "/Training", "play"),
        ]

        if not self.train_dataset_lengths:
            self.train_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_tokens,
                    self.train_num_buckets,
                    batch_size=self.batch_size,
                )
                for dataset, lengths in zip(datasets, self.train_dataset_lengths)
            ]
        )
        dataset = TransformDataset(dataset, self.train_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
        )
        return dataloader

    def val_dataloader(self):
        datasets = [
            self.kor_dysarthricspeech_cls(self.kor_dysarthricspeech_path + "/Validation", "hobby"),
            self.kor_dysarthricspeech_cls(self.kor_dysarthricspeech_path + "/Validation", "dialog"),
            self.kor_dysarthricspeech_cls(self.kor_dysarthricspeech_path + "/Validation", "play"),
        ]

        if not self.val_dataset_lengths:
            self.val_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_tokens,
                    1,
                    batch_size=self.batch_size,
                )
                for dataset, lengths in zip(datasets, self.val_dataset_lengths)
            ]
        )
        dataset = TransformDataset(dataset, self.val_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        dataset = self.kor_dysarthricspeech_cls(self.kor_dysarthricspeech_path + "/Validation", "life")
        dataset = TransformDataset(dataset, self.test_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader