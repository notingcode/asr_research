import os
import random

import torch
from pytorch_lightning import LightningDataModule
import dataset_modules.etri_converspeech as etri_converspeech
import dataset_modules.solugate_converspeech as solugate_converspeech
import dataset_modules.diquest_normalspeech as diquest_normalspeech
import dataset_modules.hallym_dysarthricspeech as hallym_dysarthricspeech
from script_normalization import cleanup_transcript
from solugate_converspeech import SUBDIR_GETTER

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


def get_sample_lengths(korspeech_dataset):
    fileid_to_target_length = {}
    
    def _etri_target_length(filename: str):
        pass
    def _diquest_length(filename: str):
        pass
    def _hallym_target_length(filename: str):
        pass
    def _solugate_target_length(filename: str):
        if filename not in fileid_to_target_length:
            subset_type, index = filename.split("_")      

            index = int(index)-1
            parent = korspeech_dataset.dataset_path
            subdir = f"{subset_type}_{(index//SUBDIR_GETTER)+1:02d}"
            filename_with_ext = f"{subdir}_scripts{korspeech_dataset._ext_txt}"

            filepath = os.path.join(parent, subdir, filename_with_ext)

            with open(filepath) as f:
                for line in f:
                    audio_default_path, transcript = line.split(" :: ", 1)
                    fileid_text = audio_default_path.split("/", 6)[-1].split(".")[0]
                    transcript = cleanup_transcript(transcript.strip())
                    if transcript is not None:
                        fileid_to_target_length[fileid_text] = len(transcript)

        return fileid_to_target_length[filename]

    if isinstance(korspeech_dataset, solugate_converspeech.SOLUGATESPEECH):
        return [_solugate_target_length(filename) for filename in korspeech_dataset._walker]
    elif isinstance(korspeech_dataset, diquest_normalspeech.DIQUESTSPEECH):
        return [_diquest_length(filename) for filename in korspeech_dataset._walker]
    elif isinstance(korspeech_dataset, hallym_dysarthricspeech.KORDYSARTHRICSPEECH):
        return [_hallym_target_length(filename) for filename in korspeech_dataset._walker]
    elif isinstance(korspeech_dataset, hallym_dysarthricspeech.KORDYSARTHRICSPEECH):
        return [_etri_target_length(filename) for filename in korspeech_dataset._walker]

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


class korSpeechDataModule(LightningDataModule):
    etrispeech_cls = etri_converspeech.ETRISPEECH

    def __init__(
        self,
        *,
        korspeech_path,
        train_transform,
        val_transform,
        test_transform,
        max_tokens=700,
        batch_size=1,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=2,
    ):
        super().__init__()
        self.korspeech_path = korspeech_path
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
            self.etrispeech_cls(self.korspeech_path, True),
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
            self.etrispeech_cls(self.korspeech_path, False),
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
        dataset = self.etrispeech_cls(self.korspeech_path, False)
        dataset = TransformDataset(dataset, self.test_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
