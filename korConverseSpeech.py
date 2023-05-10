import os
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset
from untar_unzip import _extract_tar, _load_waveform
from train_spm_base import cleanup_transcript
from common import SAMPLE_RATE

N_DIRECTORIES_STRIPPED = 9
_DATA_SUBSETS = [
    "broadcast",
    "hobby",
    "dialog",
    "life",
    "weather",
    "economy",
    "play",
    "shopping",
    "all"
]
TRAIN_SUBDIR_NAME = "Training"
VALID_SUBDIR_NAME = "Validation"
SUBDIR_GETTER = 100000
INDEX_SUBDIR_GETTER = 1000

def _unpack_korConverseSpeech(source_path: Union[str, Path], subset_type: str):
    ext_archive = ".tar"
        
    if subset_type == "all":
        tar_files = Path(source_path).glob(f"*{ext_archive}*")
    else:
        tar_files = Path(source_path).glob(f"*{subset_type}_*{ext_archive}*")
    
    args = []
    
    for file in tar_files:
        args.append((file.as_posix(), source_path, False, N_DIRECTORIES_STRIPPED))
    
    pool = mp.Pool(min(mp.cpu_count(), len(args)))
    
    pool.starmap(_extract_tar, args, chunksize=1)


def _get_korConverseSpeech_metadata(
    filename: str, dataset_path: str, ext_txt: str,
) -> Tuple[str, int, str]:
    subset_type, index = filename.split("_")

    index = int(index)-1

    transcript_file = filename+ext_txt
    subset_subdir = f"{subset_type}_{(index//SUBDIR_GETTER)+1:02d}"
    indexed_dir = f"{(index//INDEX_SUBDIR_GETTER)+1:03d}"
    
    filepath = os.path.join(dataset_path, subset_subdir, indexed_dir, transcript_file)

    # Load text
    with open(filepath) as f:
        transcript = cleanup_transcript(f.readline().strip())
        if len(transcript) == 0:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {filepath}")

    return (
        filepath,
        SAMPLE_RATE,
        transcript,
    )


class KORCONVERSESPEECH(Dataset):
    """
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset_type (str): Type of subset to be trained on.
    """

    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(
        self,
        root: Union[str, Path],
        training: bool,
        subset_type: str,
    ) -> None:
        self.root = os.fspath(root)
        self.subset_type = subset_type.lower()
        
        if training:
            dataset_path = os.path.join(root, TRAIN_SUBDIR_NAME)
        else:
            dataset_path = os.path.join(root, VALID_SUBDIR_NAME)
            
        self.dataset_path = dataset_path

        assert(self.subset_type in _DATA_SUBSETS)

        _unpack_korConverseSpeech(self.dataset_path, self.subset_type)
        
        if self.subset_type == "all":
            audio_files_path = Path(self.dataset_path).rglob("*"+self._ext_audio)
        else:
            audio_files_path = Path(self.dataset_path).rglob(f"{subset_type}_*"+self._ext_audio)
        
        file_paths = []

        for path in audio_files_path:
            with open(path.with_suffix(self._ext_txt)) as f:
                transcript = cleanup_transcript(f.readline().strip())
                if len(transcript) > 0:
                    file_paths.append(path.stem)
                        
        self._walker = file_paths

    def get_metadata(self, n: int) -> Tuple[str, int, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Sample rate
            str:
                Transcript
        """
        file_name = self._walker[n]
        return _get_korConverseSpeech_metadata(file_name, self.dataset_path, self._ext_txt)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(metadata[0], metadata[1])
        return (waveform, ) + metadata[1:]

    def __len__(self) -> int:
        return len(self._walker)
