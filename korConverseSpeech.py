import os
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset
from untar_unzip import _extract_tar, _load_waveform
from train_spm_base import cleanup_transcript

N_DIRECTORIES_STRIPPED = 9
SAMPLE_RATE = 16000
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


def _unpack_korConverseSpeech(source_path, subset_type: str):
    ext_archive = ".tar"
    
    assert(subset_type in _DATA_SUBSETS)
    
    if subset_type == "all":
        tar_files = Path(source_path).glob(f"*원천*{ext_archive}*")
    else:
        tar_files = Path(source_path).glob(f"*원천*{subset_type}_*{ext_archive}*")
    
    args = []
    
    for file in tar_files:
        to_path = file.with_suffix("")
        if file.name.endswith("tar.gz"):
            to_path = to_path.with_suffix("")
            
        to_path.mkdir(exist_ok=True)
        
        args.append((file.as_posix(), to_path.as_posix(), N_DIRECTORIES_STRIPPED))
    
    pool = mp.Pool(min(mp.cpu_count(), len(args)))
    
    pool.starmap(_extract_tar, args, chunksize=1)


def _get_korConverseSpeech_metadata(
    filepath: Path, ext_txt: str,
) -> Tuple[str, int, str]:

    # Load text
    with open(filepath.with_suffix(ext_txt)) as f:
        transcript = cleanup_transcript(f.readline().strip())
        if len(transcript) == 0:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {filepath.name}")

    return (
        filepath.as_posix(),
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
        subset_type: str
    ) -> None:
        self.root = os.fspath(root)

        _unpack_korConverseSpeech(root, subset_type)
        
        if subset_type == "all":
            audio_files_path = Path(root).rglob("*"+self._ext_audio)
        else:
            audio_files_path = Path(root).rglob(f"{subset_type}_*"+self._ext_audio)
        
        file_paths = []

        for path in audio_files_path:
            with open(path.with_suffix(self._ext_txt)) as f:
                transcript = cleanup_transcript(f.readline().strip())
                if len(transcript) > 0:
                    file_paths.append(path)
                        
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
        file_path = self._walker[n]
        return _get_korConverseSpeech_metadata(file_path, self._ext_txt)

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
