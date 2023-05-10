import os
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset
from untar_unzip import _extract_zip, _load_waveform
from common import SAMPLE_RATE

N_DIRECTORIES_STRIPPED = 0
DYS_SAMPLE_RATE = 44100
_DATA_SUBSETS = [
    "뇌신경장애",
    "언어청각장애",
    "후두장애",
]


def _unpack_korDysarthricSpeech(source_path, subset_type):
    ext_archive = ".zip"
    
    assert(subset_type in _DATA_SUBSETS)
    
    if subset_type == "all":
        zip_files = Path(source_path).glob(f"*{ext_archive}")
    else:
        zip_files = Path(source_path).glob(f"*{subset_type}{ext_archive}")
    
    args = []
    
    for file in zip_files:
        args.append((file.as_posix(), file.with_suffix("").as_posix(), N_DIRECTORIES_STRIPPED))
    
    pool = mp.Pool(min(mp.cpu_count(), len(args)))
    
    pool.starmap(_extract_zip, args, chunksize=1)


def _get_korDysarthricSpeech_metadata(
    fileid: str, root: str, folder: str, ext_audio: str, ext_json: str
) -> Tuple[str, int, str]:
    
    filepath = fileid
    
    transcript = ""

    return (
        filepath,
        SAMPLE_RATE,
        transcript,
    )


class KORDYSARTHRICSPEECH(Dataset):
    """
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ````)
    """

    _ext_json = ".json"
    _ext_audio = ".wav"

    def __init__(
        self,
        root: Union[str, Path],
        subset_type: str = 'all',
    ) -> None:

        root = os.fspath(root)

        _unpack_korDysarthricSpeech(root, subset_type)

        self._walker = sorted(str(p.stem) for p in Path(root).glob("*/*/*" + self._ext_audio))

    def get_metadata(self, n: int) -> Tuple[str, int, str, int, int, int]:
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
        fileid = self._walker[n]
        return _get_korDysarthricSpeech_metadata(fileid, self._archive, self._path, self._ext_audio, self._ext_json)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
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
