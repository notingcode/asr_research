import os
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset
from untar_unzip import _extract_zip, _load_waveform
from train_spm_base import cleanup_transcript
from common import(
    SAMPLE_RATE,
    TRAIN_SUBDIR_NAME,
    VALID_SUBDIR_NAME,
)

N_DIRECTORIES_STRIPPED = 0
SUBDIR_HEADER = "일반남여"

def unpack_diquestSpeech(source_path: Union[str, Path]):
    ext_archive = ".zip"
        
    zip_files = Path(source_path).glob(f"*{ext_archive}")
    
    args = []
    
    for file in zip_files:
        args.append((file.as_posix(), source_path, False, N_DIRECTORIES_STRIPPED))
    
    pool = mp.Pool(min(mp.cpu_count(), len(args)))
    
    pool.starmap(_extract_zip, args, chunksize=1)


def _get_korConverseSpeech_metadata(
    filename: str, dataset_path: str, ext_audio: str, ext_txt: str,
) -> Tuple[str, int, str]:



    # Load text
    with open(filename) as f:
        transcript = cleanup_transcript(f.readline().strip())
        if transcript is None:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {filename}")

    return (
        audio_filepath,
        SAMPLE_RATE,
        transcript,
    )


class DIQUESTSPEECH(Dataset):
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

        unpack_diquestSpeech(self.dataset_path)
        
        audio_files_path = Path(self.dataset_path).rglob("*"+self._ext_audio)
        
        self._walker = []

        for path in audio_files_path:
            with open(path.with_suffix(self._ext_txt)) as f:
                transcript = cleanup_transcript(f.readline().strip())
                if transcript is not None:
                    self._walker.append(path.stem)                        

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
        return _get_korConverseSpeech_metadata(file_name, self.dataset_path, self._ext_audio, self._ext_txt)

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
