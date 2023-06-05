import os
import multiprocessing as mp
import json
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset
from untar_unzip import _extract_zip, _load_waveform
from common import(
    SAMPLE_RATE,
    TRAIN_SUBDIR_NAME,
    VALID_SUBDIR_NAME,
)

N_DIRECTORIES_STRIPPED = 1

TOP_SUBDIR_NAME = "01.데이터"
LABEL_DIR_NAME = "라벨링데이터"
SOURCE_DIR_NAME = "원천데이터"

def _get_script_from_json(transcript_path):
    with open(transcript_path) as f:
        data = json.load(f)
        return data['Transcript'].strip()

def _unpack_dysarthricSpeech(datasets_parentPath):
    ext_archive = ".zip"

    zip_files = Path(datasets_parentPath).glob(f"*/*{ext_archive}")
    
    args = []
    
    for file in zip_files:
        args.append((file.as_posix(), file.with_suffix("").as_posix(), N_DIRECTORIES_STRIPPED))
    
    pool = mp.Pool(min(mp.cpu_count(), len(args)))
    
    pool.starmap(_extract_zip, args, chunksize=1)


def _get_korDysarthricSpeech_metadata(
    relative_filepath: str, dataset_path: str, ext_audio: str, ext_script: str
) -> Tuple[str, int, str]:
    
    audio_relative_filepath = f"{relative_filepath}{ext_audio}"
    script_relative_filepath = f"{relative_filepath}{ext_script}"
    
    audio_filepath = os.path.join(dataset_path, SOURCE_DIR_NAME, audio_relative_filepath)
    transcript_filepath = os.path.join(dataset_path, LABEL_DIR_NAME, script_relative_filepath)
    
    transcript = _get_script_from_json(transcript_filepath)

    return (
        audio_filepath,
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
        training: bool,
    ) -> None:

        self.root = os.fspath(root)
        
        if training:
            dataset_path = os.path.join(root, TOP_SUBDIR_NAME, f"1.{TRAIN_SUBDIR_NAME}")
        else:
            dataset_path = os.path.join(root, TOP_SUBDIR_NAME, f"2.{VALID_SUBDIR_NAME}")

        self.dataset_path = dataset_path
        self.audio_dataset_path = os.path.join(self.dataset_path, SOURCE_DIR_NAME)
        
        _unpack_dysarthricSpeech(self.dataset_path)

        audio_files_path = Path(self.audio_dataset_path).glob("*/*"+self._ext_audio)

        self._walker = [audio_filepath.relative_to(self.audio_dataset_path).as_posix().rsplit(".")[0] for audio_filepath in audio_files_path]


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
        relative_filepath = self._walker[n]
        return _get_korDysarthricSpeech_metadata(relative_filepath, self.dataset_path, self._ext_audio, self._ext_json)


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
