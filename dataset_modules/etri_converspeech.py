import os
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset
from untar_unzip import _extract_zip, _load_waveform
from script_normalization import etri_normalize
from common import SAMPLE_RATE

_NAME_HEADER = 'KsponSpeech'
_VAL_DATA_DIR = '평가용_데이터'
_TRAIN_DATA_DIR = '한국어_음성_분야'
_SCRIPTS_FILES_DIR = '전시문_통합_스크립트'

_TRAIN_SCRIPT_FILENAME = 'train'
_VAL_SCRIPT_FILENAME_CLEAN = 'eval_clean'
# _VAL_SCRIPT_FILENAME_OTHER = 'eval_other'

def _get_all_scripts(transcript_filepath, separator):
    new_list = []
    with open(transcript_filepath) as f:
        for line in f:
            modified_line = etri_normalize(line.split(separator, 1)[-1].strip())
            if modified_line is not None:
                new_list.append(modified_line)
        return new_list


def _unpack_etriSpeech(source_path: str | Path, n_directories_stripped: int = 0):
    ext_archive = '.zip'
        
    zip_files = Path(source_path).glob(f"*{ext_archive}")
    
    args = []
    
    for file in zip_files:
        args.append((file.as_posix(), source_path, False, n_directories_stripped))
    
    pool = mp.Pool(min(mp.cpu_count(), len(args)))
    
    pool.starmap(_extract_zip, args, chunksize=1)


def _get_korConverseSpeech_metadata(
    file_idx: str, dataset_path: str, ext_audio: str, ext_txt: str,
) -> Tuple[str, int, str]:
    
    index_subdir_getter = int(1000)

    index = int(file_idx)-1

    transcript_filename = f"{_NAME_HEADER}_{file_idx}{ext_txt}"
    audio_filename = f"{_NAME_HEADER}_{file_idx}{ext_audio}"
    
    indexed_dir = f"{_NAME_HEADER}_{(index//index_subdir_getter)+1:04d}"
    
    transcript_filepath = os.path.join(dataset_path, indexed_dir, transcript_filename)
    audio_filepath = os.path.join(dataset_path, indexed_dir, audio_filename)

    # Load text
    with open(transcript_filepath) as f:
        transcript = etri_normalize(f.readline().strip())
        if transcript is None:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {transcript_filepath}")

    return (
        audio_filepath,
        SAMPLE_RATE,
        transcript,
    )


class ETRISPEECH(Dataset):
    """
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset_type (str): Type of subset to be trained on.
    """

    _ext_txt = ".txt"
    _ext_audio = ".pcm"
    _ext_scripts = ".trn"

    def __init__(
        self,
        root: str | Path,
        training: bool,
    ) -> None:
        scripts_dataset_path = os.path.join(root, _SCRIPTS_FILES_DIR)
        
        self.root = os.fspath(root)
        self.scripts_dataset_path = scripts_dataset_path

        if training:
            audio_dataset_path = os.path.join(root, _TRAIN_DATA_DIR)
            scripts_filename = f"{_TRAIN_SCRIPT_FILENAME}{self._ext_scripts}"
        else:
            audio_dataset_path = os.path.join(root, _VAL_DATA_DIR)
            scripts_filename = f"{_VAL_SCRIPT_FILENAME_CLEAN}{self._ext_scripts}"
            
        self.audio_dataset_path = audio_dataset_path

        _unpack_etriSpeech(scripts_dataset_path)
        _unpack_etriSpeech(audio_dataset_path, n_directories_stripped=1)
        
        scripts_filepath = os.path.join(scripts_dataset_path, scripts_filename)

        with open(scripts_filepath) as f:
            self._walker = [Path(line.split('::')[0].strip()).stem.split('_')[-1] for line in f]
                                   

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
        file_index = self._walker[n]
        return _get_korConverseSpeech_metadata(file_index, self.dataset_path, self._ext_audio, self._ext_txt)

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
