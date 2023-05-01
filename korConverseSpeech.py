import os
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from untar_unzip import _extract_tar, _load_waveform
from train_spm_base import cleanup_transcript

N_DIRECTORIES_STRIPPED = 6
SAMPLE_RATE = 16000

_DATA_SUBSETS = [
    "broadcast",
    "hobby",
    "dialog",
    "life",
    "life",
    "weather",
    "economy",
    "play",
    "shopping",
]


def _unpack_korConverseSpeech(source_path, subset_type):
    ext_archive = ".tar"
    
    tar_files = Path(source_path).glob(f"*/*_{subset_type}_*{ext_archive}")
    
    for file in tar_files:
        file.with_suffix("").mkdir(exist_ok=True)
        _extract_tar(from_path=file.as_posix(), to_path=file.with_suffix("").as_posix(), n_directories_stripped=N_DIRECTORIES_STRIPPED)


def _get_korConverseSpeech_metadata(
    fileid: str, root: str, folder: str, ext_audio: str, ext_txt: str
) -> Tuple[str, int, str, int, int, int]:

    # Get audio path and sample rate
    fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
    filepath = os.path.join(folder, speaker_id, chapter_id, f"{fileid_audio}{ext_audio}")

    # Load text
    file_text = f"{speaker_id}-{chapter_id}{ext_txt}"
    file_text = os.path.join(root, folder, speaker_id, chapter_id, file_text)
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {fileid_audio}")

    return (
        filepath,
        SAMPLE_RATE,
        transcript,
    )


class KORCONVERSESPEECH(Dataset):
    """*LibriSpeech* :cite:`7178964` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(
        self,
        root: Union[str, Path],
        subset_type: str
    ) -> None:
        root = os.fspath(root)

        _unpack_korConverseSpeech(root, subset_type)

        self._walker = sorted(str(p.stem) for p in Path(root).rglob(f"{subset_type}_*" + self._ext_audio))

    def get_metadata(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
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
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        fileid = self._walker[n]
        return _get_korConverseSpeech_metadata(fileid, self._ext_audio, self._ext_txt)

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
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self._walker)
