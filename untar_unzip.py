import logging
import os
import tarfile
import zipfile
from typing import Any, List, Optional
from pathlib import Path

import torchaudio


def _extract_tar(from_path: str, to_path: Optional[str] = None, overwrite: bool = False, n_directories_stripped: int = 1,) -> List[str]:
    if to_path is None:
        to_path = os.path.dirname(from_path)
    with tarfile.open(from_path, "r") as tar:
        logging.info("Opened tar file {}.", from_path)
        files = []
        for member in tar.getmembers():
            member.path = member.path.split('/', n_directories_stripped)[-1]
            file_path = os.path.join(to_path, member.path)
            if os.path.exists(file_path):
                logging.info("{} already extracted.".format(file_path))
                if overwrite:
                    continue
            files.append(member)
            
        tar.extractall(path=to_path, members=files)
        return files


def _extract_zip(from_path: str, to_path: Optional[str] = None, overwrite: bool = False, n_directories_stripped: int = 0,) -> List[str]:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as zfile:
        logging.info("Opened zip file {}.", from_path)
        files = []
        for file_ in zfile.namelist():
            file_path = os.path.join(to_path, file_)
            if os.path.exists(file_path):
                logging.info("{} already extracted.".format(file_path))
                if not overwrite:
                    continue
            files.append(file_)
            
        zfile.extractall(to_path, members=files)
    return files


def _load_waveform(
    file_path: str,
):
    waveform, sample_rate = torchaudio.load(file_path)
    return (waveform, sample_rate)
