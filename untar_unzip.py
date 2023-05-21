import logging
import os
import tarfile
import zipfile
from typing import List, Optional
from pathlib import Path
from common import (
    SAMPLE_RATE,
    DYS_SAMPLE_RATE
)

import torchaudio

_resampler = torchaudio.transforms.Resample(DYS_SAMPLE_RATE, SAMPLE_RATE, lowpass_filter_width=12)

def _extract_tar(from_path: str, to_path: Optional[str] = None, overwrite: bool = False, n_directories_stripped: int = 9,) -> List[str]:
    if to_path is None:
        to_path = os.path.dirname(from_path)
    
    mode = "r"
    if from_path.endswith("tar.gz"):
        mode = "r:gz"
    
    with tarfile.open(from_path, mode) as tar:
        logging.info("Opened tar file {}.", from_path)
        files = []
        for member in tar.getmembers():
            member.path = member.path.split('/', n_directories_stripped)[-1]
            file_path = os.path.join(to_path, member.path)
            if member.isfile():
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
            tar.extract(member, to_path)
            
        return files


def _extract_zip(from_path: str, to_path: Optional[str] = None, overwrite: bool = False, n_directories_stripped: int = 0,) -> List[str]:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as zfile:
        logging.info("Opened zip file {}.", from_path)
        files = zfile.namelist()
        for file_info in zfile.infolist():
            file_info.filename = file_info.filename.split('/', n_directories_stripped)[-1]
            file_path = os.path.join(to_path, file_info.filename)
            if os.path.exists(file_path):
                logging.info("{} already extracted.".format(file_path))
                if not overwrite:
                    continue
            zfile.extract(file_info, to_path)
            
    return files


def _load_waveform(
    file_path: str,
    exp_sample_rate: int,
):
    waveform, sample_rate = torchaudio.load(file_path)
    
    if exp_sample_rate != sample_rate:
        if sample_rate == DYS_SAMPLE_RATE:
            waveform = _resampler(waveform)
        else:
            raise ValueError(f"sample rate should be {exp_sample_rate}, but got {sample_rate}")
    return waveform
