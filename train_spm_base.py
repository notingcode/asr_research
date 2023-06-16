#!/usr/bin/env python3
"""Trains a SentencePiece model on transcripts.

Example:
python train_spm.py --kor-datasets-path ./datasets
"""

from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from dataset_modules.etri_converspeech import(
    _unpack_speechData,
    _get_all_scripts,
    _SCRIPTS_FILES_DIR,
    _TRAIN_SCRIPT_FILENAME,
)
from common import ETRI_DIR_NAME
import sentencepiece as spm

TRN_EXT = '.trn'


def get_transcripts(datasets_path: Path):
    
    merged_transcripts = []
    
    dataset_path = datasets_path.joinpath(ETRI_DIR_NAME, _SCRIPTS_FILES_DIR)
    
    if not dataset_path.is_dir():
        return merged_transcripts
    
    _unpack_speechData(dataset_path)
    
    search_str = f"{_TRAIN_SCRIPT_FILENAME}{TRN_EXT}"
    
    file_path = dataset_path.joinpath(search_str)

    merged_transcripts = _get_all_scripts(file_path, "::")

    return merged_transcripts


def train_spm(list_of_texts, prefix):
    spm.SentencePieceTrainer.Train(
        sentence_iterator = iter(list_of_texts),
        model_prefix = prefix,
        vocab_size=10000,
        model_type="unigram",
        input_sentence_size=-1,
        pad_id=0,
        pad_piece="<pad>",
        unk_id=1,
        unk_piece="<unk>",
        bos_id=2,
        bos_piece="<s>",
        eos_id=3,
        eos_piece="</s>",
        user_defined_symbols=["<sep>","<mask>","<cls>"]
    )


def parse_args():
    default_prefix = "baseline"
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--kor-datasets-path",
        required=True,
        type=Path,
        help="Path to script dataset.",
    )
    parser.add_argument(
        "--model-prefix",
        default=default_prefix,
        type=str,
        help=f"File to save model to. (Default: '{default_prefix}.model')",
    )
    return parser.parse_args()


def run_cli():
    args = parse_args()

    merged_transcripts = get_transcripts(args.kor_datasets_path)        

    with open(r"./aggregated_scripts.txt", 'w') as fp:
        fp.write('\n'.join(merged_transcripts))

    train_spm(merged_transcripts, args.model_prefix)

if __name__ == "__main__":
    run_cli()