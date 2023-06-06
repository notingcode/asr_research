#!/usr/bin/env python3
"""Trains a SentencePiece model on transcripts.

Example:
python train_spm.py --kor-datasets-path ./datasets
"""

from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from dataset_modules.solugate_converspeech import(
    _unpack_solugateSpeech,
    _get_all_scripts as get_solugate_scripts,
)
from dataset_modules.diquest_normalspeech import(
    _unpack_diquestSpeech,
    _get_script_from_json,
)
from dataset_modules.hallym_dysarthricspeech import(
    _unpack_dysarthricSpeech,
    TOP_SUBDIR_NAME,
    LABEL_DIR_NAME,
)
from dataset_modules.etri_converspeech import(
    _unpack_etriSpeech,
    _get_all_scripts as get_etri_scripts,
    _SCRIPTS_FILES_DIR,
    _TRAIN_SCRIPT_FILENAME,
)

from common import(
    DIQUEST_DIR_NAME,
    SOLUGATE_DIR_NAME,
    DYSARTHRIC_DIR_NAME,
    ETRI_DIR_NAME,
    TRAIN_SUBDIR_NAME,
)

import sentencepiece as spm

EXT_TYP_KEY = 'ext_typ'
SEPARATOR_TYP_KEY = 'sep_typ'

JSON_EXT = '.json'
SCRIPTS_EXT = '_scripts.txt'
TRN_EXT = '.trn'

DATASET_OPTIONS = {
    # DIQUEST_DIR_NAME : {EXT_TYP_KEY : JSON_EXT, SEPARATOR_TYP_KEY : ''},
    # SOLUGATE_DIR_NAME : {EXT_TYP_KEY : SCRIPTS_EXT, SEPARATOR_TYP_KEY : '::'},
    ETRI_DIR_NAME : {EXT_TYP_KEY : TRN_EXT, SEPARATOR_TYP_KEY : '::'},
}


def get_transcripts(datasets_path: Path, dataset_name: str, ext: str, separator: str=''):
    
    merged_transcripts = []
    
    dataset_path = datasets_path.joinpath(dataset_name)
    
    if not dataset_path.is_dir():
        return merged_transcripts
    
    if dataset_name is DYSARTHRIC_DIR_NAME:
        dataset_path = dataset_path.joinpath(TOP_SUBDIR_NAME, f'1.{TRAIN_SUBDIR_NAME}', LABEL_DIR_NAME)
        search_str = f"*/*{ext}"
    elif dataset_name is SOLUGATE_DIR_NAME:
        dataset_path = dataset_path.joinpath(TRAIN_SUBDIR_NAME)
        _unpack_etriSpeech(dataset_path, 'all')
        search_str = f"*/*{ext}"
    elif dataset_name is DIQUEST_DIR_NAME:
        dataset_path = dataset_path.joinpath(TRAIN_SUBDIR_NAME)
        _unpack_diquestSpeech(dataset_path)
        search_str = f"*/*{ext}"
    elif dataset_name is ETRI_DIR_NAME:
        dataset_path = dataset_path.joinpath(_SCRIPTS_FILES_DIR)
        _unpack_etriSpeech(dataset_path)
        search_str = f"{_TRAIN_SCRIPT_FILENAME}{ext}"
    
    file_paths = dataset_path.glob(search_str)

    if ext is JSON_EXT:
        for file_path in file_paths:
            merged_transcripts.append(_get_script_from_json(file_path))
    elif ext is SCRIPTS_EXT:
        for file_path in file_paths:
            merged_transcripts += get_solugate_scripts(file_path, separator)
    elif ext is TRN_EXT:
        for file_path in file_paths:
            merged_transcripts += get_etri_scripts(file_path, separator)

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

    merged_transcripts = []
    
    for dataset_name, options in DATASET_OPTIONS.items():
        merged_transcripts += get_transcripts(args.kor_datasets_path, dataset_name, options[EXT_TYP_KEY], options[SEPARATOR_TYP_KEY])        

    with open(r"./aggregated_scripts.txt", 'w') as fp:
        fp.write('\n'.join(merged_transcripts))

    train_spm(merged_transcripts, args.model_prefix)

if __name__ == "__main__":
    run_cli()