#!/usr/bin/env python3
"""Trains a SentencePiece model on transcripts.

Example:
python train_spm.py --kor-scripts-path ./datasets
"""

from pathlib import Path
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from script_normalization import(
    cleanup_transcript,
    edit_annotation,
)
from solugate_converspeech import unpack_solugateSpeech
from diquest_normalspeech import unpack_diquestSpeech
from hallym_dysarthricspeech import(
    # unpack_dysarthricSpeech,
    TOP_SUBDIR_NAME,
    LABEL_DIR_NAME,
)

from common import(
    DIQUEST_DIR_NAME,
    SOLUGATE_DIR_NAME,
    DYSARTHRIC_DIR_NAME,
    TRAIN_SUBDIR_NAME,
)
EXT_TYP_KEY = 'ext_typ'
SEPARATOR_TYP_KEY = 'sep_typ'

JSON_EXT = '.json'
SCRIPTS_EXT = '_scripts.txt'

import sentencepiece as spm

DATASET_OPTIONS = {
    DIQUEST_DIR_NAME : {EXT_TYP_KEY : JSON_EXT, SEPARATOR_TYP_KEY : ''},
    SOLUGATE_DIR_NAME : {EXT_TYP_KEY : SCRIPTS_EXT, SEPARATOR_TYP_KEY : ' :: '},
    # DYSARTHRIC_DIR_NAME : {EXT_TYP_KEY : '.json', SEPARATOR_TYP_KEY : ''},
}

def get_scripts_from_txt(transcript_path, separator):
    new_list = list()
    with open(transcript_path) as f:
        for line in f:
            modified_line = cleanup_transcript(line.split(separator, 1)[-1].strip())
            if modified_line is not None:
                new_list.append(modified_line)
        return new_list
    
def get_script_from_json(transcript_path):
    with open(transcript_path) as f:
        data = json.load(f)
        modified_line = edit_annotation(data['발화정보']['stt'].strip('\\'))
        
    return modified_line

def get_transcripts(datasets_path: Path, dataset_name: str, ext: str, separator: str=''):
    
    merged_transcripts = []
    
    dataset_path = datasets_path.joinpath(dataset_name)
    
    if not dataset_path.is_dir():
        return merged_transcripts
    
    if dataset_name is DYSARTHRIC_DIR_NAME:
        dataset_path = dataset_path.joinpath(TOP_SUBDIR_NAME, f'1.{TRAIN_SUBDIR_NAME}', LABEL_DIR_NAME)
    elif dataset_name is SOLUGATE_DIR_NAME:
        dataset_path = dataset_path.joinpath(TRAIN_SUBDIR_NAME)
        unpack_solugateSpeech(dataset_path, 'all')
    elif dataset_name is DIQUEST_DIR_NAME:
        dataset_path = dataset_path.joinpath(TRAIN_SUBDIR_NAME)
        unpack_diquestSpeech(dataset_path)
    
    
    file_paths = dataset_path.glob(f"*/*{ext}")

    if ext is JSON_EXT:
        for file_path in file_paths:
            merged_transcripts.append(get_script_from_json(file_path))
    elif ext is SCRIPTS_EXT:
        for file_path in file_paths:
            merged_transcripts += get_scripts_from_txt(file_path, separator)

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

    train_spm(merged_transcripts, args.model_prefix)

if __name__ == "__main__":
    run_cli()