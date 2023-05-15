#!/usr/bin/env python3
"""Trains a SentencePiece model on transcripts.

Example:
python train_spm.py --kor-scripts-path ./datasets
"""

import io
import pathlib
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from script_normalization import(
    cleanup_transcript,
    edit_annotation,
)

import sentencepiece as spm

SCRIPTS_TXT_EXT = "*_scripts.txt"
JSON_EXT = "*.json"
TXT_EXT = "*.txt"

EXT_DICT = {"txt": TXT_EXT, "json": JSON_EXT, "scripts": SCRIPTS_TXT_EXT}

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
        modified_line = edit_annotation(data['발화정보']['stt'].strip("\\"))
        
    return modified_line

def get_script_from_txt(transcript_path, separator):
    with open(transcript_path) as f:
        modified_line = cleanup_transcript(f.readline().split(separator, 1)[-1].strip())
        
    return modified_line

def get_transcripts(dataset_path, typ: str="txt", separator: str=""):
    
    ext = EXT_DICT[typ]
    
    training_metadata_base_path = pathlib.PosixPath(dataset_path)
    training_scripts_filePath = training_metadata_base_path.rglob(ext)
    merged_transcripts = []
    for path in training_scripts_filePath:
        match typ:
            case "scripts": merged_transcripts += get_scripts_from_txt(path, separator)
            case "json": merged_transcripts.append(get_script_from_json(path))
            case "txt": merged_transcripts.append(get_script_from_txt(path, separator))

    return merged_transcripts

def train_spm(input_file, prefix):
    spm.SentencePieceTrainer.Train(
        input = input_file,
        model_prefix = prefix,
        vocab_size=6000,
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
        "--kor-scripts-path",
        required=True,
        type=pathlib.Path,
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

    merged_transcripts = get_transcripts(args.kor_scripts_path, " :: ")
        
    with open(args.kor_scripts_path.as_posix()+"/aggregated_scripts.txt", 'w') as fp:
        fp.write('\n'.join(merged_transcripts))

    train_spm(args.kor_scripts_path.as_posix()+"/aggregated_scripts.txt", args.model_prefix)

if __name__ == "__main__":
    run_cli()