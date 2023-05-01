#!/usr/bin/env python3
"""Trains a SentencePiece model on transcripts.

Example:
python train_spm.py --kor-scripts-path ./datasets
"""

import io
import pathlib
import numpy as np
import re
from argparse import ArgumentParser, RawTextHelpFormatter

import sentencepiece as spm

def get_transcript_text(transcript_path, sep):
    new_list = list()
    file_names = list()
    with open(transcript_path) as f:
        for line in f:
            modified_line = cleanup_transcript(line.split(sep, 1)[-1].strip())
            if modified_line != None:
                new_list.append(modified_line)
                file_names.append(line.split(sep, 1)[0].strip())
        return [new_list, file_names]


def get_transcripts(dataset_path, sep):
    training_metadata_base_path = pathlib.PosixPath(dataset_path)
    training_scripts_filePath = training_metadata_base_path.rglob("*_scripts.txt")
    merged_transcripts = []
    for path in training_scripts_filePath:
        merged_transcripts += get_transcript_text(path, sep)
    return merged_transcripts


def cleanup_transcript(line_of_text):
    if(len(line_of_text) < 5):
        return None
    
    cond = False
    partitions = re.split("(\([^\)]*\)/\([^\)]*\))", line_of_text)
    if(len(partitions) != 1):
        for part in partitions:
            curr = re.findall("\(([^\)]*)\)", part)
            for element in curr:
                if(('(' in element) or (')' in element)):
                    cond = True
                    break
    if cond == True:
        return None

    left = line_of_text.count('(')
    right = line_of_text.count(')')
    
    if(left != right or left%2 == 1):
        return None

    line_of_text = spelling_rep(line_of_text)
    
    if(bool(re.search('\d', line_of_text))):
        return None
    
    if('(' in line_of_text or ')' in line_of_text):
        return None

    line_of_text = remove_special_words(line_of_text)
    line_of_text = remove_dup_spaces(line_of_text)
    
    left = line_of_text.count('(')
    right = line_of_text.count(')')
    
    if(left != right or left%2 == 1):
        return None
    
    return line_of_text


def spelling_rep(line_of_transcript):
    result = ""
    
    segment_list = re.split("(\([^\)]*\)/\([^\)]*\))", line_of_transcript)
    if(len(segment_list) != 1):
        for segment in segment_list:
            curr = re.findall("\(([^\)]*)\)", segment)
            if(len(curr) == 0):
                result += segment
            else:
                if(bool(re.search('\d', curr[0]))):
                    result += curr[1]
                else:
                    result += curr[0]
        return result
    
    return line_of_transcript


def remove_special_words(x):
    special_symbol = re.compile("[\*\+/blon\.\?]")
    return re.sub(special_symbol, "", x)


def remove_dup_spaces(x):
    special_symbol = re.compile(r"\s+")
    return re.sub(special_symbol, " ", x).strip()


def train_spm(input_file, prefix):
    spm.SentencePieceTrainer.train(
        input = input_file,
        model_prefix = prefix,
        vocab_size=350,
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

    merged_transcripts = []
    file_names = []

    temp = get_transcripts(args.kor_scripts_path, " :: ")
    
    merged_transcripts += temp[0]
    file_names += temp[1]

    with open(r"./aggregated_scripts.txt", 'w') as fp:
        fp.write('\n'.join(merged_transcripts[:100]))
        
    for idx, transcript in enumerate(file_names[:100]):
        file_names[idx] = file_names[idx] + " :: " + merged_transcripts[idx]
        
    with open(r"./aggregated_filenames_scripts.txt", 'w') as fp2:
        fp2.write('\n'.join(file_names[:100]))

    train_spm("./aggregated_scripts.txt", args.model_prefix)

if __name__ == "__main__":
    run_cli()