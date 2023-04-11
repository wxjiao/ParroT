##############################
# Function: convert alpaca data format to training data format: {'text': "", 'prefix': ""}
# Author: Wenxiang Jiao
# Last modified: 2023/04/15
##############################

import argparse
import time
import json
from tqdm import tqdm
import random
import numpy as np
import csv, json



# Instrauct language
lang_instruction = {
    'de': {'de': "Deutsch", 'en': "Englisch", 'ja': "Japanisch", 'zh': "Chinesisch"},
    'en': {'de': "German", 'en': "English", 'ja': "Japanese", 'zh': "Chinese"},
    'ja': {'de': "ドイツ語", 'en': "英語", 'ja': "日本語", 'zh': "中国語"},
    'zh': {'de': "德语", 'en': "英语", 'ja': "日语", 'zh': "中文"},
}

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_prompt(path):
    list_data_dict = read_json(path)
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]
    targets = [f"{example['output']}{DEFAULT_EOS_TOKEN}" for example in list_data_dict]
    return targets, sources


def write_json(in_file, out_file):
    prompts, sources = create_prompt(in_file)
    with open(out_file, 'w', encoding='utf-8') as fo:
        for p,s in zip(prompts, sources):
            jsoned = json.dumps({'text': p, 'prefix': s}, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')


if __name__ == "__main__":
    """
    python3 ../create_prompt.py --in-file ./alpaca_data.json --out-file data_alp_hf.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file','-i', type=str, required=True, help='input file')
    parser.add_argument('--out-file','-o', type=str, required=True, help='output file')
    args = parser.parse_args()
    in_file = args.in_file
    out_file = args.out_file

    # Start
    write_json(in_file, out_file)
