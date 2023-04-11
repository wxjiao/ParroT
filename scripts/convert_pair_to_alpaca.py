##############################
# Function: convert bilingual sentence pairs to alpaca data format
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



def read_instruct(path, src, tgt, lang_ins):
    source, target = lang_instruction[lang_ins][src], lang_instruction[lang_ins][tgt]
    ins_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip().replace("[SRC]", source).replace("[TGT]", target)
            ins_list.append(line)
    return ins_list


def create_prompt(slines, tlines, ins_list, max_cxt):
    prompts = []
    for i in range(len(slines)):
        p = dict()
        num_demo = random.randint(0, max_cxt)
        st = max(0, i - num_demo)
        idxs = list(range(st, i + 1))
        demo_s, demo_t = [], []
        for j in idxs:
            demo_s.append(slines[j].strip())
            demo_t.append(tlines[j].strip())
        # randomly decide if '\n' is added
        draw = random.randint(0, 1000)
        if draw % 2 == 0:
            demo_s_merge = "\n".join(demo_s)
            demo_t_merge = "\n".join(demo_t)
        else:
            demo_s_merge = " ".join(demo_s)
            demo_t_merge = " ".join(demo_t)
        ins_idx = random.randint(0, len(ins_list) - 1)
        #print("Draw: {}".format(draw))
        instruct = ins_list[ins_idx]
        p["instruction"] = instruct
        p["input"] = demo_s_merge
        p["output"] = demo_t_merge
        prompts.append(p)
    return prompts


def write_json(src_file, tgt_file, out_file, ins_list, max_cxt, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    with open(src_file, 'r', encoding='utf-8') as fs, open(tgt_file, 'r', encoding='utf-8') as ft, \
            open(out_file, 'w', encoding='utf-8') as fo:
        data = dict()
        slines, tlines = fs.readlines(), ft.readlines()
        prompts = create_prompt(slines, tlines, ins_list, max_cxt)
        json.dump(prompts, fo, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    """
    python3 ../create_s2t_alpaca.py -sf train.en-de.en -tf train.en-de.de -s en -t de -if ../instruct_follow.txt -of data_pair_alp.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-s', type=str, required=True, help='src language, en, de, ja, zh')
    parser.add_argument('--tgt', '-t', type=str, required=True, help='tgt language, en, de, ja, zh')
    parser.add_argument('--lang-ins', '-li', type=str, default='en', help='instruct language, en, de, ja, zh')
    parser.add_argument('--ins-file','-if', type=str, required=True, help='ins file')
    parser.add_argument('--src-file','-sf', type=str, required=True, help='src file')
    parser.add_argument('--tgt-file','-tf', type=str, required=True, help='tgt file')
    parser.add_argument('--out-file','-of', type=str, required=True, help='out file')
    parser.add_argument('--max-cxt', type=int, default=3, help='max context')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    src, tgt = args.src, args.tgt
    lang_ins = args.lang_ins
    ins_file = args.ins_file
    src_file = args.src_file
    tgt_file = args.tgt_file
    out_file = args.out_file
    max_cxt = args.max_cxt
    seed = args.seed

    # Start
    ins_list = read_instruct(ins_file, src, tgt, lang_ins)
    print("Number of instructs: {}".format(len(ins_list)))
    write_json(src_file, tgt_file, out_file, ins_list, max_cxt, seed)
