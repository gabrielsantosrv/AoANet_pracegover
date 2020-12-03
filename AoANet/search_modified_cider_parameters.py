from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

import misc.utils as utils
import sys

from dataloader import DataLoader
from nltk.tokenize import word_tokenize
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
import numpy as np
import random

def decode_gts(data_gts, ix_to_word):
    sentences = []
    for array in data_gts:
        text = ''
        # there is only one reference
        ref = array[0]
        for idx in ref:
            if idx > 0:
                text += ' ' + ix_to_word[str(idx)]
            else:
                break
        sentences.append(text.strip())
    return sentences


parser = argparse.ArgumentParser()
parser.add_argument("--input_json", default="/work/recod/gabriel.santos/AoANet/data/pracegovertalk.json")
parser.add_argument("--input_label_h5", default="/work/recod/gabriel.santos/AoANet/data/pracegovertalk_label.h5")
parser.add_argument("--input_fc_dir", default="/work/recod/gabriel.santos/AoANet/data/pracegovertalk_fc")
parser.add_argument("--input_att_dir", default="/work/recod/gabriel.santos/AoANet/data/pracegovertalk_att")
parser.add_argument("--input_box_dir", default="/work/recod/gabriel.santos/AoANet/data/pracegovertalk_box")
parser.add_argument("--seq_per_img", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--cached_tokens", default="/work/recod/gabriel.santos/AoANet/data/pracegover-train-idxs_len_100")
parser.add_argument('--val_images_use', type=int, default=-1)

opt = parser.parse_args()
loader = DataLoader(opt)
split = 'val'
#loader.reset_iterator(split)
#n = 0

#sample_files = ['/work/recod/gabriel.santos/AoANet/experiments/baseline/val_len_100_cider.json',
#                '/work/recod/gabriel.santos/AoANet/experiments/cider_pen_repetition/val_len_100_cider_pen_repetition.json']
sample_files = ['/work/recod/gabriel.santos/AoANet/experiments/cider_pen_02_len_08_rep/val_len_100_cider_pen_02_len_08_rep.json']
for i, filename in enumerate(sample_files):
    with open(filename) as json_file:
        sampled_labels = json.load(json_file)

    sampled_labels = {sample["image_id"]: sample["caption"] for sample in sampled_labels}
    num_images = opt.val_images_use
    results = {}
    coefficient_rep_list = [random.uniform(0, 1) for _ in range(50)]

    for coefficient_rep in coefficient_rep_list:
        coefficient_len = 1 - coefficient_rep
        CiderD_scorer = CiderD(df=opt.cached_tokens, alpha=1.0, penalize_repetition=True,
                               coefficient_rep=coefficient_rep, coefficient_len=coefficient_len,
                               verbose=True)

        loader.reset_iterator(split)
        n = 0
        scores_list = []
        batch_index = 0
        print("coefficient_len", coefficient_len)
        print("coefficient_rep", coefficient_rep)
        while True:
            data = loader.get_batch(split)
            n = n + loader.batch_size
            
            sents = decode_gts(data['gts'], loader.get_vocab())           
            generated = []
            ground_truth = {}

            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': [sampled_labels[data['infos'][k]['id']]]}
                generated.append(entry)
                ground_truth[data['infos'][k]['id']] = [sent]             
            mean, cider_scores = CiderD_scorer.compute_score(ground_truth, generated)
            print("Batch", batch_index, "Cider", mean)

            scores_list.append(cider_scores)
            batch_index += 1
            if data['bounds']['wrapped']:
                break
            if num_images >= 0 and n >= num_images:
                break

        results["{},{}".format(coefficient_rep, coefficient_len)] = np.append([], scores_list).tolist()

    with open('result_prod_cider_parameters_model_08_rep_02_len.txt', 'w') as f:
        f.write(json.dumps(results))

