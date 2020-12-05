from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

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
def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')

    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator(split)

    n = 0
    predictions = []
    iterations = 0
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq, _, att = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')
            seq = seq.data
            att = att.data
            print("seq", seq)
            print("att", att)

        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image {}: {}'.format(entry['image_id'], entry['caption']))

        iterations += 1

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... {}/{}'.format(ix0 - 1, ix1))
        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

