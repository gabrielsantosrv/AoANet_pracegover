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
import matplotlib.pyplot as plt
from PIL import Image

def plot_attention(image, seq, attention_plot, output_file):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_seq = len(seq)
    for l in range(len_seq):
        temp_att = np.resize(attention_plot[l], (32, 32))
        ax = fig.add_subplot(len_seq//2, len_seq//2, l+1)
        ax.set_title(seq[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    output_file = output_file.split('/')[1]
    plt.savefig("att_"+output_file)

def get_attention_for_sentence(sent_index, attention_maps):
    attention = []
    for array in attention_maps:
        attention.append(array[sent_index, :])

    return attention


def eval_split(model, loader, eval_kwargs={}):
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
            seq, _, attention_maps = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')
            seq = seq.data
            print("seq", seq.size())
            print("att", len(attention_maps))
            print("att", len(attention_maps[0].shape))

        sents = utils.decode_sequence(loader.get_vocab(), seq)
        print("sent size", len(sents[0]))
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            image_path = os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path'])
            attention = get_attention_for_sentence(k, attention_maps)
            plot_attention(image_path, sent.split(), attention, data['infos'][k]['file_path'])

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

import opts
import models
from dataloader import *
from dataloaderraw import *
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--output_file', type=str, default='',
                help='path to output')
opts.add_eval_options(parser)

opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
opt.dataset = opt.input_json
eval_split(model, loader, vars(opt))
