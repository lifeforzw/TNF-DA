import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from baukit import Trace, TraceDict
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import random
import numpy as np
from collections import defaultdict
import json

def write2json(output, file, op='w', indent=2):
    with open(file, op, encoding='UTF-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=indent)

def batch(iterable, bsize=1):
    total_len = len(iterable)
    for ndx in range(0, total_len, bsize):
        yield list(iterable[ndx:min(ndx+bsize, total_len)])

def dict2device(d, device):
    for k, v in d.items():
        d[k] = v.to(device)
    return d

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

def build_position_ids(enc_inputs:dict):
    position_ids = enc_inputs['attention_mask'].long().cumsum(-1) - 1
    position_ids[position_ids == -1] = 1
    enc_inputs['position_ids'] = position_ids
    return enc_inputs

def dict_cat(dicts):
    ip, atm, pti = [], [], []
    for d in dicts:
        ip.append(d['input_ids'])
        atm.append(d['attention_mask'])
        pti.append(d['position_ids'])
    ip = torch.stack(ip, dim=0)
    atm = torch.stack(atm, dim=0)
    pti = torch.stack(pti, dim=0)
    return dict(
        input_ids=ip,
        attention_mask=atm,
        position_ids=pti
    )

def get_context_template(model, tok, samples=5):
    context_templates = []
    templates = ["The", "Therefore", "Because", "I", "You", "How", "What", "A", "It", "But"]
    enc_templates = tok(templates[:samples], padding=True, return_tensors='pt').to(model.device)
    position_ids = enc_templates['attention_mask'].long().cumsum(-1) - 1
    position_ids[position_ids == -1] = 1
    enc_templates['position_ids'] = position_ids
    output = model.generate(**enc_templates, pad_token_id=tok.eos_token_id, max_length=10)
    output = tok.batch_decode(output, skip_special_tokens=True)
    output = [ tt.replace("{", " ").replace("}", " ") + ". {}"
        for tt in output
    ]
    return output

def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    #print(toks)
    whole_string = "".join(toks)
    #print(whole_string)
    #print(substring)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def locate_neurons(delta_p, k, mt, start_id=0, reverse=False, random=False, isabs=False, interblock='neuron', raw_neurons=False):
    layernum, h = mt.num_layers, mt.inner_size
    if interblock == 'hidsize':
        h = mt.hidden_size
    if isinstance(k, int):
        cosk = k
        sumk = cosk + start_id
    else:
        cosk = int(k*layernum*h)
        sumk = cosk + start_id
    if isabs:
        delta_p = np.abs(delta_p)
    if not random :
        if not isinstance(delta_p, torch.Tensor):
            delta_p = torch.tensor(delta_p)
        if not reverse :
            neurons = torch.topk(delta_p.view(-1), sumk, largest=True)[1][start_id:]
        else:
            neurons = torch.topk(delta_p.view(-1), sumk, largest=False)[1][start_id:]
    else:
        neurons = torch.randperm(layernum*h)[:sumk]
    if raw_neurons:
        return neurons
    neurons = sorted(neurons)
    top_neuron = defaultdict(list)
    for n in neurons:
        top_neuron[mt.getlayername(n//h, 'mlp')].append((n%h).item())
    
    return top_neuron

def build_position_ids(input_ids:dict):
    if 'position_ids' in input_ids.keys():
        return 
    position_ids = input_ids['attention_mask'].long().cumsum(-1) - 1
    position_ids[position_ids == -1] = 1
    input_ids['position_ids'] = position_ids.to(input_ids['attention_mask'].device) 
    return input_ids


def fig_heatmap(data, output_dir, xlabel='neuron', ylabel='data', annot=False, vmin=None, vmax=None, cmap=None):
    print('building heatmap...')
    ax = sns.heatmap(data, annot=annot, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    figure = ax.get_figure()
    figure.savefig(output_dir)
    figure.clf()

def fig_linechart(data, indices, output_dir, xlabel='layer', ylabel='proportion', startid=0, x_ticks=None, vline=None):
    print('building linechart...')
    #tars = ['b*--', 'rs--', 'go--', 'y+--', 'k^--']
    #tars = []
    colors = ['r', 'g', 'b', 'c', 'y']
    if isinstance(data[0], list):
        for idx, ydata in enumerate(data):
            plt.plot(indices, ydata, color=colors[idx], alpha=0.5, linewidth=1, label=f'data-{startid+idx}')
    
    else:
        plt.plot(indices, data, color=colors[0], alpha=0.5, linewidth=1)
    
    if x_ticks:
        plt.xticks(x_ticks)
    if vline:
        plt.axvline(vline)

    plt.legend()  #显示上面的label

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_dir)
    plt.clf()