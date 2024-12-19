### do causal trace for model and data


from mt import ModelandTokenizer
import torch

import os 
import sys
sys.path.append('../')

from dset import KnownsDataset, EvalDataset
from dset import tok_dataset
from util.runningstats import Covariance, tally
from util.utils import dict2device, seed_torch
import argparse
import logging
from datasets import load_dataset
from tqdm import tqdm
import baukit 
import numpy as np


datapth = ''
modelpth = ''

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parserargs():
    parser = argparse.ArgumentParser(description='Causal Tracing for neuron')

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--model_name",
        default="gpt2-medium", 
        choices=[
            "gpt2-xl",
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neox-20b",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
        ],
    )
    aa('--output_dir', default='./result/{model_name}/causaltrace/')
    aa('--noise_level', default=3, type=float)
    aa('--mode', type=str, default='known_id', choices=['known_id', 'gen_id', 'loc_id'])
    aa('--inter_block', type=str, default='neuron', choices=['neuron', 'hidsize'])
    aa('--erange', type=str, default='whole', choices=['whole', 'subject', 'prompt'])
    aa('--de_rate', type=float, default=0.2)
    aa('--sd', type=int, default=0)
    aa('--ed', type=int, default=3000)
    # aa('--mode', default='ct', choices=['ct', 'ctn'])
    # aa('--k', type=float, default=0.04)

    return parser.parse_args()

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    seed_torch()
    args = parserargs()
    interrange = args.erange
    interblock = args.inter_block
    nlevel = args.noise_level
    de_rate = args.de_rate
    kind = args.mode
    model_name = args.model_name
    sd = args.sd
    ed = args.ed
    
    

    logger.info('loading model&tokenizer')
    mt = ModelandTokenizer(model_name)
    if 'gpt-j' in model_name:
        model_name = 'gpt-j'
    #breakpoint()
    logger.info('building noise')
    noise_pth = f'''./result/checkpoints/noise_{model_name}.pth'''
    if os.path.exists(noise_pth):
        noise_level = torch.load(noise_pth)
    else:
        noise_level = mt.collect_embedding_gaussian(nlevel=nlevel)
        torch.save(noise_level, noise_pth)
    logger.info('loading data...')



    output_dir = args.output_dir.format(model_name=model_name)
    result_dir = f'{output_dir}{interblock}/corfirst{int(de_rate*100)}_{interrange}_results_s{int(nlevel)}'
    os.makedirs(result_dir, exist_ok=True)
    print(result_dir)
    if kind == 'known_id':
        Knowns = KnownsDataset(data_dir=datapth, model_name=model_name)
    else:
        Knowns = EvalDataset(data_dir=datapth, model_name=model_name, kind=kind)
    print(result_dir)
    mx = []
    mm = []
    mp = []
    for d_ii, knowledge in enumerate(tqdm(Knowns, desc='datas')):
        case_id = knowledge['case_id']
        if d_ii < sd or d_ii >= ed:
            continue
        if kind == 'known_id':
            tmp = ''
        elif kind == 'gen_id':
            tmp = 'G'
        elif kind == 'loc_id':
            tmp = 'L'
        filename = f'''{result_dir}/case_{case_id}{tmp}{knowledge['index']}.npz'''
        if not os.path.exists(filename):
            inp = mt.encode_input(knowledge['prompt'])
            #print(knowledge['prompt'])
            inp = dict2device(inp, mt.device)
            target_id = ' ' + knowledge['target']
            target_id = mt.encode_input(target_id)['input_ids'].squeeze()
            try:
                cor_probs, cl_probs, correct_prediction, delta_activation = mt.clcoRun(inp, knowledge['subject'], target_id, intervention=noise_level, interrange=interrange)
                
                top_k_neurons = torch.topk(delta_activation.view(-1), int(mt.num_layers*mt.inner_size*de_rate), largest=True)
                ids = top_k_neurons.indices
                
                probs = mt.neurons_intervention(inp, knowledge['subject'], target_id, bsize=800, intervention=noise_level, interblock=interblock, interrange=interrange, neuron_range=ids,  times=1)
            except ValueError:
                print('jump one')
                continue
            #breakpoint()
            result = dict(
                correct_prediction = correct_prediction,
                target = knowledge['target'],
                clean_target_p = cl_probs,
                cor_target_p = cor_probs,
                delta_target_p = probs - cor_probs,
                max_delta_act=torch.max(delta_activation).item(),
                min_delta_act=torch.min(delta_activation).item()
            )
            numpy_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in result.items()
            }
            np.savez(filename, **numpy_result)
        else:
            numpy_result = np.load(filename, allow_pickle=True)
        mx.append(numpy_result['max_delta_act'])
        #print(numpy_result['max_delta_act'], numpy_result['min_delta_act'])
        mm.append(numpy_result['min_delta_act'])
        mp.append(numpy_result['cor_target_p']/ numpy_result['clean_target_p'])
    print(sum(mx)/len(mx))
    print(sum(mm)/len(mm))  
    print(1-sum(mp)/len(mp))     







# def collect_embedding_gaussian(mt):
#     m,c = get_embedding_cov(mt)
#     return make_generator_transform(m, c)


# def make_generator_transform(mean=None, cov=None):
#     d = len(mean) if mean is not None else len(cov)
#     device = mean.device if mean is not None else cov.device
#     layer = torch.nn.Linear(d, d, dtype=torch.double)
#     baukit.set_requires_grad(False, layer)
#     layer.to(device)
#     layer.bias[...] = 0 if mean is None else mean
#     if cov is None:
#         layer.weight[...] = torch.eye(d).to(device)
#     else:
#         _, s, v = cov.svd()
#         w = s.sqrt()[None, :] * v
#         layer.weight[...] = w
#     return layer
    
# def get_embedding_cov(mt):
#     model = mt.model
#     tokenizer = mt.tokenizer

#     def get_ds():
#         ds_name = "wikitext"
#         raw_ds = load_dataset(
#             os.path.join(datapth, ds_name),
#             dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
#         )
#         try:
#             maxlen = model.config.n_positions
#         except:
#             maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
#         return tok_dataset.TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)
#     logger.info('loading wikitext...')
#     ds = get_ds()
#     sample_size = 1000
#     batch_size = 5
#     filename = None
#     batch_tokens = 100

#     progress = lambda x, **k: x

#     stat = Covariance()
#     loader = tally(
#         stat,
#         ds,
#         cache=filename,
#         sample_size=sample_size,
#         batch_size=batch_size,
#         collate_fn=tok_dataset.length_collation(batch_tokens),
#         pin_memory=True,
#         random_sample=1,
#         num_workers=0,
#     )
#     logger.info('computing mean & cov...')
#     with torch.no_grad():
#         for batch_group in loader:
#             for batch in batch_group:
#                 batch = tok_dataset.dict_to_(batch, "cuda")
#                 del batch["position_ids"]
#                 with baukit.Trace(model, mt.getlayername(0, "embed")) as tr:
#                     model(**batch)
#                 feats = tok_dataset.flatten_masked_batch(tr.output, batch["attention_mask"])
#                 stat.add(feats.cpu().double())
#     return stat.mean(), stat.covariance()