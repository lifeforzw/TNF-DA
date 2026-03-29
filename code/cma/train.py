### train for edit ---- finetune

from trainer import Trainer
import logging
import argparse
from mt import ModelandTokenizer
from dset import TrainDataset, Cali_TrainDataset, AttributeSnippets, get_tfidf_vectorizer
import torch.nn.functional as F
import torch.optim
import os
import sys 
sys.path.append('../')
from util.utils import locate_neurons, get_context_template
import numpy as np
import json
from evaluate import compute_rewrite_quality_counterfact
from tqdm import tqdm



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
        default="gpt2-xl", 
        choices=[
            "gpt2-xl",
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neox-20b",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
        ],
    )
    aa('--output_dir', default='./result/checkpoints/pytorch_model.bin')
    aa('--input_dir', default='./result/{model_name}/causaltrace/{edit_type}/')
    aa('--ktop', type=float, default=0.01) # need 
    aa('--nn', type=int, default=100) # need
    aa('--edit_batch', type=int, default=1)
    aa('--isabs', type=bool, default=False)
    aa('--edit_type', type=str, default='neuron', choices=['neuron', 'hidsize'])
    aa('--theta', type=float, default=0.0) # need
    aa('--erange', type=str, default='whole', choices=['whole', 'subject','prompt'])
    aa('--de_rate', type=float, default=0.2) # need
    aa('--noise_level', default=3, type=int) # need
    aa('--template', type=str, default='')
    aa('--RN', type=str, default='')
    aa('--editpos', type=str, default='all', choices=['c_proj', 'c_fc', 'all'])# c_proj ==> ''
    aa('--calidata', action='store_true', help='if run with cali data and klloss')
    # aa('--mode', default='ct', choices=['ct', 'ctn'])
    # aa('--k', type=float, default=0.04)

    return parser.parse_args()


# TODO: 尝试前后一起选？
# TODO: 尝试loss加上遗忘之前的？
def get_neurons_indices(mt, dataset, k, inputpth, start_id, isabs=False, edit_type='neuron', isrn=False):
    neuron_ids = []
    for idx in tqdm(range(dataset.N), desc='locate top neurons'):
        case_id = dataset.case_id[idx]
        pth = os.path.join(inputpth, f'''case_{case_id}.npz''')
        if not os.path.exists(pth):
            neuron_ids.append({})
            continue
        numpy_result = np.load(pth, allow_pickle=True)
        delta_p = numpy_result['delta_target_p']
        neuron_ids.append(locate_neurons(delta_p, k, mt, start_id=start_id, isabs=isabs, interblock=edit_type, random=isrn))
    
    return neuron_ids


start = 0


if __name__ == '__main__':
    args = parserargs()
    #os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    model_name = args.model_name 
    edit_type = args.edit_type
    input_dir = args.input_dir.format(model_name=model_name, edit_type=edit_type)
    output_dir = args.output_dir #.format(model_name=model_name)
    data_num = args.nn
    edit_batch = args.edit_batch
    theta = args.theta
    interrange = args.erange
    nlevel = args.noise_level
    de_rate = args.de_rate
    input_dir = os.path.join(input_dir, f'''corfirst{int(de_rate*100)}_{interrange}_results_s{nlevel}/''')
    istemplate = (args.template != '')
    iscalidata = 1 if args.calidata else 0

    isrn = False if args.RN == '' else True

    logger.info('loading model && data')
    mt = ModelandTokenizer(model_name)
    #breakpoint()
    if iscalidata:
        bs = 50
        Knowns = Cali_TrainDataset(datapth, model_name, mt.tokenizer, data_num, bs)
        
    elif istemplate:
        templates = get_context_template(mt.model, mt.tokenizer, samples=10)
        Knowns = TrainDataset(datapth, model_name, mt.tokenizer, data_num, templates=templates)
    else:
        Knowns = TrainDataset(datapth, model_name, mt.tokenizer, data_num)
    #breakpoint()
    criterion = F.cross_entropy
    opt = torch.optim.Adam
    k = args.ktop 
    if k >= 1:
        k = int(k)

    print(str(k))
    str_k = str(k).replace('.','')
    str_theta = str(theta).replace('.', '')
    sfx = 'T' if args.isabs else 'F'
    print(f'neuron choose path is {input_dir}')
    eval_dir = f'''./result/{model_name}/train_results/corfirst{int(de_rate*100)}_{interrange}_results_s{nlevel}{args.template}{iscalidata}/{edit_type}_top{str_k}{sfx}_{str_theta}{args.RN}{args.editpos[-1]}/'''
    os.makedirs(eval_dir, exist_ok=True)
    print(f'''output is {eval_dir}''')

    skip_generation_tests = False
    snips = AttributeSnippets(datapth) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(datapth) if not skip_generation_tests else None
    gen_test_vars = [snips, vec]

    neuron_ids = get_neurons_indices(mt, Knowns, k, input_dir, start_id=start, isabs=args.isabs, edit_type=edit_type, isrn=isrn)
    
    trainer = Trainer(mt.model, mt.tokenizer, mt.hidden_size, mt.inner_size, Knowns, neuron_ids, mt.device, 5, criterion, opt, theta=theta, alpha=0, \
                       edit_type=edit_type, lr=0.0001, isabs=args.isabs, edit_pos=args.editpos)
    logger.info('training...')


    trainer.train(output_dir, eval_dir, gen_test_vars)

    ## evaluate
    # with open(datapth+'counterfact.json', 'r') as f:
    #     source_data = json.load(f)
    
    # for ii, dt in enumerate(Knowns.data):
    #     if ii >= data_num:
    #         break
    #     gen_id = Knowns.dataidx['gen_id'][ii]
    #     loc_id = Knowns.dataidx['loc_id'][ii]
    #     dt['generation_prompts'] = [dt['generation_prompts'][i] for i in gen_id]
    #     dt['neighborhood_prompts'] = [dt['neighborhood_prompts'][i] for i in loc_id]
    #     metrics = {
    #         'case_id': dt['case_id'],
    #         'prompt': dt['prompt'],
    #         'post': compute_rewrite_quality_counterfact(
    #             trainer.edit_model.model,
    #             mt.tokenizer,
    #             dt
    #         )
    #     }
    #     with open(f'''./tmp_result_{dt['case_id']}.json''', 'w') as f:
    #         json.dump(metrics, f, indent=2)
