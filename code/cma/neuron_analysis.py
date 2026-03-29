


import torch 
import os 
import sys
sys.path.append('../')
from util.utils import locate_neurons , fig_heatmap, write2json
import argparse
from mt import ModelandTokenizer
import logging
from dset import KnownsDataset, EvalDataset
import numpy as np
import matplotlib.pyplot as plt
import re
import json 
from collections import Counter, defaultdict

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

colors_map = ['viridis',  'Accent', 'Blues_r', 'blue', 'yellow', 'purple', 'darkblue', 'gold', 'gray']
colors = ['red', 'gray', 'gold', 'green', 'blue']

datapth = ''
modelpth = ''

isreverse = False
result_dir = './result/mid_result/'

def parserargs():
    parser = argparse.ArgumentParser(description='Causal Tracing for neuron')

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--model_name",
        default="gpt2", 
        choices=[
            "gpt2-xl",
            "gpt2-medium"
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neox-20b",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
        ],
    )
    aa('--output_dir', default='./result/{model_name}/causaltrace/')
    aa('--input_dir', default='./result/{model_name}/causaltrace/{edit_type}/')
    aa('--noise_level', default=3, type=float)
    aa('--mode', type=str, default='known_id', choices=['known_id', 'gen_id', 'loc_id'])
    aa('--inter_block', type=str, default='neuron', choices=['neuron', 'hidsize'])
    aa('--erange', type=str, default='whole', choices=['whole', 'subject', 'prompt'])
    aa('--de_rate', type=float, default=1)
    aa('--reverse', type=int, default=0, choices=[0,1])

    return parser.parse_args()


def build_by_prompt(ori_data):
    ### prompt: (subject, o)
    # with open(datapth, 'r') as f:
    #     ori_data = json.load(f)
    ex_data = {}
    #prompt_set = set()
    for case in ori_data:
        cur_prompt = case['requested_rewrite']['prompt']
        if case['requested_rewrite']['prompt'] not in ex_data.keys():    
            ex_data[cur_prompt] = {}
        cur_id = case['case_id']
        for sub_case in ori_data:
            if sub_case['case_id'] < cur_id or sub_case['requested_rewrite']['prompt'] != cur_prompt:
                continue 
            sub_case_target = sub_case['requested_rewrite']['target_true']['str']
            if sub_case_target not in ex_data[cur_prompt].keys():
                ex_data[cur_prompt][sub_case_target] = set()
            ex_data[cur_prompt][sub_case_target].add(( sub_case['case_id'], sub_case['requested_rewrite']['subject']))
    
    return ex_data


def build_by_subject(ori_data):
    ex_data = {}
    prompt_set = set()
    id_set = set()
    for case in ori_data:
        cur_subject = case['subject']
        
        if cur_subject not in ex_data.keys():
            ex_data[cur_subject] = set()
        cur_id = case['case_id']
        if cur_id not in id_set:
            id_set.add(cur_id)
            ex_data[cur_subject].add(( f'''case_{case['case_id']}.npz''', ''))
        #breakpoint()
        if case['prompt'] not in prompt_set:
            prompt_set.add(case['prompt'])
            ex_data[cur_subject].add(( f'''case_{case['case_id']}G{case['index']}.npz''', case['prompt'].replace(cur_subject, '{}')))
        #breakpoint()
    
    return ex_data

def build_by_object(ori_data):
    ex_data = {}
    for case in ori_data:
        cur_object = case['requested_rewrite']['target_true']['str']
        if cur_object not in ex_data.keys():    
            ex_data[cur_object] = set()
        cur_id = case['case_id']
        ex_data[cur_object].add( (cur_id, case['requested_rewrite']['prompt'].format(case['requested_rewrite']['subject'])) )
    
    return ex_data



def ana_neuron_object(inp:dict, ct_dir, ktop, mt, inter_block='neuron'):
    global isreverse
    sim_dict = {}
    for object, prompts in inp.items():
        sim_dict[object] = Counter()
        for case_id, prompt in prompts:
            ct_result_dir = os.path.join(ct_dir, f'''case_{case_id}.npz''')
            if not os.path.exists(ct_result_dir):
                continue
            ct_result = np.load(ct_result_dir, allow_pickle=True)
            delta_p = ct_result['delta_target_p']
            topk_neurons = locate_neurons(delta_p, ktop, mt, interblock=inter_block, raw_neurons=True, reverse=isreverse)
            for n in topk_neurons:
                sim_dict[object][n.item()] += 1
        if len(prompts) >1:
            print(object, len(prompts), len(sim_dict[object].keys()), sim_dict[object].most_common(3))
    
    return sim_dict


def ana_neuron_subject(inp:dict, ct_dir, ktop, mt, inter_block='neuron'):
    global isreverse
    sim_dict = {}
    for subject, prompts in inp.items():
        sim_dict[subject] = Counter()
        for case, prompt in prompts:
            ct_result_dir = os.path.join(ct_dir, case)
            if not os.path.exists(ct_result_dir):
                print(ct_result_dir)
                continue
            ct_result = np.load(ct_result_dir, allow_pickle=True)
            delta_p = ct_result['delta_target_p']
            topk_neurons = locate_neurons(delta_p, ktop, mt, interblock=inter_block, raw_neurons=True, reverse=isreverse)
            for n in topk_neurons:
                sim_dict[subject][n.item()] += 1
        if len(prompts) > 1:
            print(subject, len(prompts), len(sim_dict[subject].keys()), sim_dict[subject].most_common(3))
            if len(prompts) == 3 and len(sim_dict[subject].keys()) < 30:
                print(sim_dict[subject])
                breakpoint()
                
    
    return sim_dict


def ana_neuron_prompt_allobj(inp:dict, ct_dir, ktop, mt, inter_block='neuron'):
    global isreverse
    #breakpoint()
    sim_dict = {}
    for prompt, osdict in inp.items():
        #print(prompt, osdict)
        sim_dict[prompt] = Counter()
        #breakpoint()
        for obj in osdict.keys():
            #breakpoint()
            case_list = osdict[obj]
            for case in case_list:
                case_id, sub = case 
                ct_result_dir = os.path.join(ct_dir, f'''case_{case_id}.npz''')
                if not os.path.exists(ct_result_dir):
                    continue
                ct_result = np.load(ct_result_dir, allow_pickle=True)
                delta_p = ct_result['delta_target_p']
                topk_neurons = locate_neurons(delta_p, ktop, mt, interblock=inter_block, raw_neurons=True, reverse=isreverse)
                #topk_neuron = torck.topk(delta_p.view(-1), sumk, largest=True)
                for n in topk_neurons:
                    sim_dict[prompt][n.item()] += 1
        if sum([len(v) for v in osdict.values()]) >1:
            suffix = 'prompt'
            print(prompt, ' ', sum([len(v) for v in osdict.values()]), len(list(sim_dict[prompt].keys())), sim_dict[prompt].most_common(3))
    return sim_dict

def ana_neuron_prompt(inp:dict, ct_dir, ktop, mt, inter_block='neuron'):
    ### TODO: no matter obj just prompt
    global isreverse
    sim_dict = {}
    for prompt, osdict in inp.items():
        #print(prompt, osdict)
        sim_dict[prompt] = {}
        for obj in osdict.keys():
            print(f'''{prompt} {obj}: ''')
            sim_dict[prompt][obj] = Counter()
            case_list = osdict[obj]
            for case in case_list:
                case_id, sub = case 
                ct_result_dir = os.path.join(ct_dir, f'''case_{case_id}.npz''')
                if not os.path.exists(ct_result_dir):
                    continue
                ct_result = np.load(ct_result_dir, allow_pickle=True)
                delta_p = ct_result['delta_target_p']
                topk_neurons = locate_neurons(delta_p, ktop, mt, interblock=inter_block, raw_neurons=True, reverse=isreverse)
                #topk_neuron = torck.topk(delta_p.view(-1), sumk, largest=True)
                for n in topk_neurons:
                    sim_dict[prompt][obj][n.item()] += 1
            if len(case_list) > 1:
                print(prompt, len(case_list), len(list(sim_dict[prompt][obj].keys())), sim_dict[prompt][obj].most_common(3))
    return sim_dict
                




        



def rebuild_data(nns):
    x_, y_ = [], []
    for k, v in nns.items():
        xx = re.findall('[0-9]', k)
        if len(xx) == 1:
            xx = int(xx[0])
        else:
            xx = 10*int(xx[0]) + int(xx[1])
        y_ += [ii for ii in v]
        x_ += [xx]*len(v)
    assert len(x_) == len(y_)
    #print(x_, y_)
    return np.array(x_), np.array(y_)


def reshape_simdict(simdict:dict, topa=3):
    ### from prompt:[neurons] --> neuron:[prompt]
    n2p = defaultdict(list)
    for k, v in simdict.items():
        ### k-->prompt, v-->counter of neuron
        topneuron = v.most_common(topa)
        for neuron, _ in topneuron:
            n2p[neuron].append(k)
    
    return n2p








if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    args = parserargs()
    input_dir = args.input_dir
    de_rate = args.de_rate
    interrange = args.erange 
    nlevel = args.noise_level
    isreverse = True if args.reverse else False
    result_dir = result_dir+'re_' if isreverse else result_dir

    input_dir = os.path.join(input_dir, f'''corfirst{int(de_rate*100)}_{interrange}_results_s{nlevel}/''')
    input_dir = input_dir.format(model_name=args.model_name, edit_type=args.inter_block)
    mt = ModelandTokenizer(args.model_name)
    ktop = 0.01
    topa = 3

    print('---------------------same prompt---------------------')
    Knowns = KnownsDataset(data_dir=datapth, model_name=args.model_name)
    post_data_prompt = build_by_prompt(Knowns)
    print(len(list(post_data_prompt.keys())))
    # for k in post_data.keys():
    #     print(k, ': ', len(post_data[k]))
    prompt_simdict = ana_neuron_prompt_allobj(post_data_prompt, input_dir, ktop, mt)
    prompt_n2p = reshape_simdict(prompt_simdict, topa)
    write2json(prompt_n2p, result_dir+'prompt.json')
    #breakpoint()
    #ana_neuron_prompt(post_data, input_dir, ktop, mt)
    #print(so_sum)

    print('---------------------same subject---------------------')

    # Evals = EvalDataset(data_dir=datapth, model_name=args.model_name, kind='gen_id')
    # post_data_sub = build_by_subject(Evals)
    # print(len(list(post_data_sub.keys())))
    # #print(post_data_sub)
    # # for k in post_data_sub.keys():
    # #     print(k, ': ', len(post_data_sub[k]))
    # subject_simdict = ana_neuron_subject(post_data_sub, input_dir, ktop, mt)
    # subject_n2p = reshape_simdict(subject_simdict, topa)
    # write2json(subject_n2p, result_dir+'subject.json')


    print('---------------------same object---------------------')
    Knowns = KnownsDataset(data_dir=datapth, model_name=args.model_name)
    post_data_obj = build_by_object(Knowns)
    print(len(list(post_data_obj.keys())))
    # print(post_data_obj)
    object_simdict = ana_neuron_object(post_data_obj, input_dir, ktop, mt)
    object_n2p = reshape_simdict(object_simdict, topa)
    write2json(object_n2p, result_dir+'object.json')
    #breakpoint()
    
    print('---------------------sum up result---------------------')
    so_sum = 0
    toptime, sectime, thirdtime, mintime = 0,0,0,0
    act_neuron_nums = 0
    #breakpoint()
    for k, v in prompt_simdict.items():
        
        if sum([len(g) for g in post_data_prompt[k].values()]) <= 1:
            continue
        so_sum += sum([ len(vv) for vv in post_data_prompt[k].values()]) # 当前prompt有效知识总数
        a,b,c = v.most_common(3)
        #breakpoint()
        act_neuron_nums += len(v.keys())
        toptime, sectime, thirdtime,mintime = toptime+a[1], sectime+b[1], thirdtime+c[1], mintime+1
    #breakpoint()
    if ktop < 1:
        ktop = int(mt.inner_size * mt.num_layers*ktop)
    print(f'''min_frequency = {mintime/so_sum :.4f} 
            frequency_top = {toptime/so_sum :.4f} 
            frequency_sec = {sectime/so_sum :.4f}  
            frequency_thi = {thirdtime/so_sum :.4f} 
            repeat_rate =  {act_neuron_nums/(so_sum*ktop)}  (all_neuron in top{ktop} / num_data * {ktop})''') 


    # so_sum = sum([len(vv) for kk,vv in post_data_sub.items() if len(post_data_sub[kk]) > 1])
    # toptime, sectime, thirdtime, mintime = 0,0,0,0
    # act_neuron_nums = 0
    # for k, v in subject_simdict.items():
    #     if len(post_data_sub[k]) <= 1:
    #         continue
    #     try:
    #         a,b,c = v.most_common(3)
    #     except ValueError:
    #         #print(v)
    #         continue
    #     act_neuron_nums += len(v.keys())
    #     toptime, sectime, thirdtime, mintime = toptime+a[1], sectime+b[1], thirdtime+c[1], mintime+1
    # print(f'''minrate=={mintime/so_sum :.2f}  {toptime/so_sum :.2f}, {sectime/so_sum :.2f}, {thirdtime/so_sum :.2f} \t {act_neuron_nums/(so_sum*ktop)}''') 


    so_sum = sum([len(vv) for kk,vv in post_data_obj.items() if len(post_data_obj[kk]) > 1])
    toptime, sectime, thirdtime, mintime = 0,0,0,0
    act_neuron_nums = 0
    for k, v in object_simdict.items():
        if len(post_data_obj[k]) <=1:
            continue
        a,b,c = v.most_common(3)
        act_neuron_nums += len(v.keys())
        toptime, sectime, thirdtime, mintime = toptime+a[1], sectime+b[1], thirdtime+c[1], mintime+1
    print(f'''minrate=={mintime/so_sum :.2f}  {toptime/so_sum :.2f}, {sectime/so_sum :.2f}, {thirdtime/so_sum :.2f} \t {act_neuron_nums/(so_sum*ktop)}''') 
