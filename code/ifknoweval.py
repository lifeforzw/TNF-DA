import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json 
import os 
import argparse
from tqdm import tqdm
from dataset import CFDatasetEval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    #parser.add_argument("--data_path", type=str, default='./data/counterfact.json')
    args = parser.parse_args()
    return args


def build_position_ids(input_ids:dict):
    if 'position_ids' in input_ids.keys():
        return 
    position_ids = input_ids['attention_mask'].long().cumsum(-1) - 1
    position_ids[position_ids == -1] = 1
    input_ids['position_ids'] = position_ids.to(input_ids['attention_mask'].device) 
    return input_ids

model_pth = ''
data_dir = ''

def main(
    model_name: str,
    #datapth: str ,
    device

):
    model_pth = model_pth + model_name

    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if 'gpt-j' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_pth, torch_dtype=torch.float16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_pth).to(device)
    model.padding_side = 'left'
    model.eval()

    if 'gpt-j' in model_name:
        model_name = 'gpt-j'
    dataset = CFDatasetEval(model_name, data_dir, tokenizer)
    have_gen = []
    have_loc = []
    have_para = []
    #breakpoint()

    for idx, (gen_input, para_input, loc_input, tar) in enumerate(tqdm(dataset, desc="processing data")):
        for k, v in gen_input.items():
            gen_input[k] = v.to(device)
        for k, v in loc_input.items():
            loc_input[k] = v.to(device)
        for k, v in para_input.items():
            para_input[k] = v.to(device)

        # gen_input = build_position_ids(gen_input)
        # loc_input = build_position_ids(loc_input)
        # para_input = build_position_ids(para_input)

        l_gen = gen_input['input_ids'].shape[-1] + 1
        l_loc = loc_input['input_ids'].shape[-1] + 1
        l_para = para_input['input_ids'].shape[-1] + 1
        gen_txt = model.generate(**gen_input, pad_token_id=tokenizer.eos_token_id, max_length=l_gen)
        gen_txt = tokenizer.batch_decode(gen_txt, skip_special_tokens=True)
        #print(targets, gen_txt)
        loc_txt = model.generate(**loc_input, pad_token_id=tokenizer.eos_token_id, max_length=l_loc)
        loc_txt = tokenizer.batch_decode(loc_txt, skip_special_tokens=True)

        para_txt = model.generate(**para_input, pad_token_id=tokenizer.eos_token_id, max_length=l_para)
        para_txt = tokenizer.batch_decode(para_txt, skip_special_tokens=True)


        # print(gen_txt)
        # print(loc_txt)
        # breakpoint()
        index_gen = []
        index_loc = []
        index_para = []
        for i in range(gen_input['input_ids'].shape[0]):
            #gen_ += gen_txt
            if gen_txt[i] in gen_txt[:i]:
                print('gen repeat')
                continue
            last_token = gen_txt[i].split(" ")[-1]
            #print(last_token)
            if tar == last_token:
                index_gen.append(i)
        have_gen.append(index_gen)

        for i in range(loc_input['input_ids'].shape[0]):
            #gen_ += gen_txt
            if loc_txt[i] in loc_txt[:i]:
                print('loc_repeat')
                continue
            last_token = loc_txt[i].split(" ")[-1]
            #print(last_token)
            if tar == last_token:
                index_loc.append(i)
        #print(index_gen, index_loc)
        have_loc.append(index_loc)

        for i in range(para_input['input_ids'].shape[0]):
            #gen_ += gen_txt
            if para_txt[i] in para_txt[:i]:
                print('para repeat')
                continue
            last_token = para_txt[i].split(" ")[-1]
            #print(last_token)
            if tar == last_token:
                index_para.append(i)
        have_para.append(index_para)
    
    # with open('./data/gen_'+model_name+'.json', 'w') as fout:
    #     json.dump(gen_, fout, indent=4)
    known_id = {
        'known_id': dataset.dataidx,
        'gen_id': have_gen,
        'para_id': have_para,
        'loc_id': have_loc
    }

    with open('./data/evalhave_'+model_name+'.json', 'w') as fout:
        json.dump(known_id, fout)

        




if __name__ == "__main__":
    args = parse_args() 
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cuda:0'
    main(
        args.model_name,
        #args.data_path,
        device
    )