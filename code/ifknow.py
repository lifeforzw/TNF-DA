import torch 
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import json 
import os 
import argparse
from tqdm import tqdm
from dataset import CFDataset 
from torch.utils.data import DataLoader

# from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='EleutherAI/gpt-j-6B')
    parser.add_argument("--data_modelpth", type=str, default='./data/counterfact.json')
    args = parser.parse_args()
    return args



modelpth = ''

def main(
    model_name: str,
    datapth: str ,
    device='cuda:0'

):
        
    with open(datapth, 'r') as fin:
        data = json.load(fin)

    model_pth = modelpth + model_name

    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if 'gpt-j' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_pth, torch_dtype=torch.float16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_pth).to(device)
    model.eval()
    breakpoint()
    model.padding_side = 'left'

    #accelerator = Accelerator()
    dataset = CFDataset(data, tokenizer)
    bs = 40

    index = []
    gen_ = []
    Loader = DataLoader(dataset, batch_size=bs, shuffle=False)

    #model = accelerator.prepare(model)
    #breakpoint()
    for idx, (inputs, targets) in enumerate(tqdm(Loader, desc="processing data")):
        #print(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        l = inputs['input_ids'].shape[-1] + 1
        #inputs = build_position_ids(inputs)
        #breakpoint()
        gen_txt = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_length=l)
        #gen_txt = accelerator.gather_for_metrics(gen_txt)
        #breakpoint()
        gen_txt = tokenizer.batch_decode(gen_txt, skip_special_tokens=True)
        #breakpoint()
        #print(targets, gen_txt)

        for i in range(inputs['input_ids'].shape[0]):
            gen_ += gen_txt
            last_token = gen_txt[i].split(" ")[-1]
            #print(last_token)
            #breakpoint()
            if targets[i].strip() == last_token:
                index.append(idx*bs+i)
    #breakpoint()
    if 'gpt-j' in model_name:
        model_name='gpt-j'
    with open('./data/gen_'+model_name+'.json', 'w') as fout:
        json.dump(gen_, fout, indent=4)

    with open('./data/have_'+model_name+'.json', 'w') as fout:
        json.dump(index, fout)

        




if __name__ == "__main__":
    args = parse_args() 
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #device = 'cuda'
    main(
        args.model_name,
        args.data_modelpth,
        #device
    )