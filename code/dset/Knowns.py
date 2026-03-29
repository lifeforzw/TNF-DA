import json 
import os 
import torch 
from torch.utils.data import Dataset, DataLoader 
import sys
sys.path.append('../')
from util.utils import batch, build_position_ids, dict2device



class KnownsDataset(Dataset):
    def __init__(self, data_dir: str, model_name, is_generation=False, *args, **kwargs):
        source_dir = os.path.join(data_dir, f'counterfact.json')
        data_dir = os.path.join(data_dir, f'evalhave_{model_name}.json')
        self.model_name = model_name
        assert os.path.exists(data_dir), f'the known dataset is not exist {data_dir}'

        with open(source_dir, 'r') as f:
            database = json.load(f)

        with open(data_dir, 'r') as f:
            self.dataidx = json.load(f)

        self.data = [database[i] for i in self.dataidx['known_id']]
        self._BuildPrompt()
        if is_generation:
            self._BuildGeneration()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def _BuildPrompt(self):
        if 'prompt' in self.data[0].keys():
            return 
        for i in range(len(self.data)):
            subject = self.data[i]['requested_rewrite']['subject']
            self.data[i]['subject'] = subject
            self.data[i]['prompt'] = self.data[i]['requested_rewrite']['prompt'].format(subject)
            self.data[i]['target'] = self.data[i]['requested_rewrite']['target_true']['str']
            self.data[i]['index'] = ''
    

class EvalDataset(KnownsDataset):
    def __init__(self, data_dir: str, model_name, kind='gen_id', *args, **kwargs):
        source_dir = os.path.join(data_dir, f'counterfact.json')
        data_dir = os.path.join(data_dir, f'evalhave_{model_name}.json')
        self.model_name = model_name
        assert os.path.exists(data_dir), 'the known dataset is not exist'

        with open(source_dir, 'r') as f:
            database = json.load(f)

        with open(data_dir, 'r') as f:
            dataidx = json.load(f)  

        known_data = [database[i] for i in dataidx['known_id']]
        self.dataidx = dataidx[kind]
        self.kind = 'generation_prompts' if kind == 'gen_id' else 'neighborhood_prompts'
        self.data = []
        self._BuildPrompt(known_data)
    
    def _BuildPrompt(self, known_data):
        for idx, kd in enumerate(known_data):
            case_id = kd['case_id']
            v = self.dataidx[idx]
            #breakpoint()
            if v == []:
                continue 
            self.data += [{
                'case_id': case_id,
                'subject': kd['requested_rewrite']['subject'],
                'index': str(p),
                'target': kd['requested_rewrite']['target_true']['str'],
                'prompt': kd[self.kind][p]}
                          for p in v]

    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, index):
        return self.data[index]

class Cali_TrainDataset(KnownsDataset):
    def __init__(self, data_dir: str, model_name, tokenizer, N, batch, *args, **kwargs):
        super().__init__(data_dir, model_name, **kwargs)
        self.N = N 
        self.batch = batch
        cali_multiple = batch-1
        ### all subjects
        subjects = [d['subject'] for d in self.data]
        self.prompts = []
        for i in range(N):
            self.prompts.append(self.data[i]['prompt'])
            import random 
            random_indices = random.sample(range(len(subjects)), cali_multiple)
            for random_i in random_indices:
                if subjects[random_i] != self.data[i]['subject']:
                    self.prompts.append(self.data[i]['requested_rewrite']['prompt'].format(subjects[random_i]))
                else:
                    self.prompts.append(self.prompts[-1])
        # with open('./promtps.json', 'w') as f:
        #     json.dump(self.prompts, f, indent=2)
        
        self.enc_input = tokenizer(self.prompts, padding=True, return_tensors='pt')

        self.target = [' '+self.data[d]['requested_rewrite']['target_new']['str'] for d in range(N)]
        self.ori_target = [' '+self.data[d]['requested_rewrite']['target_true']['str'] for d in range(N)]
        self.case_id = [self.data[d]['case_id'] for d in range(N)]
        self.target_id = [tokenizer(d, return_tensors='pt')['input_ids'].squeeze() for d in self.target]
        self.ori_target_id = [tokenizer(d, return_tensors='pt')['input_ids'].squeeze() for d in self.ori_target]
        
        position_ids = self.enc_input['attention_mask'].long().cumsum(-1)-1
        position_ids[position_ids == -1] = 1
        self.enc_input['position_ids'] = position_ids

    def __len__(self):
        return len(self.prompts) 

    def __getitem__(self, i):
        return dict(
            input_ids = self.enc_input['input_ids'][i*self.batch: (i+1)*self.batch],
            attention_mask = self.enc_input['attention_mask'][i*self.batch: (i+1)*self.batch],
            position_ids = self.enc_input['position_ids'][i*self.batch: (i+1)*self.batch],
        ), self.target_id[i], self.ori_target_id[i]


class FT_Cali_Dataset(KnownsDataset):
    def __init__(self, data_dir, model_name, input:dict, subjects:list, target, ori_target, mt, Calisize=200, templates=None, randomtxt=None, *args, **kwargs):
        # input = subject, relation, object
        super().__init__(data_dir, model_name, **kwargs)
        num_edit_samples = len(templates) if templates is not None else 1
        self.prompts = [input['prompt'].format(input['subject'])] * num_edit_samples

        #self.prompts += [input['prompt'].format(input['subject'])] * int(num_edit_samples/2)
        self.size = Calisize
        self.tar = target 
        self.ori_tar = ori_target
        self.case_id = input['case_id']
        import random
        if randomtxt is None:
             
            random_indices = random.sample(range(len(subjects)), self.size-num_edit_samples)
            for random_i in random_indices:
                if subjects[random_i] != input['subject']:
                    self.prompts.append(input['requested_rewrite']['prompt'].format(subjects[random_i]))
                else:
                    self.prompts.append(self.prompts[-1])
        
        else:
            self.prompts += randomtxt
        
        #self.prompts += [f'''{input['subject']} is a''' for _ in range(5)]
        #self.size += 5
        if templates is not None:
            nt = len(templates)
            self.prompts = [templates[random.randint(0,nt-1)].format(self.prompts[ii]) for ii in range(len(self.prompts))]
        #breakpoint()

        self.targets = [mt.tokenizer(' '+self.tar, return_tensors='pt')['input_ids'].squeeze(0)]*num_edit_samples +\
              [torch.tensor([-100])]* (self.size-num_edit_samples)
        self.enc_input = mt.tokenizer(self.prompts, padding=True, return_tensors='pt')
        self.enc_input = build_position_ids(self.enc_input)

        #breakpoint()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return dict(
            input_ids=self.enc_input['input_ids'][i], 
            attention_mask=self.enc_input['attention_mask'][i],
            position_ids=self.enc_input['position_ids'][i]
        ), self.targets[i], self.kl_init_tar[i]
    
    def init_kl_tar(self, mt):
        self.kl_init_tar = []
        bs = 16
        with torch.no_grad():
            for idx in range(0, self.size, bs):
                e_idx = min(self.size, idx+bs)
                batch_inp = dict(
                    input_ids=self.enc_input['input_ids'][idx:e_idx],
                    attention_mask=self.enc_input['attention_mask'][idx:e_idx],
                    position_ids=self.enc_input['position_ids'][idx:e_idx]
                )
                batch_inp = dict2device(batch_inp, mt.device)
                #breakpoint()
                logits = mt.model(**batch_inp).logits
                kl_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                self.kl_init_tar.append(kl_log_probs.detach().clone())
        self.kl_init_tar = torch.cat(self.kl_init_tar, dim=0).to('cpu')

        

      

class TrainDataset(KnownsDataset):
    def __init__(self, data_dir: str, model_name, tokenizer, N, templates=None, is_generation=False, *args, **kwargs):
        super().__init__(data_dir, model_name, **kwargs)
        assert N <= len(self.data), f'invalid N ( > len(data)), {len(self.data)}'
        self.templates = templates
        self.lent = len(self.templates)
        self.N = N
        prompts = [self.data[d]['prompt'] for d in range(N)]
        if self.templates is not None:
            self.prompt = [tt.format(pp) for pp in prompts for tt in self.templates]
            #breakpoint()
        else:
            self.prompt = prompts

        self.target = [' '+self.data[d]['requested_rewrite']['target_new']['str'] for d in range(N)]
        self.ori_target = [' '+self.data[d]['requested_rewrite']['target_true']['str'] for d in range(N)]
        self.case_id = [self.data[d]['case_id'] for d in range(N)]
        self.target_id = [tokenizer(d, return_tensors='pt')['input_ids'].squeeze() for d in self.target]
        self.ori_target_id = [tokenizer(d, return_tensors='pt')['input_ids'].squeeze() for d in self.ori_target]
        #breakpoint()
        self.enc_input = tokenizer(self.prompt, padding=True, return_tensors='pt')

        
        position_ids = self.enc_input['attention_mask'].long().cumsum(-1)-1
        position_ids[position_ids == -1] = 1
        self.enc_input['position_ids'] = position_ids
    
    def __len__(self):
        return len(self.prompt)
    
    def __getitem__(self, i):
        if self.templates:
            return dict(
            input_ids = self.enc_input['input_ids'][i*self.lent:(i+1)*self.lent],
            attention_mask = self.enc_input['attention_mask'][i*self.lent:(i+1)*self.lent],
            position_ids = self.enc_input['position_ids'][i*self.lent:(i+1)*self.lent]
        ), self.target_id[i], self.ori_target_id[i]
        return dict(
            input_ids = self.enc_input['input_ids'][i],
            attention_mask = self.enc_input['attention_mask'][i],
            position_ids = self.enc_input['position_ids'][i]
        ), self.target_id[i], self.ori_target_id[i]


