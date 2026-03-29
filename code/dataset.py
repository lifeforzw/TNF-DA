import torch 
from torch.utils.data import Dataset 
import os
import json

class CFDataset(Dataset):
    def __init__(self, data, tok):
        super().__init__()
        self.prompts = [d['requested_rewrite']['prompt'].format(d['requested_rewrite']['subject']) for d in data]
        self.enc_p = tok(self.prompts, padding=True, return_tensors='pt')
        self.targets = [' ' + d['requested_rewrite']['target_true']['str'] for d in data]

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, i):
        return {'input_ids': self.enc_p['input_ids'][i], 'attention_mask': self.enc_p['attention_mask'][i]}, self.targets[i]
    

class CFDatasetEval(Dataset):
    def __init__(self, model_name, data_dir, tok, **kwargs):
        super().__init__()
        source_dir = os.path.join(data_dir, f'counterfact.json')
        data_dir = os.path.join(data_dir, f'have_{model_name}.json')
        self.model_name = model_name
        breakpoint()
        assert os.path.exists(data_dir), 'the known dataset is not exist'

        with open(source_dir, 'r') as f:
            database = json.load(f)

        with open(data_dir, 'r') as f:
            self.dataidx = json.load(f)
        data = [database[i] for i in self.dataidx]
        self.generation_p = []
        self.paraphrase_p = []
        self.target = []
        self.local_p = []
        for data_piece in data:
            self.generation_p.append(tok(data_piece['generation_prompts'], padding=True, return_tensors='pt'))
            self.paraphrase_p.append(tok(data_piece['paraphrase_prompts'], padding=True, return_tensors='pt'))
            self.local_p.append(tok(data_piece['neighborhood_prompts'], padding=True, return_tensors='pt'))
            self.target.append(data_piece['requested_rewrite']['target_true']['str'])
        
    
    def __len__(self):
        return len(self.generation_p)
    
    def __getitem__(self, i):
        return self.generation_p[i], self.paraphrase_p[i], self.local_p[i], self.target[i]

        

    