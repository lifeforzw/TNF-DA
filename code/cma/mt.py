### class for mt

import torch 
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os 
import sys
sys.path.append('../')
import numpy as np
from util.runningstats import Covariance, tally
from datasets import load_dataset
from dset import tok_dataset

import sys 
sys.path.append('../')
from util.utils import batch, find_token_range
import re
from tqdm import tqdm
import baukit
import time
from collections import OrderedDict, defaultdict
import math

datapth = ''
modelpth = ''

class ModelandTokenizer:
    def  __init__(
        self,
        model_name=None,
        tokenizer=None,
        device='cuda:0'
    ):
        assert model_name is not None
        self.model_name = model_name if 'gpt-j' not in model_name else 'gpt-j'
        #logger.info('loading model and tokenizer...')
        pth = os.path.join(modelpth, model_name)
        print(pth)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(pth)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        
        self.device = device 
        self.tokenizer = tokenizer
        if 'gpt-j' in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(pth, torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pth)
        self.model.eval().to(device)
        self.config = AutoConfig.from_pretrained(pth)
        #breakpoint()
        
        self.layer_names = [
            n for n, m in self.model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)
        self.hidden_size = self.config.n_embd
        self.inner_size = 4*self.hidden_size if self.config.n_inner is None else self.config.n_inner
        self.mlpin = '.c_fc' if 'gpt2' in model_name else '.fc_in'
        self.mlpout = '.c_proj' if 'gpt2' in model_name else '.fc_out'
        #logger.info('model and tokenizer complete initialization ')
    
    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"           
        )
    
    def getlayername(self, num, kind=None):
        if hasattr(self.model, "transformer"):
            if kind == "embed":
                return "transformer.wte"
            return f'transformer.h.{num}{"" if kind is None else "." + kind}'
        if hasattr(self.model, "gpt_neox"):
            if kind == "embed":
                return "gpt_neox.embed_in"
            if kind == "attn":
                kind = "attention"
            return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
        assert False, "unknown transformer structure"
    
    def encode_input(self, input):
        #data in cpu#
        enc_input = self.tokenizer(input, return_tensors='pt')
        position_ids = enc_input['attention_mask'].long().cumsum(-1) - 1
        position_ids[position_ids == -1] = 1
        enc_input['position_ids'] = position_ids
        return enc_input

    def collect_embedding_gaussian(self, nlevel=1.0):
        m,c = self._get_embedding_cov()
        return self._make_generator_transform(mean=None, cov=c, nlevel=nlevel)

    def _make_generator_transform(self, mean=None, cov=None, nlevel=1.0):
        d = len(mean) if mean is not None else len(cov)
        device = mean.device if mean is not None else cov.device
        layer = torch.nn.Linear(d, d, dtype=torch.double)
        baukit.set_requires_grad(False, layer)
        layer.to(device)
        layer.bias[...] = 0 if mean is None else mean
        if cov is None:
            layer.weight[...] = torch.eye(d).to(device)
        else:
            x, ss, v = cov.svd()
            #breakpoint()
            ww = ss.sqrt()[None, :] * v
            ww = ww*math.sqrt(nlevel)
            layer.weight[...] = ww
        return layer
    
    def _get_embedding_cov(self):

        def get_ds():
            ds_name = "wikitext"
            raw_ds = load_dataset(
                os.path.join(datapth, ds_name),
                dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
            )
            try:
                maxlen = self.model.config.n_positions
            except:
                maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
            return tok_dataset.TokenizedDataset(raw_ds["train"], self.tokenizer, maxlen=maxlen)
        ds = get_ds()
        sample_size = 1000
        batch_size = 5
        filename = None
        batch_tokens = 100

        progress = lambda x, **k: x

        stat = Covariance()
        loader = tally(
            stat,
            ds,
            cache=filename,
            sample_size=sample_size,
            batch_size=batch_size,
            collate_fn=tok_dataset.length_collation(batch_tokens),
            pin_memory=True,
            random_sample=1,
            num_workers=0,
        )
        with torch.no_grad():
            for batch_group in loader:
                for batch in batch_group:
                    batch = tok_dataset.dict_to_(batch, "cuda")
                    del batch["position_ids"]
                    with baukit.Trace(self.model, self.getlayername(0, "embed")) as tr:
                        self.model(**batch)
                    feats = tok_dataset.flatten_masked_batch(tr.output, batch["attention_mask"])
                    stat.add(feats.cpu().double())
        return stat.mean(), stat.covariance()

    def _build_noise_fn(self, intervention, randseed=1):
        rs = np.random.RandomState(randseed)
        prng = lambda *shape: rs.randn(*shape)
        if isinstance(intervention, float):
            intervention_fn = lambda x: intervention* x 
        else:
            intervention_fn = intervention
        return prng, intervention_fn

    def single_neuron_intervention(
        self, 
        context, 
        e_range,
        tgt_layer,
        neurons ,
        intervention=0.1,
        intervention_type='diff',
        clco_layers = None
    ):
        '''
        here neurons := a dict that key is layername and value is the idx
        e_range --> mask
        no need for padding, one sentence for repeat
        '''
        
        prng, intervention_fn = self._build_noise_fn(intervention)
        embed_layername = self.getlayername(0, 'embed')
        seqlen = torch.sum(context['attention_mask'][0])
        #len_neurons = sum(len(v) for v in tgt_layer_neurons.values())
        cp_context = {}
        batch_size = max(2, len(neurons)+1)
        for k, v in context.items():
            cp_context[k] = v.clone().detach().repeat(batch_size, 1)
        
        #print(context['input_ids'])
        indices = torch.arange(len(neurons))
        hooked_layers = [embed_layername]
        if tgt_layer is not None:
            hooked_layers.append(tgt_layer)
        if clco_layers is not None:
            hooked_layers += clco_layers


        delta_act = [] if clco_layers is not None else None
        #breakpoint()
        def interv(x, cur_layer):
            if cur_layer == embed_layername:
                #b, e = e_range
                #len_noise = int(sum(e_range).item())
                #breakpoint()
                nn = torch.from_numpy(prng(1, seqlen, x.shape[-1])) * e_range.unsqueeze(0).unsqueeze(-1)
                #breakpoint()
                noise_data = intervention_fn(
                    nn.repeat(max(1, len(neurons)), 1, 1)
                    #torch.from_numpy(prng(samples, seqlen, x.shape[-1])).repeat(max(1, len(neurons)), 1, 1)
                ).to(self.device)

                

                if intervention_type == 'diff':
                    #print('diff')
                    x[:-1, :, :] += noise_data  
                elif intervention_type == 'replace':
                    x[:-1, :, :] = noise_data

                return x 
            
            
            if cur_layer == tgt_layer and tgt_layer is not None:
                h = x[0] if isinstance(x, tuple) else x
                #breakpoint()
                # for idx, n in enumerate(neurons):
                #     x[1+idx, :, n] = h[0, :, n]

                h[indices, :, neurons] = h[-1, :, neurons].T

                return x
            
            if clco_layers is not None and cur_layer in clco_layers:
                assert x.shape[0] == 2, 'clco run batch is not 2!'
                #breakpoint()
                delta_act.append(torch.max(torch.abs((x[1] - x[0])/x[1]), dim=0).values)
            
            return x
        #breakpoint()
        with torch.no_grad():
            #output = self.model(**cp_context)
            with baukit.TraceDict(self.model, hooked_layers, edit_output=interv) as ret:
                # st  =time.time()
                output = self.model(**cp_context)
                # et = time.time()
                # print("3 ", et-st)
        #breakpoint()
        # time.sleep(0.5)
        #print(output.logits[0,0,0])
        cc_probs = torch.softmax(output.logits[:-1, -1, :], dim=-1)
        #tmp = [cc_probs[i:i+1].mean(dim=0) for i in range(0, max(1,len(neurons)))]
        #cc_probs = torch.stack(tmp)
        cl_probs = torch.softmax(output.logits[-1, -1, :], dim=-1)
        #breakpoint()

        #breakpoint()
        delta_act = torch.stack(delta_act) if delta_act is not None else None
        return cc_probs, cl_probs, delta_act

    def neurons_intervention(
        self,
        context,
        subject,
        target,
        bsize=800,
        neuron_range=None,
        intervention=0.1,
        intervention_type='diff',
        interblock='neuron',
        interrange='whole',
        times=5,
    ):
        ''' interblock == neuron or hidden_size'''
        name_suffix = 'mlp'+self.mlpin if interblock == 'neuron' else 'mlp'+self.mlpout
        inter_width = self.inner_size if interblock == 'neuron' else self.hidden_size
        
        final_probs = torch.zeros(self.num_layers, inter_width)
        seqlen = torch.sum(context['attention_mask'][0])
        e_range = torch.ones(seqlen)
        if interrange == 'subject':
            e, b = find_token_range(self.tokenizer, context['input_ids'][0], subject)
            e_range[0:e] = 0
            e_range[b:] = 0
        elif interrange == 'prompt':
            e, b = find_token_range(self.tokenizer, context['input_ids'][0], subject)
            e_range[e:b] = 0
        #breakpoint()
        mlp_layer_names = [self.getlayername(i, name_suffix) for i in range(self.num_layers)]
        if neuron_range is None:    
            for idx, layer in tqdm(enumerate(mlp_layer_names), desc='up'):
                layer_probs = []
                #for neurons in batch(range(self.hidden_size), bsize):
                
                for neurons in batch(range(inter_width), bsize):
                    #print(neurons)
                    #layer_neurons = {layer: neurons}
                    probs, _, _ = self.single_neuron_intervention(context, e_range, layer, neurons, intervention, intervention_type)
                    probs = probs[:, target].squeeze().cpu()
                    layer_probs.append(probs)
                layer_probs = torch.cat(layer_probs)
                #print(torch.max(layer_probs), torch.min(layer_probs))
                final_probs[idx]=layer_probs
        
        else:
            # build neurons from neuron_range
            layer_neurons = defaultdict(list)
            for n in neuron_range:
                layer_neurons[self.getlayername(n//inter_width, name_suffix)].append((n%inter_width).item())
            for idx, layer in enumerate(mlp_layer_names):
                #print(layer, ' ', len(layer_neurons[layer]))
                if layer_neurons[layer] == []:
                    continue
                for neurons in batch(layer_neurons[layer], bsize):
                    #build dict

                    probs, _, _ = self.single_neuron_intervention(context, e_range, layer, neurons, intervention, intervention_type)
                    probs = probs[:, target].squeeze().cpu()
                    #breakpoint()
                    #layer_probs.append(probs)
                    assert torch.sum(final_probs[idx][neurons]) == 0, f"neurons repeat!!!!!!!!!!!!  {torch.sum(final_probs[idx][neurons])}"
                    final_probs[idx][neurons]=probs
        
        return final_probs
    
    def multi_intervention(
        self, 
        context,
        subject,
        target,
        chosen_neurons,
        intervention=0.1,
        intervention_type='diff',
        interblock='neuron',
        interrange='whole',
    ):
        ''' interblock == neuron or hidden_size'''
        name_suffix = self.mlpin if interblock == 'neuron' else self.mlpout
        inter_width = self.inner_size if interblock == 'neuron' else self.hidden_size
        
        final_probs = torch.zeros(self.num_layers, inter_width)
        seqlen = torch.sum(context['attention_mask'][0])
        e_range = torch.ones(seqlen)
        if interrange == 'subject':
            e, b = find_token_range(self.tokenizer, context['input_ids'][0], subject)
            e_range[0:e] = 0
            e_range[b:] = 0
        elif interrange == 'prompt':
            e, b = find_token_range(self.tokenizer, context['input_ids'][0], subject)
            e_range[e:b] = 0
        #breakpoint()
        #mlp_layer_names = [self.getlayername(i, name_suffix) for i in range(self.num_layers)]
        chosen_neurons_cp = {}
        for k, v in chosen_neurons.items():
            chosen_neurons_cp[k+name_suffix] = v
        probs = self.multi_neuron_intervention(context, chosen_neurons_cp)
        
        return probs


    def multi_neuron_intervention(
        self, 
        context,
        chosen_neurons,
        samples=10, 
        intervention=0.1,
        intervention_type='diff'
    ):
        prng, intervention_fn = self._build_noise_fn(intervention)
        embed_layername = self.getlayername(0, 'embed')
        seqlen = torch.sum(context['attention_mask'][0])
        #breakpoint()
        cp_context = {}
        batch_size = samples + 1
        for k, v in context.items():
            cp_context[k] = v.clone().detach().repeat(batch_size, 1)
        
        layer_names = [embed_layername] + list(chosen_neurons.keys())

        def interv(x, cur_layer):
            if cur_layer == embed_layername:
                noise_data = intervention_fn(
                    torch.from_numpy(prng(x.shape[0]-1, seqlen, x.shape[-1]))
                ).to(self.device)

                if intervention_type == 'diff':
                    x[1:, -seqlen:, :] += noise_data  
                elif intervention_type == 'replace':
                    x[1:, -seqlen:, :] = noise_data

                return x 
            
            
            if cur_layer in chosen_neurons.keys():
                #breakpoint()
                h = x[0] if isinstance(x, tuple) else x
                neuron_id = chosen_neurons[cur_layer]
                x[1:, :, neuron_id] = h[0, :, neuron_id]
                #print(x)
                return x
            
            return x
        
            # clean_corrupt
        with torch.no_grad():
            with baukit.TraceDict(self.model, layer_names, edit_output=interv) as ret:
                output = self.model(**cp_context)
        
        cc_probs = torch.softmax(output.logits[1:, -1, :], dim=-1).mean(dim=0)

        return cc_probs

    
    def clcoRun(self, context, subject, target, intervention=0.1, intervention_type='diff', interblock='neuron', interrange='whole'):
        name_suffix = 'mlp'+self.mlpin if interblock == 'neuron' else 'mlp'+self.mlpout
        all_layers = [self.getlayername(i, name_suffix) for i in range(self.num_layers)]        
        seqlen = torch.sum(context['attention_mask'][0])
        e_range = torch.ones(seqlen)
        if interrange == 'subject':
            e, b = find_token_range(self.tokenizer, context['input_ids'][0], subject)
            e_range[0:e] = 0
            e_range[b:] = 0
        elif interrange == 'prompt':
            e, b = find_token_range(self.tokenizer, context['input_ids'][0], subject)
            e_range[e:b] = 0
        cor_probs, cl_probs, delta_activation =  \
                self.single_neuron_intervention(context, e_range, None, [], intervention, intervention_type, clco_layers=all_layers)
        correct_prediction = (torch.argmax(cl_probs.cpu()) == target)
        cor_probs, cl_probs = cor_probs[:, target].squeeze().cpu(), cl_probs[target].squeeze().cpu()
        #print(cor_probs, cl_probs)
        return cor_probs, cl_probs, correct_prediction, delta_activation
    
    def neuron2distribution(self, neurons, target):
        target = ' ' + target 
        target = self.encode_input(target)['input_ids']

        lm_head = self.model.lm_head
        stack_top_neurons = []
        neuron_nums = 0
        for k, v in neurons.items():
            neuron_nums += len(v)
            print(k)
            stack_top_neurons.append(self.model.state_dict()[k+'.c_proj.weight'][v, :].clone().detach())
        
        stack_top_neurons = torch.concat(stack_top_neurons)
        assert stack_top_neurons.shape[0] == neuron_nums , 'neuron2distribution has something wrong.'

        ret_distribution = torch.softmax(lm_head(stack_top_neurons), dim=-1)
        tmp = torch.max(ret_distribution, dim=-1)
        #print(target, ret_distribution[:, target], tmp)
        for i in range(neuron_nums):
            print(self.tokenizer.decode(tmp[1][i]))
        #print(stack_top_neurons.shape)

    def single_neuron_contribution(self, context, layer, neurons, step):
        pass

    def neuron_contribution(self, context, target, bsize=800, step=20, **kwargs):
        final_contribution = []
        mlp_layer_names = [self.getlayername(i, 'mlp') for i in range(self.num_layers)]
        for layer in tqdm(mlp_layer_names, 'layers'):
            layer_contribution = []
            for neurons in batch(range(self.hidden_size), bsize):
                #print(neurons)
                probs, _ = self.single_neuron_intervention(context, layer, neurons, step)
                probs = probs[:, target].squeeze().cpu()
                layer_probs.append(probs)
            layer_probs = torch.cat(layer_probs)
            #print(torch.max(layer_probs), torch.min(layer_probs))
            final_probs.append(layer_probs)
        
        final_probs = torch.stack(final_probs)

        return final_contribution





        


        

