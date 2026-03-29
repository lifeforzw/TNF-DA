### class trainer

import torch 
from baukit import TraceDict
import numpy as np 
import sys 
sys.path.append('../')
from util.utils import dict2device, dict_cat
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
from itertools import chain
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os
from evaluate import compute_rewrite_quality_counterfact, my_compute_rewrite_quality_counterfact
from copy import deepcopy
import json
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

## TODO 构建一个edited model for neuron edit

modelpth = ''


class Trainer(object):
    def __init__(self, model, tokenizer, hidden_size, inner_size, dataset, neuron_ids, device, epochs, criterion, optimizer, theta, alpha,
                  edit_type='neuron', lr=0.01, isabs=False, edit_pos='c_proj', *args, **kwargs):
        '''
        dataset := a list of each edit [data, new_target]
        neuron_id := a list of dict respond to dataset, each sub dict is (key=layername, value=list of neuron_id)
        '''
        self.model = model
        self.tokenizer = tokenizer

        self.device = device
        self.criterion = criterion 
        self.optimizer = optimizer
        self.dataset = dataset
        self.epochs = epochs
        self.lr = lr
        self.theta = theta # loss rate
        self.alpha = alpha
        self.isabs = isabs
        self.edit_type = edit_type
        #self.inner_size = inner_size
        self.hidden_size = hidden_size
        if self.edit_type == 'hidsize':
            self.hidden_size = inner_size
        
        self.edit_pos = edit_pos


        self.neuron_ids = neuron_ids 
        #neuron_nums = sum([len(v) for v in neuron_ids.values()])
        self.delta = {}


         
    def build_delta(self, neuron_id):
        '''
        input neuron_id a dict
        build delta weight
        '''
        delta_weights = {}
        for k, v in neuron_id.items():
            delta_weights[k] = nn.Parameter(torch.zeros(len(v), self.hidden_size, dtype=torch.float).to(self.device))
            #delta_weights[k].requires_grad=True
        
        return delta_weights
    

    def copy_weight(self, neuron_id):
        cp_layer_weight = {}
        for k, v in neuron_id.items():
            layername = k + '.c_proj.weight'
            #breakpoint()
            cp_layer_weight[k] = self.model.state_dict()[layername][v, :].clone()
        
        return cp_layer_weight


    def train(self, save_pth, eval_dir, gen_test_vars, isdebug=False):
        ### TODO: delta_w .shape
        #self.model.eval()
        #self.model.train()
        cur_loss = 0
        #breakpoint()

        #tensorboard
        edit_model = Edited_Model(self.hidden_size, self.device, copy=False, edit_type=self.edit_type, edit_pos=self.edit_pos)
        if isdebug:
            tar_writer = SummaryWriter('./runs/tar')
            ori_writer = SummaryWriter('./runs/ori')
            kl_writer = SummaryWriter('./runs/kl_loss')
            genl_writer = SummaryWriter('./runs/genloss')
            weil_writer = SummaryWriter('./runs/weiloss')
            suml_writer = SummaryWriter('./runs/sumloss')

        for idx, (inp, tar, ori_tar) in enumerate(tqdm(self.dataset, desc='data:')):
            # print('sleeping')
            # time.sleep(5)
            #breakpoint()
            edit_model.init_model(self.model)
            #breakpoint()
            neuron_index = self.neuron_ids[idx]
            if neuron_index == {}:
                print('jump one')
                continue
            #delta_w = self.build_delta(neuron_index)
            edit_model.init_delta(neuron_index)
            edit_model.build_hook()
            #breakpoint()
            ### cp_layer_weight = self.copy_weight(neuron_index)
            ### ori_layer_weight = self.copy_weight(neuron_index)
            #breakpoint()
            ### opt = self.optimizer(cp_layer_weight.values(), lr=self.lr)

            # for v in edit_model.delta_w.values():
            #     print(v.values())
            #breakpoint()
            opt = self.optimizer([v for vv in edit_model.delta_w.values() for v in vv.values()], lr=self.lr)
            #breakpoint()
            if tar.dim() > 0:
                tar = tar[0]

            #print(self.tokenizer.decode(tar))
            inp = dict2device(inp, self.device)
            tar = tar.to(self.device)
            # ori_tar = ori_tar.to(self.device)
            #breakpoint()
            #breakpoint()
            display = 0
            cur_epoch = self.epochs

            kl_init_tar = None
            tar_prob = None
            #tensorboard

            for epoch in range(self.epochs):
                # for layer_name, v in neuron_index.items():
                #     self.model.state_dict()[layer_name + '.c_proj.weight'][v, :] = cp_layer_weight[layer_name] + delta_w[layer_name]
                ### for layer_name, v in neuron_index.items():
                ###     self.model.state_dict()[layer_name + '.c_proj.weight'][v, :] = cp_layer_weight[layer_name]

                opt.zero_grad()
                out = edit_model(inp)
                #breakpoint()
                logits = out[0].unsqueeze(0) if len(out[0].shape) == 2 else out[0]

                #probs = torch.softmax(logits[:, -1, :], dim=-1)
                probs = torch.softmax(logits, dim=-1)
                if tar_prob is None:
                    tar_prob = probs[0,-1,ori_tar].detach().clone()
                
                # scalar the tar prob
                if isdebug:
                    tar_writer.add_scalar(f'''{idx}_prob''', probs[0,-1,tar], epoch)
                    ori_writer.add_scalar(f'''{idx}_prob''', probs[0,-1,ori_tar], epoch)

                #print(epoch, torch.argmax(probs), tar, probs[tar])
                #breakpoint()
                # if torch.argmax(probs[0, -1, :]) == tar:
                #     display += 1
                #     if display > 1:
                #         cur_epoch = epoch
                #         break

                ### shift for label and logits
                #breakpoint()
                # labels = torch.concat([inp['input_ids'], tar.unsqueeze(0).repeat(inp['input_ids'].shape[0], 1)], dim=1)
                # labels[labels == self.tokenizer.pad_token_id] = -100
                # shifted_labels = labels[:, 1:].contiguous()
                # shifted_out = probs.contiguous()

                # loss_gen = self.criterion(shifted_out.view(-1, shifted_out.shape[-1]), shifted_labels.view(-1))

                #breakpoint()
                loss_gen = self.criterion(logits[0,-1,:], tar)

                #loss_p = torch.exp(probs[ori_tar] - probs[tar])
                #breakpoint()
                kl_log_probs = nn.functional.log_softmax(logits[:], dim=-1)
                if kl_init_tar is None:
                    kl_init_tar = kl_log_probs.detach().clone()
                kl_loss = self.alpha*nn.functional.kl_div(
                    kl_init_tar, kl_log_probs, log_target=True, reduction='batchmean'
                )

                #loss_weight = sum([torch.sum(torch.norm(v, dim=-1)) for v in edit_model.delta_w.values()]) / edit_model.num_delta # norm 2
                loss_weight = self.theta*max([torch.max(torch.abs(v)) for vv in edit_model.delta_w.values() for v in vv.values()]) # norm 8

                loss = loss_weight + kl_loss + loss_gen
                if isdebug:
                    kl_writer.add_scalar(f'''{idx}_loss''', kl_loss, epoch)
                    genl_writer.add_scalar(f'''{idx}_loss''', loss_gen, epoch)
                    weil_writer.add_scalar(f'''{idx}_loss''', loss_weight, epoch)
                    suml_writer.add_scalar(f'''{idx}_loss''', loss, epoch)
                #breakpoint()
                #loss = loss_gen
                #print(loss_gen.item(), loss_weight.item(), loss_p.item(), loss.item())
                #breakpoint()
                #cur_loss = loss.item()
                loss.backward()
                opt.step()

            
            # for layer_name, v in neuron_index.items():
            #     self.model.state_dict()[layer_name + '.c_proj.weight'][v, :] = cp_layer_weight[layer_name] + delta_w[layer_name]
            if isdebug:
                ori_writer.close()
                tar_writer.close()
                kl_writer.close()
                genl_writer.close()
                weil_writer.close()
                suml_writer.close()

            edit_model.remove_hook()
            edit_model.update_model()
            #breakpoint()
            eval(edit_model.model, self.tokenizer, self.dataset, idx, edit_model.num_delta, gen_test_vars, output_dir=eval_dir, isabs=self.isabs, epoch=cur_epoch)
            edit_model.rollback()
            #del(edit_model)
            # self.edit_model.model.eval()
            # oout = self.model(inp)
            # pprobs = torch.softmax(oout[0][-1, :], dim=-1)
            # print('eval: ', torch.argmax(pprobs), tar)
            # breakpoint()
            #self.delta[idx] = delta_w

            # self.eval(idx, self.dataset.data[idx])

        #edit_model.model.eval()
        #torch.save(self.edit_model.model.state_dict(), save_pth)

    def kl_ft_train_new(self, data_idx, eval_dir, gen_test_vars=[None, None], isdebug=False):
            ### TODO: delta_w .shape
            #self.model.eval()
            #self.model.train()
            cur_loss = 0
            #breakpoint()

            #tensorboard
            edit_model = Edited_Model(self.hidden_size, self.device, copy=False, edit_type=self.edit_type, edit_pos=self.edit_pos)
            # tar_writer = SummaryWriter('./kl_runs/tar')
            # ori_writer = SummaryWriter('./kl_runs/ori')
            if isdebug:
                kl_writer = SummaryWriter('./kl_runs/kl_loss')
                genl_writer = SummaryWriter('./kl_runs/genloss')
                weil_writer = SummaryWriter('./kl_runs/weiloss')
                suml_writer = SummaryWriter('./kl_runs/sumloss')

            def my_collate_fn(data):
                enc_inp, tar, kl_init_tar = [], [], []
                for batch in data:
                    enc_inp.append(batch[0])
                    tar.append(batch[1])
                    kl_init_tar.append(batch[2])
                    #indices.append(batch[3])
                #breakpoint()
                enc_inp = dict_cat(enc_inp)
                #print(tar)
                tar = torch.cat(tar, dim=0)
                kl_init_tar = torch.stack(kl_init_tar, dim=0)
                #indices = torch.stack(indices, dim=0)
                #breakpoint()
                return enc_inp, tar, kl_init_tar
                
            dl = DataLoader(self.dataset, batch_size=20, shuffle=True, collate_fn=my_collate_fn)

            # new_tar = self.tokenizer(' '+self.dataset.tar, return_tensors='pt').to(self.device)
            # ori_tar = self.tokenizer(' '+self.dataset.ori_tar, return_tensors='pt').to(self.device)

            edit_model.init_model(self.model)
            #breakpoint()
            neuron_index = self.neuron_ids[0]

            #delta_w = self.build_delta(neuron_index)


            edit_model.init_delta(neuron_index)
            #breakpoint()
            edit_model.build_hook()
            opt = self.optimizer([v for vv in edit_model.delta_w.values() for v in vv.values()], lr=self.lr)

            tb_step = 0
            for epoch in tqdm(range(self.epochs), desc='epochs'):
                
                for data in dl:
                    opt.zero_grad()
                    enc_inp, tar, kl_init_tar = data 
                    enc_inp = dict2device(enc_inp, self.device)
                    tar = tar.to(self.device)
                    kl_init_tar = kl_init_tar.to(self.device)

                    out = edit_model(enc_inp)
                    logits = out.logits

                    probs = torch.softmax(logits[:, -1, :], dim=-1)



                    loss_gen = self.criterion(logits[:, -1, :], tar)

                    loss_weight = self.theta*max([torch.max(torch.abs(v)) for vv in edit_model.delta_w.values() for v in vv.values()])
                    
                    kl_log_probs = nn.functional.log_softmax(logits, dim=-1)
                    kl_loss = self.alpha*nn.functional.kl_div(
                        kl_init_tar, kl_log_probs, log_target=True, reduction='batchmean'
                    )
                    loss = kl_loss + loss_weight if torch.isnan(loss_gen) else kl_loss + loss_weight + loss_gen

                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        for vv in edit_model.delta_w.values():
                            for v in vv.values():
                                v[...] = torch.clamp(v, min=torch.zeros_like(v)-self.theta, max=torch.zeros_like(v)+self.theta)
                    if isdebug:
                        kl_writer.add_scalar(f'''{data_idx}_{self.edit_pos}_loss''', kl_loss, tb_step)
                        if not torch.isnan(loss_gen):
                            genl_writer.add_scalar(f'''{data_idx}_{self.edit_pos}_loss''', loss_gen, tb_step)
                        weil_writer.add_scalar(f'''{data_idx}_{self.edit_pos}_loss''', loss_weight, tb_step)
                        suml_writer.add_scalar(f'''{data_idx}_{self.edit_pos}_loss''', loss, tb_step)
                    tb_step += 1
                    #breakpoint()




                # #breakpoint()
                # loss.backward()
                # opt.step()


            # ori_writer.close()
            # tar_writer.close()
            if isdebug:
                kl_writer.close()
                genl_writer.close()
                weil_writer.close()
                suml_writer.close()

            edit_model.remove_hook()
            edit_model.update_model()
            #breakpoint()
            eval(edit_model.model, self.tokenizer, self.dataset, data_idx, edit_model.num_delta, gen_test_vars, \
                  output_dir=eval_dir, isabs=self.isabs, epoch=self.epochs)




    def kl_ft_train(self, data_idx, eval_dir, gen_test_vars=[None, None], isdebug=False, ismap=False, noedit=False):
            ### TODO: delta_w .shape
            #self.model.eval()
            #self.model.train()
            cur_loss = 0
            #breakpoint()

            #tensorboard
            edit_model = Edited_Model(self.hidden_size, self.device, copy=False, edit_type=self.edit_type, edit_pos=self.edit_pos)

            if noedit:
                eval(self.model, self.tokenizer, self.dataset, data_idx, 0, gen_test_vars, \
                 output_dir=eval_dir, isabs=self.isabs, epoch=self.epochs)
                return
            # tar_writer = SummaryWriter('./kl_runs/tar')
            # ori_writer = SummaryWriter('./kl_runs/ori')
            if isdebug:
                kl_writer = SummaryWriter('./kl_runs/kl_loss')
                genl_writer = SummaryWriter('./kl_runs/genloss')
                weil_writer = SummaryWriter('./kl_runs/weiloss')
                suml_writer = SummaryWriter('./kl_runs/sumloss')

            def my_collate_fn(data):
                enc_inp, tar, kl_init_tar = [], [], []
                for batch in data:
                    enc_inp.append(batch[0])
                    tar.append(batch[1])
                    kl_init_tar.append(batch[2])
                    #indices.append(batch[3])
                #breakpoint()
                enc_inp = dict_cat(enc_inp)
                #print(tar)
                tar = torch.cat(tar, dim=0)
                kl_init_tar = torch.stack(kl_init_tar, dim=0)
                #indices = torch.stack(indices, dim=0)
                #breakpoint()
                return enc_inp, tar, kl_init_tar
                
            dl = DataLoader(self.dataset, batch_size=20, shuffle=True, collate_fn=my_collate_fn)

            # new_tar = self.tokenizer(' '+self.dataset.tar, return_tensors='pt').to(self.device)
            # ori_tar = self.tokenizer(' '+self.dataset.ori_tar, return_tensors='pt').to(self.device)

            edit_model.init_model(self.model)
            #breakpoint()
            neuron_index = self.neuron_ids[0]

            #delta_w = self.build_delta(neuron_index)


            edit_model.init_delta(neuron_index)
            #breakpoint()
            edit_model.build_hook()
            opt = self.optimizer([v for vv in edit_model.delta_w.values() for v in vv.values()], lr=self.lr)

            tb_step = 0
            gen_loss = []
            epoch_loss = []
            for epoch in tqdm(range(self.epochs), desc='epochs'):
                
                for data in dl:
                    opt.zero_grad()
                    enc_inp, tar, kl_init_tar = data 
                    enc_inp = dict2device(enc_inp, self.device)
                    tar = tar.to(self.device)
                    kl_init_tar = kl_init_tar.to(self.device)

                    out = edit_model(enc_inp)
                    logits = out.logits

                    probs = torch.softmax(logits[:, -1, :], dim=-1)



                    loss_gen = self.criterion(logits[:, -1, :], tar)

                    loss_weight = self.theta*max([torch.max(torch.abs(v)) for vv in edit_model.delta_w.values() for v in vv.values()])
                    
                    kl_log_probs = nn.functional.log_softmax(logits, dim=-1)
                    kl_loss = self.alpha*nn.functional.kl_div(
                        kl_init_tar, kl_log_probs, log_target=True, reduction='batchmean'
                    )
                    loss = kl_loss + loss_weight if torch.isnan(loss_gen) else kl_loss + loss_weight + loss_gen
                    if not torch.isnan(loss_gen):
                        gen_loss.append(loss_gen.item())
                        epoch_loss.append((epoch, tb_step))

                    

                    loss.backward()
                    opt.step()


                        

                    ### test


                    if isdebug:
                        kl_writer.add_scalar(f'''{data_idx}_{self.edit_pos}_loss''', kl_loss, tb_step)
                        if not torch.isnan(loss_gen):
                            genl_writer.add_scalar(f'''{data_idx}_{self.edit_pos}_loss''', loss_gen, tb_step)
                        weil_writer.add_scalar(f'''{data_idx}_{self.edit_pos}_loss''', loss_weight, tb_step)
                        suml_writer.add_scalar(f'''{data_idx}_{self.edit_pos}_loss''', loss, tb_step)
                    tb_step += 1
                    #breakpoint()
                
                if len(epoch_loss) > 0 and gen_loss[-1] < 1e-2:
                    break



                # #breakpoint()
                # loss.backward()
                # opt.step()


            # ori_writer.close()
            # tar_writer.close()

            # build gen loss 
            if ismap:
                '''
                print('------------building map-------------')
                with open(f'./result/mid_result/images/{data_idx}r5.json', 'w') as f:
                    json.dump(gen_loss, f)
                '''
                with open(f'''./result/mid_result/images/epochs/{self.dataset.data[data_idx]['case_id']}_3.json''', 'w') as f:
                    json.dump(epoch_loss, f)

            if isdebug:
                kl_writer.close()
                genl_writer.close()
                weil_writer.close()
                suml_writer.close()

            edit_model.remove_hook()
            edit_model.update_model()
            #breakpoint()
            eval(edit_model.model, self.tokenizer, self.dataset, data_idx, edit_model.num_delta, gen_test_vars, \
                 output_dir=eval_dir, isabs=self.isabs, epoch=self.epochs)
            edit_model.rollback()


def eval(
        edit_model,
        tokenizer,
        dataset,
        ii,
        num_delta,
        gen_test_vars,
        model_name='gpt2-xl',
        output_dir='./result/{model_name}/results/top10/',
        isabs=False,
        epoch=500,
        testknown=True,
):
    edit_model.eval()
    dt = dataset.data[ii]

    #gen_id = dataset.dataidx['gen_id'][ii]
    if testknown:
        para_id = dataset.dataidx['para_id'][ii]
        loc_id = dataset.dataidx['loc_id'][ii]
        #dt['generation_prompts'] = [dt['generation_prompts'][i] for i in gen_id] if gen_id is not [] else []
        dt['paraphrase_prompts'] = [dt['paraphrase_prompts'][i] for i in para_id] if para_id is not [] else []
        dt['neighborhood_prompts'] = [dt['neighborhood_prompts'][i] for i in loc_id] if loc_id is not [] else []
    metrics = {
        'case_id': dt['case_id'],
        'prompt': dt['prompt'],
        'paraphrase': dt['paraphrase_prompts'],
        'neighbor': dt['neighborhood_prompts'],
        'neurons': num_delta,
        'abs': isabs,
        'epoch': epoch,
        'post': compute_rewrite_quality_counterfact(
            edit_model,
            tokenizer,
            dt,
            *(
                gen_test_vars
                if dt["case_id"] % 1 == 0
                else [None, None]
            ),  # Only test generation every generation_test_interval cases
        )
    }
    out_dir = os.path.join(output_dir.format(model_name=model_name), f'''result_{dt['case_id']}.json''')
    with open(out_dir, 'w') as f:
        json.dump(metrics, f, indent=2)
















module_format_len = len('transformers.h.0')


class Edited_Model(torch.nn.Module):
    def __init__(self, hidden_size, device, copy=False, edit_type='neuron', edit_pos='c_proj',  **kwargs):
        '''
        on class for one data
        neuron_id := a dict respond to dataset, each sub dict is (key=layername, value=list of neuron_id)
        '''
        super().__init__(**kwargs)
        self.copy = copy
        # if self.copy:
        #     self.model = deepcopy(model)
        # else:
        #     self.model = model
        
        self.device = device
        self.hidden_size = hidden_size
        self.num_delta = 0
        self.edit_type = edit_type
        self.edit_pos = edit_pos
        # self.neuron_ids = neuron_ids
        # self.delta_w = self.build_delta(neuron_ids)
        # self.build_hook()
    def init_model(self, model):
        if self.copy:
            self.model = deepcopy(model)
        else:
            self.model = model
        self.model.to(self.device)
        self.model.eval()

    def init_delta(self, neuron_id):
        '''
        input neuron_id a dict
        build delta weight
        '''
        self.neuron_ids = neuron_id
        self.delta_w = {}
        for k, v in neuron_id.items():
            self.delta_w[k] = {}
            if self.edit_pos == 'all':
                self.delta_w[k]['c_fc'] = nn.Parameter(torch.zeros(self.hidden_size, len(v), dtype=torch.float).to(self.device))
                self.delta_w[k]['c_proj'] = nn.Parameter(torch.zeros(len(v), self.hidden_size, dtype=torch.float).to(self.device))
            elif self.edit_pos == 'c_fc':
                self.delta_w[k]['c_fc'] = nn.Parameter(torch.zeros(self.hidden_size, len(v), dtype=torch.float).to(self.device))
            elif self.edit_type == 'neuron':
                self.delta_w[k]['c_proj'] = nn.Parameter(torch.zeros(len(v), self.hidden_size, dtype=torch.float).to(self.device))
            elif self.edit_type == 'hidsize':
                self.delta_w[k]['c_proj'] = nn.Parameter(torch.zeros(self.hidden_size, len(v), dtype=torch.float).to(self.device))
            self.num_delta += len(v)
            #delta_weights[k].requires_grad=True

    def FFN_hook(self, name):
            

        if 'c_fc' in name:
            def hook_fc(module, inp, out):
                k = name[:-5]  
                n_ids = self.neuron_ids[k]
                h = inp[0] if isinstance(inp, tuple) else inp
                delta_out = torch.bmm(h, self.delta_w[k]['c_fc'].unsqueeze(0).repeat(h.shape[0],1,1))
                out[:,:,n_ids] += delta_out
                #breakpoint() 
                return out
            return hook_fc
        
        def hook_n(module, inp, out):
            k = name[:-7]
            #breakpoint()
            n_ids = self.neuron_ids[k]
            p_inp = inp[0] # 切片维度 默认only one data
            delta_out = torch.bmm(p_inp[:,:,n_ids], self.delta_w[k]['c_proj'].unsqueeze(0).repeat(p_inp.shape[0],1,1))
            #breakpoint()
            out[:,:,:] += delta_out
            #breakpoint()
            # p_inp = p_inp.permute(-1, 0, 1).unsqueeze(-1) # from (batch, seqlen, kh) ==> (kh, batch, seqlen, 1)
            # #breakpoint()
            # for ii in range(p_inp.shape[0]):
            #     out += p_inp[ii]* self.delta_w[k][ii]
            return out

        def hook_h(module, inp, out):
            k = name[:-7]
            n_ids = self.neuron_ids[k]
            p_inp = inp[0]
            # delta_out = torch.mm(p_inp, self.delta_w[k]).unsqueeze(0)
            #breakpoint()
            delta_out = torch.bmm(p_inp, self.delta_w[k]['c_proj'].unsqueeze(0).repeat(p_inp.shape[0], 1, 1))
            #breakpoint()
            out[:, :, n_ids] += delta_out
            return out

        return hook_h if self.edit_type == 'hidsize' else hook_n
    
### TODO: all hook and build hook
    def build_hook(self):
        self.hook_list = []
        #breakpoint()
        if self.edit_pos == 'all':
            for kname in self.neuron_ids.keys():
                fcnmd = kname + '.c_fc'
                pronmd = kname + '.c_proj'
                self.hook_list.append(get_module(self.model, fcnmd).register_forward_hook(self.FFN_hook(fcnmd)))
                self.hook_list.append(get_module(self.model, pronmd).register_forward_hook(self.FFN_hook(pronmd)))

        elif self.edit_pos == 'c_proj':
            for kname in self.neuron_ids.keys():
                nmd = kname+'.c_proj'
                #breakpoint()
                self.hook_list.append(
                    get_module(self.model, nmd).register_forward_hook(self.FFN_hook(nmd))
                )
            
        else:
            for kname in self.neuron_ids.keys():
                nmd = kname+'.c_fc'
                #breakpoint()
                self.hook_list.append(
                    get_module(self.model, nmd).register_forward_hook(self.FFN_hook(nmd))
                )
    
    def remove_hook(self):
        for hook in self.hook_list:
            hook.remove()
    
    def update_model(self):
        if self.edit_pos == 'all':
            for k, v in self.neuron_ids.items():
                self.model.state_dict()[k + '.c_fc.weight'][:, v] += self.delta_w[k]['c_fc']
                self.model.state_dict()[k + '.c_proj.weight'][v, :] += self.delta_w[k]['c_proj']
        elif self.edit_pos == 'c_fc':
            for k, v in self.neuron_ids.items():
                self.model.state_dict()[k + '.c_fc.weight'][:, v] += self.delta_w[k]['c_fc']
        elif self.edit_type == 'neuron':
            for k, v in self.neuron_ids.items():
                self.model.state_dict()[k + '.c_proj.weight'][v, :] += self.delta_w[k]['c_proj']
        
        elif self.edit_type == 'hidsize':
            for k, v in self.neuron_ids.items():
                self.model.state_dict()[k + '.c_proj.weight'][:, v] += self.delta_w[k]['c_proj']

    def rollback(self):
        if self.edit_pos == 'all':
            for k, v in self.neuron_ids.items():
                self.model.state_dict()[k + '.c_fc.weight'][:, v] -= self.delta_w[k]['c_fc']
                self.model.state_dict()[k + '.c_proj.weight'][v, :] -= self.delta_w[k]['c_proj']
        elif self.edit_pos == 'c_fc':
            for k, v in self.neuron_ids.items():
                self.model.state_dict()[k + '.c_fc.weight'][:, v] -= self.delta_w[k]['c_fc']
        elif self.edit_type == 'neuron':
            for k, v in self.neuron_ids.items():
                self.model.state_dict()[k + '.c_proj.weight'][v, :] -= self.delta_w[k]['c_proj']
        
        elif self.edit_type == 'hidsize':
            for k, v in self.neuron_ids.items():
                self.model.state_dict()[k + '.c_proj.weight'][:, v] -= self.delta_w[k]['c_proj']

        

    def forward(self, inp):
        # build hook
        return self.model(**inp)
        


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)



        
