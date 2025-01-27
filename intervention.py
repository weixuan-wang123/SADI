import os, re, json
import torch
import torch.nn.functional as F
import json
import numpy as np
from numpy import *
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset
from baukit import Trace, TraceDict

from load_dataset import load_copa
from steering import select_heads_acts_actups,steer_heads,steer_neurons,steer_hiddens,evalone

torch.set_grad_enabled(False)

model_name = "meta-llama/Llama-3.1-8B"
mt = ModelAndTokenizer(
    model_name,
)
LAYERS = mt.model.config.n_layer




def baseline(test_data):
    out_f = open("./results/baseline_res.txt",'w')
    EMs = []
    for idx in range(len(test_data)):
        q,a = test_data[idx][0],test_data[idx][1]   
        target_ids = mt.tokenizer(a, return_tensors='pt')['input_ids'].to('cuda')
        inp = mt.tokenizer(f'{q} {a}', return_tensors='pt').to('cuda')
        outputs = mt.model(**inp)
        logits = outputs.logits
        ans = torch.argmax(logits, dim=-1)[:, -target_ids.size(1):-1].squeeze()
        ans_idss = ans.detach().cpu().numpy().tolist()

        textual_ans = mt.tokenizer.decode(ans_idss, skip_special_tokens=True)
        EM = evalone(textual_ans,test_data[idx][1])
        EMs.append(EM)
    EMs = mean(EMs,axis=0)
    print("baseline",EMs)
    out_f.write(f"baseline\t{EMs}\n")
    return EMs



def load_dataset(file):
    train_data,test_data = load_copa()
    return train_data,test_data

file = 'copa'
train_data,test_data = load_dataset(file)
res_base = baseline(test_data)
max_heads,min_heads,max_acts,min_acts,max_hiddens,min_hiddens = select_heads_acts_actups(mt.model,mt.tokenizer,train_data)
res_heads = steer_heads(mt.model,mt.tokenizer,test_data,max_heads,min_heads)
res_neurons = steer_neurons(mt.model,mt.tokenizer,test_data,max_acts,min_acts)
res_hiddens = steer_hiddens(mt.model,mt.tokenizer,test_data,max_hiddens,min_hiddens)

