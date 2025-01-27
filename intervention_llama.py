import os, re, json
import torch
import json
import numpy as np
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from baukit import Trace, TraceDict
import torch.nn.functional as F
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig

torch.set_grad_enabled(False)

model_name = "Mathoctopus/Parallel_7B"

mt = ModelAndTokenizer(
    model_name,
)
LAYERS = mt.model.config.num_hidden_layers


langs = ["en", "de","es", "fr", "ja", "ru", "sw", "th", "zh"]
def load_data_mgsm():
    langs = ["en", "de","es", "fr", "ja", "ru", "sw", "th", "zh"]
    names = {'en':'English','de':'German','es':'Spanish','fr':'French','ja':'Japanese','ru':'Russian','sw':'Swahili','th':'Thai','zh':'Chinese'}
    question_all, answer_all = [],[]
    for lang in langs:
        name = names[lang]
        questions,answers = [],[]
        with open(f"./data/MGSM/mgsm_{lang}.tsv") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip().split('\t')
            question, answer = line[0],line[1]
            prompt_no_input = (
                  "Below is an instruction that describes a task. "
                    f"Write a response that appropriately completes the request in {name}. Please answer in {name}.\n\n"
                    "### Instruction:\n{query}\n\n### Response:"
                )
            prompt = prompt_no_input.format(query=question)
            questions.append(prompt)
            answers.append(answer)
            
        question_all.append(questions)
        answer_all.append(answers)
        
    print(len(question_all),len(question_all[0]))
    print(question_all[-1][0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[-1][0])      
    return question_all,answer_all

question_all,answer_all = load_data_mgsm()


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  
    res = re.findall(r"(\d+(\.\d+)?)", text) 
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0
from numpy import *
punc = ['.</s>','.\n','.',';','!',',','?','\n','</s>','<pad>']

def eval(ans_res,reference,lg):
    res = []
    for i in range(len(ans_res)):
        ans = ans_res[i].strip().lower()
        for p in punc:
            ans = ans.replace(p, "")
        extract_true_num = extract_last_num(reference[lg][i].strip().lower())
        extract_pred_num = extract_last_num(ans)
        if abs(extract_true_num - extract_pred_num) < 1e-3:
            res.append(1)
        else:
            res.append(0)
    res = mean(res,axis=0)
    return res

def get_out(model, prompt, device,index): 

    model.eval()
    ATT = [f"model.layers.{i}.self_attn" for i in range(LAYERS)]
    ATT_q= [f"model.layers.{i}.self_attn.q_proj" for i in range(LAYERS)]
    ATT_k = [f"model.layers.{i}.self_attn.k_proj" for i in range(LAYERS)]
    ATT_v = [f"model.layers.{i}.self_attn.v_proj" for i in range(LAYERS)]
    ATT_o = [f"model.layers.{i}.self_attn.o_proj" for i in range(LAYERS)]
    MLP = [f"model.layers.{i}.mlp" for i in range(LAYERS)]
    MLP_gate = [f"model.layers.{i}.mlp.gate_proj" for i in range(LAYERS)]
    MLP_up = [f"model.layers.{i}.mlp.up_proj" for i in range(LAYERS)]
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(LAYERS)]
    MLP_down = [f"model.layers.{i}.mlp.down_proj" for i in range(LAYERS)]
    ATT_post = [f"model.layers.{i}.post_attention_layernorm" for i in range(LAYERS)]
    with torch.no_grad():
        with TraceDict(model, ATT+ATT_q+ATT_k+ATT_v+ATT_o+MLP+MLP_gate+MLP_act+MLP_up+MLP_down+ATT_post) as ret:
            output = model(prompt, output_hidden_states = True,output_attentions=True)
       
        mlp_act = [ret[mlp_act].output[0][index].detach().cpu().numpy() for mlp_act in MLP_act]
        ATT_value = [ret[att_value].output[0][index].detach().cpu().numpy() for att_value in ATT]
        MLP_value = [ret[mlp_value].output[0][index].detach().cpu().numpy() for mlp_value in MLP]
        MLP_up_value = [ret[mlp_up].output[0][index].detach().cpu().numpy() for mlp_up in MLP_up]
        MLP_down_value = [ret[mlp_down].output[0][index].detach().cpu().numpy() for mlp_down in MLP_down]
        ATT_post_value = [ret[att_post].output[0][index].detach().cpu().numpy() for att_post in ATT_post]

        return mlp_act,ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value
def get_out_mean(model, prompt, device,index): 

    model.eval()
    ATT = [f"model.layers.{i}.self_attn" for i in range(LAYERS)]
    ATT_q= [f"model.layers.{i}.self_attn.q_proj" for i in range(LAYERS)]
    ATT_k = [f"model.layers.{i}.self_attn.k_proj" for i in range(LAYERS)]
    ATT_v = [f"model.layers.{i}.self_attn.v_proj" for i in range(LAYERS)]
    ATT_o = [f"model.layers.{i}.self_attn.o_proj" for i in range(LAYERS)]
    MLP = [f"model.layers.{i}.mlp" for i in range(LAYERS)]
    MLP_gate = [f"model.layers.{i}.mlp.gate_proj" for i in range(LAYERS)]
    MLP_up = [f"model.layers.{i}.mlp.up_proj" for i in range(LAYERS)]
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(LAYERS)]
    MLP_down = [f"model.layers.{i}.mlp.down_proj" for i in range(LAYERS)]
    ATT_post = [f"model.layers.{i}.post_attention_layernorm" for i in range(LAYERS)]
    
    
    with torch.no_grad():
        with TraceDict(model, ATT+ATT_q+ATT_k+ATT_v+ATT_o+MLP+MLP_gate+MLP_act+MLP_up+MLP_down+ATT_post) as ret:
            output = model(prompt, output_hidden_states = True,output_attentions=True)
       
        mlp_act = [np.mean(ret[mlp_act].output[0].detach().cpu().numpy(),axis=0) for mlp_act in MLP_act]
        ATT_value = [np.mean(ret[att_value].output[0].detach().cpu().numpy(),axis=0) for att_value in ATT]
        MLP_value = [np.mean(ret[mlp_value].output[0].detach().cpu().numpy(),axis=0) for mlp_value in MLP]
        MLP_up_value = [np.mean(ret[mlp_up].output[0].detach().cpu().numpy(),axis=0) for mlp_up in MLP_up]
        MLP_down_value = [np.mean(ret[mlp_down].output[0].detach().cpu().numpy(),axis=0) for mlp_down in MLP_down]
        ATT_post_value = [np.mean(ret[att_post].output[0].detach().cpu().numpy(),axis=0) for att_post in ATT_post]
        emb_value = EMB[0]

        return mlp_act,ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value
    


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    new_mlp_up,
    question,
    answer,
):
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer not in patch_spec:
            return x
        h = untuple(x)
        idxxx = h.shape[1] - 1
        if idxxx == patch_spec[layer][0]:
            for t in patch_spec[layer]:
                ly = int(layer.split('model.layers.')[-1].split('.mlp')[0])
                h[0][t] = torch.tensor(new_mlp_up)[ly]
        return h

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        
        input_ids_w_attnmask = mt.tokenizer(question,padding=True,return_tensors="pt",).to('cuda')
        output_ids = model.generate(input_ids=input_ids_w_attnmask.input_ids,attention_mask=input_ids_w_attnmask.attention_mask,
                    generation_config=GenerationConfig(
                        max_new_tokens=600,
                        do_sample=False,
                        temperature=0.0,  
                    ),
                ).tolist()
        real_output_ids = [output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)]
        r = mt.tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0]
   
        return r.strip()




for lang_id in range(1,len(langs)):
    lang = langs[lang_id]
    print(lang)
    train_mlp_acts_en,train_mlp_acts_zh,valid_mlp_acts_en,valid_mlp_acts_zh = [],[],[],[]
  
    with open(f"./data/ncwm/en-{lang}/train.en", encoding="utf-8") as f:
        en_data = f.readlines()
    with open(f"./data/ncwm/en-{lang}/train.{lang}", encoding="utf-8") as g:
        zh_data = g.readlines()
    ind = 0    
    while ind < 500 and ind < len(en_data):
        # sent = zh_data[ind]
        if ind % 2 == 0: 
            sent = zh_data[ind]
        else:
            sent = en_data[ind] + '\n'+ zh_data[ind]
            
        encodings = mt.tokenizer(sent, return_tensors='pt')
        input_ids = encodings['input_ids'].to('cuda')
        if len(input_ids[0]) > 500:
            ind += 1
            continue
        attention_mask = encodings['attention_mask'].to('cuda')

        mlp_act, att_val,mlp_val,mlp_up,mlp_down,att_post = get_out_mean(mt.model,input_ids,mt.model.device,-1)
        mlp_val = np.array(mlp_val)

        train_mlp_acts_zh.append(mlp_val[:,:])
        
        if ind % 2 == 0: 
            sent = en_data[ind]
        else:
            sent = en_data[ind] + '\n'+ en_data[ind]
        encodings = mt.tokenizer(sent, return_tensors='pt')
        input_ids = encodings['input_ids'].to('cuda')
        attention_mask = encodings['attention_mask'].to('cuda')

        mlp_act, att_val,mlp_val,mlp_up,mlp_down,att_post = get_out_mean(mt.model,input_ids,mt.model.device,-1)
        mlp_val = np.array(mlp_val)

        train_mlp_acts_en.append(mlp_val[:,:])
        ind += 1



    train_mlp_acts_zh, train_mlp_acts_en = np.array(train_mlp_acts_zh),np.array(train_mlp_acts_en)    
    transformation_matrixs = []    
    for i in range(LAYERS):
        transformation_matrix = np.linalg.lstsq(train_mlp_acts_zh[:,i,:], train_mlp_acts_en[:,i,:], rcond=None)[0]    
        transformation_matrixs.append(transformation_matrix)
        
    counts = len(question_all[lang_id])
    sigmas = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for sigma in sigmas:
        ans_res = []
        for ind in range(counts):
            question = question_all[lang_id][ind]
            answer = answer_all[lang_id][ind]

            inputs = mt.tokenizer.encode(question, return_tensors="pt").to("cuda")
            mlp_act, att_val,mlp_val,mlp_up,mlp_down,att_post = get_out(mt.model,inputs,mt.model.device,-1)
            mlp_val = np.array(mlp_val)


            dires = []
            for i in range(LAYERS):
                dire = -np.dot([mlp_val[i]], transformation_matrixs[i])[0]
                dires.append(dire)
            dires = np.array(dires)
            mlp_dires = mlp_val + sigma * dires
            


            inp = make_inputs(mt.tokenizer, [question])
            ntoks = inp["input_ids"].shape[1]       
            question_answer = f'{question} {answer}'
            inp_question_answer = make_inputs(mt.tokenizer, [question_answer])


            r = trace_with_patch(
                    mt.model,
                    inp_question_answer,
                    [(ntoks-1, layername(mt.model, layer,'mlp')) for layer in range(LAYERS)],
                    new_mlp_up = mlp_dires,
                    question=question,
                    answer=answer)
            ans_res.append(r)


        EM = eval(ans_res,answer_all,lang_id)
        print(langs[lang_id],sigma,EM)

        