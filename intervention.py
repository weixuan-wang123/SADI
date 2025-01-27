import os, re, json
import torch
import torch.nn.functional as F
import json
import numpy as np

from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
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
from baukit import Trace, TraceDict

torch.set_grad_enabled(False)

model_name = "bigscience/bloomz-7b1-mt"
mt = ModelAndTokenizer(
    model_name,
)
LAYERS = mt.model.config.n_layer


from numpy import *

def obtain_f1_and_em(a,b):
    global tokenizer
    
    a_words = mt.tokenizer.encode(a, add_special_tokens=False)
    b_words = mt.tokenizer.encode(b, add_special_tokens=False)
    if len(a_words) == 0 and len(b_words) == 0:
        return 1.0, 1
    if len(a_words) == 0 or len(b_words) == 0:
        return 0.0, 0

    em = 1 if a == b else 0
    k = len(a_words) * len(b_words)

    intersecting_words = []
    for word in a_words.copy():
        if word in b_words:
            a_words.remove(word)
            b_words.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em
from numpy import *
punc = ['.</s>','.\n','.',';','!',',','?','\n','</s>','<pad>']

def eval(ans_res,reference,lg):
    F1, EM = [],[]
    for i in range(len(ans_res)):
        ans = ans_res[i].strip().lower()
        for p in punc:
            ans = ans.replace(p, "")
        f1,em = obtain_f1_and_em(ans,reference[lg][i].strip().lower())
        F1.append(f1)
        EM.append(em)
    F1,EM = mean(F1,axis=0),mean(EM,axis=0)
    return EM,F1



def get_out(model, prompt, device,index): 

    model.eval()
    ATT = [f"transformer.h.{i}.self_attention.att_out" for i in range(LAYERS)]
    QKV = [f"transformer.h.{i}.self_attention.query_key_value" for i in range(LAYERS)]
    MLP = [f"transformer.h.{i}.mlp" for i in range(LAYERS)]
    MLP_up = [f"transformer.h.{i}.mlp.dense_h_to_4h" for i in range(LAYERS)]
    MLP_act = [f"transformer.h.{i}.mlp.gelu_impl" for i in range(LAYERS)]
    MLP_down = [f"transformer.h.{i}.mlp.dense_4h_to_h" for i in range(LAYERS)]
    ATT_post = [f"transformer.h.{i}.post_attention_layernorm" for i in range(LAYERS)]
    EMB = [f"transformer.word_embeddings_layernorm"]
    HEADS = [f"transformer.h.{i}.self_attention.head_out" for i in range(LAYERS)]
    
    with torch.no_grad():
        with TraceDict(model, ATT+MLP+MLP_up+MLP_down+ATT_post+EMB+QKV+HEADS+MLP_act) as ret:
            output = model(prompt, output_hidden_states = True,output_attentions=True)

        ATT_value = [ret[att_value].output[0,index,:].detach().cpu().numpy() for att_value in ATT]
        MLP_value = [ret[mlp_value].output[0,index,:].detach().cpu().numpy() for mlp_value in MLP]
        MLP_up_value = [ret[mlp_up].output[0,index,:].detach().cpu().numpy() for mlp_up in MLP_up]
        MLP_act_value = [ret[mlp_act].output[0,index,:].detach().cpu().numpy() for mlp_act in MLP_act] 
        MLP_down_value = [ret[mlp_down].output[0,index,:].detach().cpu().numpy() for mlp_down in MLP_down]
        ATT_post_value = [ret[att_post].output[0,index,:].detach().cpu().numpy() for att_post in ATT_post]
        HEAD_value = [ret[head_value].output[:,index,:].detach().cpu().numpy() for head_value in HEADS] 
        

        return ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value,HEAD_value,MLP_act_value
    
def get_out_mean(model, prompt, device,index): 

    model.eval()
    ATT = [f"transformer.h.{i}.self_attention.att_out" for i in range(LAYERS)]
    QKV = [f"transformer.h.{i}.self_attention.query_key_value" for i in range(LAYERS)]
    MLP = [f"transformer.h.{i}.mlp" for i in range(LAYERS)]
    MLP_up = [f"transformer.h.{i}.mlp.dense_h_to_4h" for i in range(LAYERS)]
    MLP_act = [f"transformer.h.{i}.mlp.gelu_impl" for i in range(LAYERS)]
    MLP_down = [f"transformer.h.{i}.mlp.dense_4h_to_h" for i in range(LAYERS)]
    ATT_post = [f"transformer.h.{i}.post_attention_layernorm" for i in range(LAYERS)]
    EMB = [f"transformer.word_embeddings_layernorm"]
    HEADS = [f"transformer.h.{i}.self_attention.head_out" for i in range(LAYERS)]
    
    with torch.no_grad():
        with TraceDict(model, ATT+MLP+MLP_up+MLP_down+ATT_post+EMB+QKV+HEADS+MLP_act) as ret:
            output = model(prompt, output_hidden_states = True,output_attentions=True)

        ATT_value = [np.mean(ret[att_value].output[0].detach().cpu().numpy(),axis=0) for att_value in ATT]
        MLP_value = [np.mean(ret[mlp_value].output[0].detach().cpu().numpy(),axis=0) for mlp_value in MLP]
        MLP_up_value = [np.mean(ret[mlp_up].output[0].detach().cpu().numpy(),axis=0) for mlp_up in MLP_up]
        MLP_act_value = [np.mean(ret[mlp_act].output[0].detach().cpu().numpy(),axis=0) for mlp_act in MLP_act] 
        MLP_down_value = [np.mean(ret[mlp_down].output[0].detach().cpu().numpy(),axis=0) for mlp_down in MLP_down]
        ATT_post_value = [np.mean(ret[att_post].output[0].detach().cpu().numpy(),axis=0) for att_post in ATT_post]
        HEAD_value = [np.mean(ret[head_value].output.detach().cpu().numpy(),axis=1) for head_value in HEADS] 
        

        return ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value,HEAD_value,MLP_act_value


def trace_with_patch(
    model,  
    inp,  #
    states_to_patch,  
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
        for t in patch_spec[layer]:
            ly = int(layer.split('transformer.h.')[-1].split('.mlp')[0])
            h[0][t] = torch.tensor(new_mlp_up)[ly]
        return h

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:

        target_ids = mt.tokenizer(' '+answer + '</s>', return_tensors='pt')['input_ids'].to('cuda')

        outputs = model(**inp)
        logits = outputs.logits
        ans = torch.argmax(logits, dim=-1)[:, -target_ids.size(1):].squeeze()
        ans_idss = ans.detach().cpu().numpy().tolist()

        textual_ans = mt.tokenizer.decode(ans_idss, skip_special_tokens=True)
        return textual_ans.strip()


import os
import json
langs = ['en','et','id','it','sw','ta','th','tr','vi','zh']
questions_en = []
with open(f"./data/xcopa/test.en.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    line = json.loads(line)
    ques = line['question']
    questions_en.append(ques)
def load_data_xcopa():
    question_all,answer_all = [],[]
    for lang in langs:
        questions,answers = [],[]
        with open(f"./data/xcopa/test.{lang}.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
        for ind in range(len(lines)):
            line = lines[ind]
            line = json.loads(line)
            premise,choice1,choice2,label = line['premise'],line['choice1'],line['choice2'],line['label']
            question = questions_en[ind]
            prompt_question = f'Here is a premise: "{premise}". A: "{choice1}" B: "{choice2}" What is the {question}? "A" or "B"?'

            if int(label) == 0:
                prompt_answer = 'A'
            elif int(label) == 1:
                prompt_answer = 'B'
            questions.append(prompt_question)
            answers.append(prompt_answer)
        question_all.append(questions)
        answer_all.append(answers)
        
    print(len(question_all),len(question_all[0]))
    print(question_all[0][0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[0][0])
    return question_all,answer_all

question_all,answer_all = load_data_xcopa()




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
        if ind % 2 == 0: 
            sent = zh_data[ind]
        else:
            sent = zh_data[ind] + ' ' + en_data[ind]
        encodings = mt.tokenizer(sent, return_tensors='pt')
        input_ids = encodings['input_ids'].to('cuda')
        if len(input_ids[0]) > 500:
            ind += 1
            continue
        attention_mask = encodings['attention_mask'].to('cuda')

        att_val,mlp_val,mlp_up,mlp_down,att_post = get_out_mean(mt.model,input_ids,mt.model.device,-1)
        mlp_val = np.array(mlp_val)

        train_mlp_acts_zh.append(mlp_val[:,:])
        
        if ind % 2 == 0: 
            sent = en_data[ind]
        else:
            sent = en_data[ind] + ' ' + en_data[ind]
        encodings = mt.tokenizer(sent, return_tensors='pt')
        input_ids = encodings['input_ids'].to('cuda')
        attention_mask = encodings['attention_mask'].to('cuda')

        att_val,mlp_val,mlp_up,mlp_down,att_post = get_out_mean(mt.model,input_ids,mt.model.device,-1)
        mlp_val = np.array(mlp_val)

        train_mlp_acts_en.append(mlp_val[:,:])
        ind += 1




    train_mlp_acts_zh, train_mlp_acts_en = np.array(train_mlp_acts_zh),np.array(train_mlp_acts_en)    
    transformation_matrixs = []    
    for i in range(LAYERS):
        transformation_matrix = np.linalg.lstsq(train_mlp_acts_zh[:,i,:], train_mlp_acts_en[:,i,:], rcond=None)[0]    
        transformation_matrixs.append(transformation_matrix)

    counts = len(question_all[lang_id])
    for sigma in [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        ans_res = []
        for ind in range(counts):
            question = question_all[lang_id][ind]
            answer = answer_all[lang_id][ind]

            inputs = mt.tokenizer.encode(question, return_tensors="pt").to("cuda")
            att_val,mlp_val,mlp_up,mlp_down,att_post = get_out(mt.model,inputs,mt.model.device,-1)
            mlp_val = np.array(mlp_val)

            dires = []
            for i in range(LAYERS):
                dire = np.dot([mlp_val[i]], transformation_matrixs[i])[0]
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


        EM,F1 = eval(ans_res,answer_all,lang_id)
        print(langs[lang_id],sigma,EM)
