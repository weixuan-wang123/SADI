from torch.nn import functional as F
import numpy as np
import pandas as pd
from numpy import *
import tqdm
from causal_trace import make_inputs
from trace import get_out,trace_head_patch,trace_neuron_patch
from numpy import *
from causal_trace import layername

punc = ['.</s>','.\n','.',';','!',',','?','\n','</s>','<pad>']

def evalone(ans_res,reference):
    ans = ans_res.strip()
    for p in punc:
        ans = ans.replace(p, "")
    if ans.lower() == reference.strip().lower():
        return 1
    else:
        return 0


def select_heads_acts_actups(model,tokenizer,train_data):

    head_subs,act_subs,act_up_subs,hidden_subs = [],[],[],[]

    for idx in range(len(train_data)):
        q,ref_true,ref_false = train_data[idx][0],train_data[idx][1],train_data[idx][2]
        inp =  make_inputs(tokenizer, [f'{q.strip()} {ref_true.strip()}'])["input_ids"]
        if len(inp[0]) > 700:
            continue
        inp_true =  make_inputs(tokenizer, [f'{q.strip()} {ref_true.strip()}'])["input_ids"].to('cuda')
        att_val,mlp_val,mlp_up,mlp_down,att_post,head,mlp_act,mlp_act_up,mlp_gate = get_out(model,inp_true,model.device,-1)
        heads_true = np.mean(head,axis=-1)
        acts_true= mlp_act
        hidden_true = mlp_val

        inp_false =  make_inputs(tokenizer, [f'{q.strip()} {ref_false.strip()}'])["input_ids"].to('cuda')
        att_val,mlp_val,mlp_up,mlp_down,att_post,head,mlp_act,mlp_act_up,mlp_gate = get_out(model,inp_false,model.device,-1)
        heads_false = np.mean(head,axis=-1)
        acts_false = mlp_act
        hidden_false = mlp_val

        head_subs.append(np.array(heads_true)-np.array(heads_false))
        act_subs.append(np.array(acts_true)-np.array(acts_false))
        hidden_subs.append(np.array(hidden_true)-np.array(hidden_false))

    head_subs,act_subs,hidden_subs = np.mean(head_subs,axis=0),np.mean(act_subs,axis=0),np.mean(hidden_subs,axis=0)

    # print("******************acts**********************")
    min_acts,max_acts,min_act_ups,max_act_ups,min_heads,max_heads,min_hiddens,max_hiddens = [],[],[],[],[],[],[],[]

    act_values = np.array(act_subs)
    max_values = np.sort(act_values.flatten())[-1000:][::-1]
    max_indices = [np.unravel_index(np.where(act_values == value), act_values.shape) for value in max_values]

    min_values = np.sort(act_values.flatten())[:1000]
    min_indices = [np.unravel_index(np.where(act_values == value), act_values.shape) for value in min_values]

    for i in range(1000):
        layer, ind = min_indices[i][1][0][0],min_indices[i][1][1][0]
        min_acts.append([layer,ind])
    for i in range(1000):
        layer, ind = max_indices[i][1][0][0],max_indices[i][1][1][0]
        max_acts.append([layer,ind])



    # print("******************heads**********************")    
    head_values = np.array(head_subs)
    max_values = np.sort(head_values.flatten())[-128:][::-1]
    max_indices = [np.unravel_index(np.where(head_values == value), head_values.shape) for value in max_values]

    min_values = np.sort(head_values.flatten())[:128]
    min_indices = [np.unravel_index(np.where(head_values == value), head_values.shape) for value in min_values]

    for i in range(128):
        layer, ind = min_indices[i][1][0][0],min_indices[i][1][1][0]
        min_heads.append([layer,ind])
    for i in range(128):
        layer, ind = max_indices[i][1][0][0],max_indices[i][1][1][0]
        max_heads.append([layer,ind])
        
    # print("******************hiddens**********************")    
    hidden_values = np.array(hidden_subs)
    max_values = np.sort(hidden_values.flatten())[-128:][::-1]
    max_indices = [np.unravel_index(np.where(hidden_values == value), hidden_values.shape) for value in max_values]

    min_values = np.sort(hidden_values.flatten())[:128]
    min_indices = [np.unravel_index(np.where(hidden_values == value), hidden_values.shape) for value in min_values]

    for i in range(128):
        layer, ind = min_indices[i][1][0][0],min_indices[i][1][1][0]
        min_hiddens.append([layer,ind])
    for i in range(128):
        layer, ind = max_indices[i][1][0][0],max_indices[i][1][1][0]
        max_hiddens.append([layer,ind])


    return max_heads,min_heads,max_acts,min_acts,max_hiddens,min_hiddens



def steer_heads(model,tokenizer,file,test_data,max_heads,min_heads):
    out_f = open("./results/head_res.txt",'a')
    res = []
    for cnt in [1,2,4,6,8,10]:
        heads = max_heads[:cnt]
        for strength in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]:
            EMs = []
            for idx in range(len(test_data)):
                q,a = test_data[idx][0],test_data[idx][1]   
                inp =  make_inputs(tokenizer, [f'{q.strip()} {a.strip()}'])["input_ids"]
                if len(inp[0]) > 700:
                    continue
                inp_q = make_inputs(tokenizer, [f'{q}'])["input_ids"].to('cuda')
                att_val,mlp_val,mlp_up,mlp_down,att_post,head,mlp_act,mlp_act_up,mlp_gate = get_out(model,inp_q,model.device,-1)
                edit_head = head
                for layer_ind,head_ind in heads:
                    edit_head[layer_ind][head_ind] = strength*edit_head[layer_ind][head_ind]
                inp = make_inputs(tokenizer, [q])
                ntoks = inp["input_ids"].shape[1] 

                inp_com = tokenizer( f'{q} {a}', return_tensors='pt').to('cuda')
                ans = trace_head_patch(
                    model,
                    inp_com,
                    [(ntoks-1, layername(model, ly,'self_attn.head_out')) for ly in range(32)],
                    new_mlp_up = edit_head,
                    question=q,
                    answer=a)

                EM = evalone(ans,test_data[idx][1])
                EMs.append(EM)
            EMs = mean(EMs,axis=0)
            res.append(EMs)
            out_f.write(f"{strength}*Max-{cnt} heads\t{EMs}\n")
            out_f.flush()
    out_f.write("\n\n\n")
    out_f.close()
    print("steer_heads",max(res))
    return max(res)


def steer_neurons(model,tokenizer,file,test_data,max_acts,min_acts):
    out_f = open("./results/neuron_res.txt",'a')
    res = []
    for cnt in [3,5,10,30,50,100,200,300,400,500]:
        neurons = max_acts[:cnt]
        stren = [10,20,30,50,70,90]
        for strength in stren:
            EMs = []
            for idx in range(len(test_data)):
                q,a = test_data[idx][0],test_data[idx][1]   
                inp =  make_inputs(tokenizer, [f'{q.strip()} {a.strip()}'])["input_ids"]
                if len(inp[0]) > 700:
                    continue
                inp_q = make_inputs(tokenizer, [f'{q}'])["input_ids"].to('cuda')
                att_val,mlp_val,mlp_up,mlp_down,att_post,head,mlp_act,mlp_act_up,mlp_gate = get_out(model,inp_q,model.device,-1)
                edit_neuron = mlp_act
                for layer_ind,neuron_ind in neurons:
                    edit_neuron[layer_ind][neuron_ind] = strength*edit_neuron[layer_ind][neuron_ind]                   
                inp = make_inputs(tokenizer, [q])
                ntoks = inp["input_ids"].shape[1] 

                inp_com = tokenizer( f'{q} {a}', return_tensors='pt').to('cuda')
                ans = trace_neuron_patch(
                    model,
                    inp_com,
                    [(ntoks-1, layername(model, ly,'mlp.act_fn')) for ly in range(32)],
                    new_mlp_up = edit_neuron,
                    question=q,
                    answer=a)

                EM = evalone(ans,test_data[idx][1])
                EMs.append(EM)
            EMs = mean(EMs,axis=0)
            res.append(EMs)
            out_f.write(f"{strength}*max-{cnt} neurons\t{EMs}\n")
            out_f.flush()
    out_f.close()
    print("steer_neurons",max(res))
    return max(res)


def steer_hiddens(model,tokenizer,file,test_data,max_hiddens,min_hiddens):
    out_f = open("./results/hiddens_res.txt",'a')
    res = []
    for cnt in [1,2,4,6,8,10,12,14,16,18,20]:
        neurons = max_hiddens[:cnt]
        for strength in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]:
            EMs = []
            for idx in range(len(test_data)):
                q,a = test_data[idx][0],test_data[idx][1]   
                inp =  make_inputs(tokenizer, [f'{q.strip()} {a.strip()}'])["input_ids"]
                if len(inp[0]) > 700:
                    continue
                inp_q = make_inputs(tokenizer, [f'{q}'])["input_ids"].to('cuda')
                att_val,mlp_val,mlp_up,mlp_down,att_post,head,mlp_act,mlp_act_up,mlp_gate = get_out(model,inp_q,model.device,-1)
                edit_neuron = mlp_val
                for layer_ind,neuron_ind in neurons:
                    edit_neuron[layer_ind][neuron_ind] = strength*edit_neuron[layer_ind][neuron_ind]                   
                inp = make_inputs(tokenizer, [q])
                ntoks = inp["input_ids"].shape[1] 

                inp_com = tokenizer( f'{q} {a}', return_tensors='pt').to('cuda')
                ans = trace_neuron_patch(
                    model,
                    inp_com,
                    [(ntoks-1, layername(model, ly,'mlp')) for ly in range(32)],
                    new_mlp_up = edit_neuron,
                    question=q,
                    answer=a)

                EM = evalone(ans,test_data[idx][1])
                EMs.append(EM)
            EMs = mean(EMs,axis=0)
            res.append(EMs)
            out_f.write(f"{strength}*max-{cnt} hiddens\t{EMs}\n")
            out_f.flush()
    out_f.close()
    print("steer_hiddens",max(res))
    return max(res)