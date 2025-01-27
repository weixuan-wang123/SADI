import torch
from baukit import TraceDict
from collections import defaultdict
from util import nethook
def get_out(model, prompt, device,index): 

    model.eval()
    ATT = [f"model.layers.{i}.self_attn" for i in range(32)]
    ATT_q= [f"model.layers.{i}.self_attn.q_proj" for i in range(32)]
    ATT_k = [f"model.layers.{i}.self_attn.k_proj" for i in range(32)]
    ATT_v = [f"model.layers.{i}.self_attn.v_proj" for i in range(32)]
    ATT_o = [f"model.layers.{i}.self_attn.o_proj" for i in range(32)]
    MLP = [f"model.layers.{i}.mlp" for i in range(32)]
    MLP_gate = [f"model.layers.{i}.mlp.gate_proj" for i in range(32)]
    MLP_up = [f"model.layers.{i}.mlp.up_proj" for i in range(32)]
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(32)]
    MLP_act_up = [f"model.layers.{i}.mlp.act_up" for i in range(32)]
    MLP_down = [f"model.layers.{i}.mlp.down_proj" for i in range(32)]
    ATT_post = [f"model.layers.{i}.post_attention_layernorm" for i in range(32)]
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(32)]
    with torch.no_grad():
        with TraceDict(model, ATT+ATT_q+ATT_k+ATT_v+ATT_o+MLP+MLP_gate+MLP_act+MLP_up+MLP_down+ATT_post+HEADS+MLP_act_up) as ret:
            output = model(prompt, output_hidden_states = True,output_attentions=True)
       
        MLP_act_value = [ret[mlp_act].output[0][index].detach().cpu().numpy() for mlp_act in MLP_act]
        ATT_value = [ret[att_value].output[0][index].detach().cpu().numpy() for att_value in ATT]
        MLP_value = [ret[mlp_value].output[0][index].detach().cpu().numpy() for mlp_value in MLP]
        MLP_up_value = [ret[mlp_up].output[0][index].detach().cpu().numpy() for mlp_up in MLP_up]
        MLP_gate_value = [ret[mlp_gate].output[0][index].detach().cpu().numpy() for mlp_gate in MLP_gate]
        MLP_act_up_value = [ret[mlp_act_up].output[0][index].detach().cpu().numpy() for mlp_act_up in MLP_act_up]
        MLP_down_value = [ret[mlp_down].output[0][index].detach().cpu().numpy() for mlp_down in MLP_down]
        ATT_post_value = [ret[att_post].output[0][index].detach().cpu().numpy() for att_post in ATT_post]
        HEAD_value = [ret[head_value].output[0][index].detach().cpu().numpy() for head_value in HEADS] #torch.Size([32, 10, 128])

        return ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value,HEAD_value,MLP_act_value,MLP_act_up_value,MLP_gate_value

def trace_head_patch(
    model,  
    tokenizer,
    inp,  
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
        for t in patch_spec[layer]:
            ly = int(layer.split('model.layers.')[-1].split('.self_attn.head_out')[0])
            h[:,t,:] = torch.tensor(new_mlp_up[ly])
        return h

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:

        target_ids = tokenizer(answer, return_tensors='pt')['input_ids'].to('cuda')

        outputs = model(**inp)
        logits = outputs.logits
        ans = torch.argmax(logits, dim=-1)[:, -target_ids.size(1):-1].squeeze()
        ans_idss = ans.detach().cpu().numpy().tolist()

        textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True)
        return textual_ans.strip()


def trace_neuron_patch(
    model,  
    tokenizer,
    inp,  
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
        for t in patch_spec[layer]:
            ly = int(layer.split('model.layers.')[-1].split('.mlp')[0])
            h[:,t,:] = torch.tensor(new_mlp_up[ly])
        return h

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        target_ids = tokenizer(answer, return_tensors='pt')['input_ids'].to('cuda')

        outputs = model(**inp)
        logits = outputs.logits
        ans = torch.argmax(logits, dim=-1)[:, -target_ids.size(1):-1].squeeze()
        ans_idss = ans.detach().cpu().numpy().tolist()

        textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True)
        return textual_ans.strip()
    