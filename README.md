<div align="center">
<h1>
INCLINE
</h1>
</div>

This repository contains the data and codes for our paper "[Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](https://arxiv.org/pdf/2410.12462)".
### 1. Data & Model

Please download the data of downstream tasks and put it in ./data/

### 2. Intervention

For Discriminative and Generative tasks:

```
python intervention.py
```

For MGSM task:

```
python intervention_llama.py
```


### Citation
If you find this work is useful or use the data in your work, please consider cite our paper:

```
@article{wang2024bridging,
  title={Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention},
  author={Wang, Weixuan and Wu, Minghao and Haddow, Barry and Birch, Alexandra},
  journal={arXiv preprint arXiv:2410.12462},
  year={2024}
}
```
