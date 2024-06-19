# PromptSub
This repository is for the paper Lexical Substitution as Causal Language Modeling. In *Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (\*SEM 2024)*, Mexico City, Mexico. Association for Computational Linguistics.

[[Paper](https://github.com/ShiningLab/PromptSub/blob/main/assets/paper.pdf)] [[Poster](https://github.com/ShiningLab/PromptSub/blob/main/assets/poster.pdf)] [[Slides](https://github.com/ShiningLab/PromptSub/blob/main/assets/slides.pdf)]

## Dependencies
+ python >= 3.11.9
+ torch >= 2.3.1
+ lightning >= 2.3.0
+ transformers >= 4.41.2
+ wandb >= 0.17.2
+ rich >= 13.7.1
+ nltk >= 3.8.1

## Setups
It is recommended to use a virtual environment to manage dependencies. Follow the steps below to set up the environment and install the required packages:
```sh
$ cd PromptSub
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Run
Before training, review and modify the training configurations in config.py as needed:
```
$ vim config.py
$ python main.py
```

## Outputs
If all goes well, you should see progress similar to the output below:
```
$ python main.py
2024-06-19 18:58:39 | Logger initialized.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
Seed set to 0
2024-06-19 18:58:39 | *Configurations:*
2024-06-19 18:58:39 |   accelerator: auto
2024-06-19 18:58:39 |   seed: 0
...
2024-06-19 18:53:06 |   device: cuda
2024-06-19 18:53:06 | Start training...
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃   ┃ Name  ┃ Type            ┃ Params ┃ Mode ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ 0 │ model │ GPT2LMHeadModel │  124 M │ eval │
└───┴───────┴─────────────────┴────────┴──────┘
Trainable params: 124 M
Non-trainable params: 0
Total params: 124 M
Total estimated model params size (MB): 497
...
2024-06-19 18:55:41 | Start postprocessing...
2024-06-19 18:55:42 | Start ranking...
...
2024-06-19 18:56:34 | Results saved as ./res/results/ls21/wsample/best/base/gpt2-medium/0.pkl.
2024-06-19 18:56:34 | Done.
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```
TODO
```