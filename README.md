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

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```
TODO
```