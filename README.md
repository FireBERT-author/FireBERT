# FireBERT
Hardening BERT classifiers against adversarial attack

## Paper

Please read our paper: 
[FireBERT 1.0](https://github.com/FireBERT-author/FireBERT/blob/master/FireBERT.pdf). When citing our work, please include a link to this repository.

## Instructions

The best way to run our project is to download the .zip files in [release v1.0](https://github.com/FireBERT-author/FireBERT/releases/tag/v1.0). Expand the "data.zip", "resources-1.zip" and "resources-2.zip" files into "data" and "resources" folders, respectively. 

- To obtain the values from tables 1 and 2 in the "Results" section of the paper, run the respective "eval_xxxx.ipynb" notebooks.

- To obtain the values from table 3, run "generate_adversarials.ipynb"

- To tune a basic BERT model, run the "bert_xxxx_tuner.ipynb" notebooks.

- To co-tune FACT on synthetic adversarials, run the "firebert_xxxx_and_adversarial_co-tuner.ipynb" - notebooks.

- To recreate the illustrations in the "Analysis" section, play around with "analysis.ipynb". It produces many possible graphs. Try changing the values at the top of cells.

## Major Pre-requisites

- tensorflow 2.1 or higher, GPU preferred

- torch, torchvision (PyTorch) 1.3.1 or higher

- pytorch-lightning 0.7.1 or higher

- transformers (Hugging Face) 2.5.1 or higher
