# FireBERT
Hardening BERT classifiers against adversarial attack

Gunnar Mein, UC Berkeley MIDS Program (gunnarmein@berkeley.edu)\
Kevin Hartman, UC Berkeley MIDS Program (kevin.hartman@berkeley.edu)\
Andrew Morris, UC Berkeley MIDS Program (andrew.morris@berkeley.edu)

With many thanks to our advisors: Mike Tamir, Daniel Cer and Mark Butler for their guidance on this research. And to our significant others as the three of us hunkered down over the three month project.

*Note: This repo used to be anonymous while the paper was in blind review. 

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

## Hardware and run-time expectations

Authors used Intel i7-9th generation personal computers with 64 GB of main memory and NVIDIA 2080 (Max-Q and ti) graphics cards, and various GCP instances. Full evaluation runs for pre-made adversarial samples can be done in a small number of hours. Active attack benchmarks with TextFooler are done in hours for MNLI, but might take days for IMDB. Co-tuning for FACT is expected to run for multiple hours. 
