# coding=utf-8
# Copyright 2020 FireBERT authors. All rights reserved.
#
# Licensed under the MIT license
# See https://github.com/FireBERT-author/FireBERT/blob/master/LICENSE for details
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import pytorch_lightning as pl

from processors import MnliProcessor
from firebert_fct import FireBERT_FCT
from bert_base_model import LightningBertForSequenceClassification

num_gpus = -1 if torch.cuda.is_available() else None

import gc
import random
gc.enable()

import time

from torch.utils.data import TensorDataset

# prepare hyperparameters

max_steps = -1 # if -1 then calculate number of training steps based on the length of the train set
len_train_set = 392702

gradient_accumulation_steps = 1
learning_rate = 2e-5
weight_decay = 0.00
adam_epsilon = 1e-8
warmup_proportion = 0 

num_train_epochs = 1
batch_size = 7

if max_steps > 0:
    num_train_epochs = max_steps // (len_train_set // gradient_accumulation_steps) + 1
    num_training_steps = max_steps
else:
    num_training_steps = len_train_set // gradient_accumulation_steps * num_train_epochs
    
warmup_steps = num_training_steps // num_train_epochs * warmup_proportion

def load_examples(features_file):

    features = torch.load(features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    all_idxs = torch.tensor([i for i in range(len(all_input_ids))], dtype=torch.long)
        
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_idxs)
    
    return dataset

def elapsed_time():
    global t_start

    t_now = time.time()
    t = t_now-t_start
    t_start = t_now
    return t

use_full_example = True
use_USE = True # if random.randint(1, 2) == 1 else False
USE_method = 'rank' #'filter' if random.randint(1, 2) == 1 else 'rank'
USE_multiplier = 12 #random.randint(5, 15) # 3
stop_words = True
perturb_words = 9 #random.randint(4, 9) # 2
candidates_per_word = 10 #random.randint(5, 15) #10
total_alternatives = 4 #random.randint(4, 7) # 5
match_pos = True #if random.randint(1, 2) == 1 else False
leave_alone = 0
random_out_of = 0
judge_bert = False

hparams = { 'learning_rate': learning_rate,
            'adam_epsilon': adam_epsilon,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'num_training_steps': num_training_steps,
            'batch_size': batch_size,
            'use_USE': use_USE,
            'USE_method': USE_method,
            'USE_multiplier': USE_multiplier,
            'stop_words': stop_words,
            'perturb_words': perturb_words,
            'candidates_per_word': candidates_per_word,
            'total_alternatives': total_alternatives,
            'match_pos': match_pos,
            'use_full_example': use_full_example,
            'leave_alone': leave_alone,
            'random_out_of': random_out_of,
            'judge_bert': judge_bert
           }

print(hparams)

proc_hparams = {}
# delete this next line to run full 100%
proc_hparams.update({'sample_percent': 3,
                     'randomize': True})

# instantiate the model used for SWITCH
switch_model = LightningBertForSequenceClassification(load_from = 'resources/models/MNLI/pytorch_model.bin', 
                                                      processor = MnliProcessor(), 
                                                      hparams = {'batch_size': 6 })
switch_model.cuda()

model = FireBERT_FCT(switch_model=switch_model, processor=MnliProcessor(hparams=proc_hparams), hparams=hparams)

processor = model.get_processor()

train_dataset, train_examples = processor.load_and_cache_examples("data/MNLI", example_set='train')
val_dataset, _ = processor.load_and_cache_examples("data/MNLI", example_set='dev')
test_dataset, _ = processor.load_and_cache_examples("data/MNLI", example_set='test')

model.set_train_dataset(train_dataset, train_examples)
model.set_val_dataset(val_dataset)
model.set_test_dataset(test_dataset)

amp_opt_level='O1' 
max_grad_norm = 1.0

t_start = time.time()

trainer = pl.Trainer(gpus=num_gpus,
                     max_epochs = num_train_epochs, amp_level=amp_opt_level, gradient_clip_val=max_grad_norm,
                     max_steps = num_training_steps)

trainer.fit(model)

training_time = round(elapsed_time(),2)

trainer.test(model)

train_results = trainer.tqdm_metrics


del train_dataset
del train_examples
del val_dataset
del test_dataset
del trainer

gc.collect()
torch.cuda.empty_cache()

# compare how well the model does against adversarial samples
test_dataset = load_examples('data/MNLI/generated/mnli_adversarial_samples_for_dev')
model.set_test_dataset(test_dataset)
trainer = pl.Trainer(gpus=num_gpus)
trainer.test(model)
adv_results = trainer.tqdm_metrics

results = "training_time: " + str(training_time) + "s, " + \
          "val_loss: " + str(train_results['val_loss']) + ", " + \
          "val_acc: " + str(train_results['avg_val_acc']) + ", " + \
          "adv_val_acc: " + str(adv_results['avg_test_acc']) + ", " + \
          "use_USE: " + str(use_USE) + ", " + \
          "USE_multiplier: " + str(USE_multiplier) + ", " + \
          "USE_method: " + str(USE_method) + ", " + \
          "perturb_words: " + str(perturb_words) + ", " + \
          "candidates_per_word: " + str(candidates_per_word) + ", " + \
          "total_alternatives: " + str(total_alternatives) + ", " + \
          "match_pos: " + str(match_pos)

print(results)

fname = str("results/fct/mnli-hparams-results.txt")
f = open(fname, "a")
f.write(results)
f.write( "\n")
f.close()

del processor
del test_dataset
del model
del switch_model
del trainer
del train_results
del adv_results
del results
del fname
del f

gc.collect()
torch.cuda.empty_cache() 


