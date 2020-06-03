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

import argparse
import torch
from torch import nn
import numpy as np

# these are ours
from bert_base_model import LightningBertForSequenceClassification

class FireBERT_base(LightningBertForSequenceClassification):

    def __init__(self, load_from=None, processor=None, hparams={}):
        super(FireBERT_base, self).__init__(load_from=load_from, processor=processor, hparams=hparams)

        # merge passed hparams over default hparams
        hdict = self.get_default_hparams()
        hdict.update(hparams)
        self.hparams = argparse.Namespace(**hdict)
        self.actualize_hparams()
        return


    # methods to set hparams on the fly
    def update_hparams(self, new_hparams):
        hdict = vars(self.hparams)
        hdict.update(new_hparams)
        self.hparams = argparse.Namespace(**hdict)
        self.actualize_hparams()

    def actualize_hparams(self):
        return





    # hooks for making fuzzy samples in train phase, no-op here
    def extend_batch_examples_train(self, batch):
        input_ids, attention_mask, token_type_ids, label, example_idx = batch
        return input_ids, attention_mask, token_type_ids, label, example_idx

    # hooks for making fuzzy samples in eval mode, no-op here except erases input_embeds (can't have both ids and embeds)
    def extend_batch_examples_eval(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                                    position_ids=None, head_mask=None, inputs_embeds=None, example_idx=None):
        group_ids = None
        group_sizes = None
        return input_ids, attention_mask, token_type_ids, position_ids, head_mask, None, example_idx, group_ids, group_sizes


    #
    # forward is extended with calling hook and voting on results
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, example_idx=None, extend=True):

        group_ids = None
        output_length = len(input_ids) if input_ids is not None else len(inputs_embeds)

        if extend and not self.training:
            # use the hook
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, example_idx, group_ids, group_sizes =\
                 self.extend_batch_examples_eval(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            example_idx=example_idx)

        if not extend and input_ids is not None and inputs_embeds is not None:
            inputs_embeds=None

        result_logits = None
        for i in range(0,attention_mask.shape[0], self.hparams.batch_size):
            bs = min(self.hparams.batch_size, attention_mask.shape[0]-i)
            if input_ids is not None:
                batch_input_ids=input_ids.narrow(0,i,bs)
            else:
                batch_input_ids = None

            if inputs_embeds is not None:
                batch_inputs_embeds=inputs_embeds.narrow(0,i,bs)
            else:
                batch_inputs_embeds = None

            batch_attention_mask=attention_mask.narrow(0,i,bs)
            batch_token_type_ids=token_type_ids.narrow(0,i,bs)

            if position_ids is not None:
                batch_position_ids=position_ids.narrow(0,i,bs)
            else:
                batch_position_ids = None

            if head_mask is not None:
                batch_head_mask = head_mask.narrow(0,i,bs)
            else:
                batch_head_mask = None

            logits = LightningBertForSequenceClassification.forward(self, 
                            input_ids=batch_input_ids, 
                            attention_mask=batch_attention_mask, 
                            token_type_ids=batch_token_type_ids,
                            position_ids=batch_position_ids,
                            head_mask=batch_head_mask,
                            inputs_embeds=batch_inputs_embeds)

            if result_logits is None:
                result_logits = logits
            else:
                result_logits = torch.cat((result_logits, logits), dim=0)

        logits = result_logits

        #
        # time to vote
        # Makes big empty tensor for all groups
        # uses torch.view to update individual group
        #
        if group_ids is not None:
            # prepare a couple of output tensors of the right dimensions
            avg_logits = torch.zeros(output_length, self.num_labels).to(self.device)
            counted_logits = torch.zeros(output_length, self.num_labels).to(self.device)
            original_logits = logits[:output_length]

            # now go through the whole extended batch
            for i, (logit, group_id) in enumerate(zip(logits, group_ids)):

                # first, tally logits by averaging across replacement groups
                current_group_logits = torch.narrow(avg_logits, 0, group_id, 1)
                torch.add(current_group_logits, torch.div(logit, group_sizes[group_id]), out=current_group_logits)

                # but also, record the individual VOTES (argmax)
                current_vote = torch.argmax(logit).item()
                counted_logits[group_id, current_vote] += 1/group_sizes[group_id]

                # let us know what is happening here
                self._debug_print_vote(i, group_id, logit, example_idx)
    
            # print the results for this batch
            self._debug_print_votes(original_logits, avg_logits, counted_logits)

            if self.hparams.vote_avg_logits:
                logits = avg_logits
            else:
                logits = counted_logits


        #
        # return whatever we have at this point
        #
        return logits

    #
    # training step is extended with calling the hook
    #
    def training_step(self, batch, batch_nb):
        # extension hook is called from here in training
        # instead of in forward, so that backward steps happen with new examples as well
        input_ids, attention_mask, token_type_ids, label, example_idx = self.extend_batch_examples_train(batch)

        return LightningBertForSequenceClassification.training_step(self, batch=(input_ids, attention_mask, token_type_ids, label, example_idx),batch_nb=batch_nb)


    # debug print:
    # 
    def _debug_print_vote(self, i, group_id, logit, example_idx):
        if self.hparams.verbose:
            example = self.all_examples[i] 
            text = self.processor.get_text_to_perturb(example)
            print("Group ID:", group_id, "vote:", torch.argmax(logit).item(), "logits:", logit.tolist(), 
                    "text:",text[:40], "..." if len(text)>40 else "")

    def _debug_print_votes(self, original, average, counted):
        if self.hparams.verbose:
            print()
            for group_id, (o, a, c) in enumerate(zip(original, average, counted)):
                print("Group ID:", group_id, 
                    "original:", torch.argmax(o).item(),
                    "average:", torch.argmax(a).item(),
                    "majority:", torch.argmax(c).item()
                )
            print()

#
# simple tests
#


def test_FireBERT_base(task, set, reps=1, sample=1, hparams_default={}, tf=False):

    # prepare hyperparameters
    hparams = hparams_default

    # load the right processor class
    if task == "MNLI":
        processor = MnliProcessor({'sample_percent':sample}) # negative number means abs number of samples, not percent
        adv_processor = MnliProcessor({'sample_percent':sample}) # negative number means abs number of samples, not percent
    elif task == "IMDB":
        processor = ImdbProcessor({'sample_percent':sample})
        adv_processor = ImdbProcessor({'sample_percent':sample})

    lightning = "_on_lightning" if not tf else ""
    # now instantiate the models - one for the regular set
    model = FireBERT_base(load_from='resources/models/'+task+lightning+'/pytorch_model.bin', 
                        processor=processor, 
                        hparams=hparams_default)

    dataset, examples = processor.load_and_cache_examples("data/"+task, example_set=set)
    model.set_test_dataset(dataset, examples)

    # one for the adversarial set

    model_adv = FireBERT_base(load_from='resources/models/'+task+lightning+'/pytorch_model.bin', 
                        processor=processor, 
                        hparams=hparams_default)

    dataset_adv, examples_adv = adv_processor.load_and_cache_examples("data/"+task, example_set="adv_"+set)
    model_adv.set_test_dataset(dataset_adv, examples_adv)

    #
    # now test them both, and log results
    #
    trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
    trainer.test(model)
    result = trainer.tqdm_metrics

    f = open("results/base/hparams-results.csv", "a+")
    print(task, ",", set, ",", sample, ',"',hparams,'",',result['avg_test_acc'], sep="", file=f)
    f.close()

    trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
    trainer.test(model_adv)
    result = trainer.tqdm_metrics

    f = open("results/base/hparams-results.csv", "a+")
    print(task, ",", "adv_"+set, ",", sample, ',"',hparams,'",',result['avg_test_acc'], sep="", file=f)
    f.close()


    print("baseline data logged.")
    elapsed_time()
    print()

def elapsed_time():
    global t_start
    t_now = time.time()
    t = t_now-t_start
    print("elapsed time: ",round(t,2), "s")
    t_start = t_now
    return t

  

if __name__ == "__main__":
    import random
    import time
    import pytorch_lightning as pl
    from processors import MnliProcessor, ImdbProcessor

    t_start = time.time()
    # prepare hyperparameters
    hparams_default = {
                'batch_size':32,
                # this is for base
                'verbose':False,
            }
    sample = 100

    #test_FireBERT_base("MNLI", "dev", reps=1, sample=sample, hparams_default=hparams_default)
    test_FireBERT_base("IMDB", "dev", reps=1, sample=sample, hparams_default=hparams_default, tf=True)
    
