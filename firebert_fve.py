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

#
# FVE as a subclass
# overrides extend_batch_examples_test() used in forward()
#

import torch
from torch import nn
import numpy as np
import argparse

from switch import SWITCH
from firebert_base import FireBERT_base

class FireBERT_FVE(FireBERT_base):

    def __init__(self, load_from=None, processor=None, hparams=None):
        super(FireBERT_FVE, self).__init__(load_from=load_from, processor=processor, hparams=hparams)

        # need SWITCH to tell us what the important words are
        self.switch = SWITCH(hparams=hparams, model=self, tokenizer=self.tokenizer, device=self.device)

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
        self.switch.update_hparams(new_hparams)
        super().update_hparams(new_hparams)
        self.actualize_hparams()

    def actualize_hparams(self):
        # remember some hparams a little better
        self.count = self.hparams.vector_count
        self.perturb_words = self.hparams.perturb_words
        self.std = self.hparams.std
        return

    #
    # here are some useful defaults
    #
    def get_default_hparams(self):
        d = FireBERT_base.get_default_hparams(self)
        d.update({
                # these are for SWITCH
                'use_USE':False,
                'stop_words':True,
                'perturb_words':2,
                # this is for base
                'verbose':False,
                'vote_avg_logits':True,
                # this is for us
                'std':0.05,
                'vector_count':10
            })
        return d


        

    # this fills in the hook prepared in the base class
    def extend_batch_examples_eval(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                                    position_ids=None, head_mask=None, inputs_embeds=None, example_idx=None):
        group_ids = None
        group_sizes = []

        inputs_embeds_results = inputs_embeds
    
        # we do need the examples for SWITCH
        if example_idx is not None:
            # let's get the embeddings for the original samples
            inputs_embeds = self.bert.get_input_embeddings()(input_ids).detach()
            inputs_embeds_results = inputs_embeds

            # in groups, we keep track of samples tha belong together for voting
            group_ids = list(range(0,len(example_idx)))
            group_sizes = []
            current_group = 0
            
            # gotta go through one by one. Yes, batch logic is nicer, but SWICH is a one-by-one thing, anyway.
            for i, idx in enumerate(example_idx):
                example = self.processor.all_examples[idx] 
                # call the perturbation method individually for each example 
                perturbed_inputs_embeds, sample_attention_mask, sample_token_type_ids, _, _, _ = \
                    self.perturb_example(example,
                                            sample_index = i,
                                            input_ids=input_ids, 
                                            attention_mask=attention_mask, 
                                            token_type_ids=token_type_ids)

                # Put tensors together
                # print("perturbed examples: ", perturbed_inputs_embeds[0])
                if perturbed_inputs_embeds is None:
                    # didn't get any important words to perturb, so no batch extensions
                    group_sizes.append(1) # pus one for the original
                else:
                    inputs_embeds_results = torch.cat((inputs_embeds_results, perturbed_inputs_embeds), dim=0)
                    attention_mask = torch.cat((attention_mask, sample_attention_mask), dim=0)
                    token_type_ids = torch.cat((token_type_ids, sample_token_type_ids), dim=0)
                    group_ids += [current_group]*(self.count)
                    group_sizes.append(self.count+1) # plus one for the original

                current_group += 1

                                        
            # need to erase the tokens (input_ids) so that BERT will use the embeddings
            input_ids = None
            # we probably received these as None, but if we don't set them to that, we might have to 
            # adjust them for the new batch size and that would be tedious
            head_mask = None
            position_ids = None
            # won't need these anymore
            example_idx = None

        return input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds_results, example_idx, group_ids, group_sizes


    #
    # this perturbs an example and returns it in a batch with others
    #

    def perturb_example(self,example, count=None, std=None, sample_index=0, input_ids=None, attention_mask=None, token_type_ids=None):
        if count is None:
            count = self.count
        if std is None:
            std = self.std
        if input_ids is None:
            # convert example into features
            input_ids, attention_mask, token_type_ids, _ = self.processor.create_feature_tensors([example], device=self.device)
            sample_index = 0

        # use SWITCH to figure out word importance within the list
        word_indices, token_indices, word_list = \
            self.switch.get_important_indices_from_example(example, 
                                                            input_ids[sample_index].unsqueeze(0), 
                                                            token_type_ids[sample_index].unsqueeze(0),
                                                            attention_mask[sample_index].unsqueeze(0))

        # filter out useless stuff
        word_indices = list(filter(lambda x:x!=-1, word_indices))
        token_indices = list(filter(lambda x:x!=-1, token_indices))

        # identify most important words
        important_words = [word_list[i] for i in word_indices[:self.perturb_words]]
        token_indices = token_indices[:self.perturb_words]

        # get embeddings from BERT for the whole sample (set of words)
        embeddings = self.bert.get_input_embeddings()(input_ids[sample_index]).detach()
        #print("embeddings:",embeddings.shape)
        #print(embeddings[0,:20])
        
        batch = None
        points = None

        for token_index in token_indices:
            # get the embedding vector for the most important word
            v = embeddings[token_index].clone().detach()
            #print("vector:", v.shape)
            #print(v)

            # scale the single sample set of tokens/embeddings up to a whole batch
            batch = embeddings.repeat(count,1,1)
            # hopefully give some GPU memory back
            #del embeddings
        
            # scatter a region around this vector
            points = self.region_around(v, std=std, count=count)
            #print("region:", points.shape)
            #print(points[0])

            #print("batch embeddings before clobber:",batch.shape)
            #print(batch[0,:20])

            # clobber the tensor for the region of perturbed vectors in there
            batch[:,token_index,:] = points
            #print("batch embeddings after clobber:",batch.shape)
            #print(batch[0,:20])

        attention_mask = attention_mask[sample_index].repeat(count, 1)
        token_type_ids = token_type_ids[sample_index].repeat(count, 1)
        return batch, attention_mask, token_type_ids, points, important_words, None

    #
    # helper methods
    #

    #
    # make a field of Gaussian-perturbed vectors around a given vector v
    #
    def region_around(self, vector, std, count, device=None):
        vectors = vector.repeat(count, 1)
        #print("vectors:",vectors.shape)
        region = torch.normal(mean=vectors, std=std).cpu()
        #print("region:",region.shape)
        return region

    def get_single_vector(self, word):
        #tbd
        return None

    def get_hparam(self, name):
        return self.hparams[name]






#
# Tests. 
#

def test_FireBERT_FVE(task, set, reps=1, sample=1, hparams_default={}, hparams_lists=None, lightning=''):

    # prepare hyperparameters
    hparams = hparams_default

    # load the right processor class
    if task == "MNLI":
        processor = MnliProcessor({'sample_percent':sample}) # negative number means abs number of samples, not percent
    elif task == "IMDB":
        processor = ImdbProcessor({'sample_percent':sample})

    # now instantiate the models
    model = FireBERT_FVE(load_from='resources/models/'+task+lightning+'/pytorch_model.bin', 
                        processor=processor, 
                        hparams=hparams_default)
    processor.set_tokenizer(model.tokenizer)

    dataset, examples = processor.load_and_cache_examples("data/"+task, example_set=set)
    model.set_test_dataset(dataset, examples)

    #adv set
    # load the right processor class
    if task == "MNLI":
        adv_processor = MnliProcessor({'sample_percent':sample}) # negative number means abs number of samples, not percent
    elif task == "IMDB":
        adv_processor = ImdbProcessor({'sample_percent':sample})

    model_adv = FireBERT_FVE(load_from='resources/models/'+task+lightning+'/pytorch_model.bin', 
                        processor=processor, 
                        hparams=hparams_default)
    adv_processor.set_tokenizer(model.tokenizer)

    dataset_adv, examples_adv = adv_processor.load_and_cache_examples("data/"+task, example_set="adv_"+set)
    model_adv.set_test_dataset(dataset_adv, examples_adv)

    for i in range(reps):
        if hparams_lists is None:
            print("FireBERT_FVE specific test", task, set)
        else:
            print("FireBERT_FVE hparam test", task, set)
            print("{")
            for item in hparams_lists.items():
                key = item[0]
                values = item[1]
                hparams[key] = random.choice(values)
                print("  '"+key+"':",str(hparams[key])+",")
            print("}")

        # set the new hparams
        model.update_hparams(hparams)
        model_adv.update_hparams(hparams)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        trainer.test(model)
        result1 = trainer.tqdm_metrics

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        trainer.test(model_adv)
        result2 = trainer.tqdm_metrics

        f = open("results/five/hparams-results.csv", "a+")
        print(task, ",", "adv_"+set, ",", sample, ',"',hparams,'",',result1['avg_test_acc'],",",result2['avg_test_acc'], sep="", file=f)
        f.close()

        print("iteration",i,"logged.")
        elapsed_time()
        print()
    
        if hparams_lists is None:
            break

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
                'batch_size':8,
                # these are for SWITCH
                'use_USE':False,
                'stop_words':True,
                'perturb_words':2,
                # this is for base
                'verbose':False,
                'vote_avg_logits':True,
                # this is for us
                'std':0.05,
                'vector_count':10
            }

    hparams_lists = {
            # these are for SWITCH
            'stop_words':[True, False],
            'perturb_words':range(1,20),
            # this is for base
            'vote_avg_logits':[True, False],
            # this is for us
            'std':np.arange(0.1,10,0.01),
            'vector_count':range(3,15)
        }


    # parameter search
    sample = 15
    #test_FireBERT_FVE("IMDB", "dev", reps=500, sample=sample, hparams_default=hparams_default, hparams_lists=hparams_lists)
    #test_FireBERT_FVE("MNLI", "dev", reps=500, sample=sample, hparams_default=hparams_default, hparams_lists=hparams_lists)

    # Monday, on lightning model
    best_IMDB_lightning =  {'batch_size': 8, 'use_USE': False, 'stop_words': True, 'perturb_words': 1,
                 'verbose': False, 'vote_avg_logits': False, 'std': 8.4, 'vector_count': 11}
    best_mnli_lightning = {'batch_size': 8, 'use_USE': False, 'stop_words': False, 'perturb_words': 1, 
                        'verbose': False, 'vote_avg_logits': False, 'std': 2.31, 'vector_count': 8}

    # Tuesday, on paper model
    best_mnli =  {'batch_size': 8, 'use_USE': False, 'stop_words': True, 'perturb_words': 1, 
                        'verbose': False, 'vote_avg_logits': True, 'std': 8.14, 'vector_count': 8}
    best_IMDB={'batch_size': 8, 'use_USE': False, 'stop_words': True, 'perturb_words': 1, 
                'verbose': False, 'vote_avg_logits': True, 'std': 2.29, 'vector_count': 10}

    # actual test runs
    #test_FireBERT_FVE("IMDB", "test", reps=1, sample=100, hparams_default=best_IMDB)
    test_FireBERT_FVE("MNLI", "test", reps=1, sample=100, hparams_default=best_mnli)
