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
# FSE as a subclass
# overrides extend_batch_examples_test() used in forward()
#

import argparse
import torch
from torch import nn
from switch import SWITCH
from firebert_base import FireBERT_base

class FireBERT_FSE(FireBERT_base):

    def __init__(self, load_from=None, processor=None, hparams={}):
        super(FireBERT_FSE, self).__init__(load_from=load_from, processor=processor, hparams=hparams)
        
        self.switch = SWITCH(hparams=hparams, model=self, tokenizer=self.tokenizer, device=self.device)

        # merge passed hparams over default hparams
        hdict = self.get_default_hparams()
        hdict.update(hparams)
        self.hparams = argparse.Namespace(**hdict)
        self.actualize_hparams()
        return

    #
    # here are some useful defaults
    #
    def get_default_hparams(self):
        d = FireBERT_base.get_default_hparams(self)
        d.update({
                # these are for SWITCH
                'use_USE':True,
                'USE_method':"filter",
                'USE_multiplier':3,
                'stop_words':True,
                'perturb_words':2,
                'candidates_per_word':10,
                'total_alternatives':5,
                'match_pos':True,
                # this is for base
                'verbose':False,
                'vote_avg_logits':True
            })
        return d

    # methods to set hparams on the fly
    def update_hparams(self, new_hparams):
        hdict = vars(self.hparams)
        hdict.update(new_hparams)
        self.hparams = argparse.Namespace(**hdict)
        self.switch.update_hparams(new_hparams)
        super().update_hparams(new_hparams)
        self.actualize_hparams()

    def actualize_hparams(self):
        return




    # this fills in the hook prepared in the base class
    def extend_batch_examples_eval(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                                    position_ids=None, head_mask=None, inputs_embeds=None, example_idx=None):
        group_ids = None
        group_sizes = []
        if example_idx is not None:
            group_ids = list(range(0,len(example_idx)))
            group_sizes = []
            current_group = 0
            for idx in example_idx:
                example = self.processor.all_examples[idx] 
                new_texts = self.switch.generate_candidates_from_example(example)

                if new_texts != []:
                    alternates, ex_idx = self.processor.create_adversarial_examples(example, new_texts)
                    
                    # todo: this code needs to move into processor.create_adversarial_examples()
                    # right now it cannot go there as the model stores the examples (why?)
                    # ex_idx as returned above should already be correct, and examples should have been added to master list
                    base = len(self.processor.all_examples)
                    self.processor.all_examples += tuple(alternates)
                    ex_idx  = [idx+base for idx in ex_idx]
                    # end
                    
                    alt_input_ids, alt_attention_mask, alt_token_type_ids, _ = self.processor.create_feature_tensors(alternates)

                    input_ids = torch.cat((input_ids, alt_input_ids.to(self.device)), dim=0)
                    attention_mask = torch.cat((attention_mask, alt_attention_mask.to(self.device)), dim=0)
                    token_type_ids = torch.cat((token_type_ids, alt_token_type_ids.to(self.device)), dim=0)
                    # example_idx = torch.cat((example_idx, torch.tensor(ex_idx, device=self.device)), dim=0)

                    group_ids += [current_group]*(len(new_texts))

                group_sizes.append(len(new_texts)+1) # plus one for the original
                current_group += 1


        return input_ids, attention_mask, token_type_ids, None, None, None, None, group_ids, group_sizes


    #
    # this perturbs an example and returns a batch
    # for vector plotting purposes only, will only process one important word
    #

    def perturb_example(self,example, count=None, std=None, sample_index=0, input_ids=None, attention_mask=None, token_type_ids=None):
        if count is None:
            count = self.count
        if count > 100:
            raise("FSE.perturb_example cannot deliver more than 100 alternative vectors due to cos_nn precomputed size.")

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

        # identify most important word
        important_words = [word_list[i] for i in word_indices[:1]]
        token_index = token_indices[0]
        #print("most important word:", "'"+word+"',", "token index:", token_index)

        # get embeddings from BERT for the whole sample (set of words)
        embeddings = self.bert.get_input_embeddings()(input_ids[sample_index]).detach().cpu()
        #print("embeddings:",embeddings.shape)
        #print(embeddings[0,:20])
        
        # get the embedding vector for the most important word
        #v = embeddings[token_index].clone().detach()
        #print("vector:", v.shape)
        #print(v)

        # scale the single sample set of tokens/embeddings up to a whole batch
        batch = embeddings.repeat(count,1,1)
        # hopefully give some GPU memory back
        del embeddings
    
        # scatter a region around this vector
        points, words = self.words_around(important_words[0], std=std, count=count)
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
        print("fse: batch:",batch.size())
        print("fse: points:",points.size())
        return batch, attention_mask, token_type_ids, points, important_words, words




    #
    # make a field of replacement-word-vectors around a given word
    #
    def words_around(self, word, std, count, device=None):
        #print("fse: count:", count)
        words = self.switch.get_replacement_words_for_display(word, count-1)
        words = [word]+words
        #print("fse: words:",words)
        token_ids = []
        for word in words:
            token_id = self.tokenizer.encode(word, add_special_tokens = False)[0] # take only the first token of any word
            token_ids.append(token_id)

        # now get BERT word-input embeddings
        token_ids = torch.tensor(token_ids, device=self.device)
        #print("fse: token_ids:",token_ids.size())
        region = self.bert.get_input_embeddings()(token_ids).detach().cpu()
        #print("fse: region:",region.size())
            
        return region, words





#
# Tests.  notes that test_firebert alone is a simple 'does it work' check for base
# Code.  The iteratble generates output based on random paramaters for evaluations
#

def test_param_FireBERT_FSE(hparams, dset='mnli', sample_pct=3):
    '''This class exists specifically to reproduce results from randomized
    control trials.  Pass in hparams as an argument'''

    u = hparams['use_USE']
    um = hparams['USE_method']
    p = hparams['perturb_words']
    c = hparams['candidates_per_word']
    t = hparams['total_alternatives']
    v = hparams['vote_avg_logits']

    print("Simple Static FireBERT_FSE test")
    if dset == 'mnli':
        # now instantiate the MNLI model
        processor = MnliProcessor({'sample_percent':sample_pct})
        model = FireBERT_FSE(load_from='resources/models/MNLI/pytorch_model.bin', 
                            processor=processor, 
                            hparams=hparams)

        processor = model.get_processor()

        dataset, examples = processor.load_and_cache_examples("data/MNLI", example_set='dev')
        model.set_test_dataset(dataset, examples)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        print("Start testing")
        trainer.test(model)
        resulta = trainer.tqdm_metrics

        dataset, examples = processor.load_and_cache_examples("data/MNLI", example_set='adv_test')
        model.set_test_dataset(dataset, examples)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        print("Start testing")
        trainer.test(model)
        resultb = trainer.tqdm_metrics

        print("Results: ")
        print("Regular results", resulta['avg_test_acc'])
        print("Adversarial results", resultb['avg_test_acc'])

        fname = str("results/fse/ind-result-mnli-"+str(u)+str(um)+"-"+str(p)+"-"+str(c)+"-"+str(t)+"-"+str(v)+".txt")
        f = open(fname, "a")
        f.write(str(u)+"-"+str(um)+"-"+str(p)+"-"+str(c)+"-"+str(t)+"-"+str(v)+","+str(resulta['avg_test_acc'])+","+str(resultb['avg_test_acc']))
        f.write("\n")
        f.close()
        processor = None
        dataset = None
        examples = None
        model = None
        trainer = None

    elif dset == 'imdb':
        # now instantiate the IMDB model
        processor = ImdbProcessor({'sample_percent':sample_pct}) # negative number means abs number of samples, not percent

        model = FireBERT_FSE(load_from='resources/models/IMDB/pytorch_model.bin', 
                            processor=processor, 
                            hparams=hparams)

        processor = model.get_processor()

        dataset, examples = processor.load_and_cache_examples("data/IMDB", example_set='dev')
        model.set_test_dataset(dataset, examples)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        print("Start testing")
        trainer.test(model)
        resulta = trainer.tqdm_metrics

        dataset, examples = processor.load_and_cache_examples("data/IMDB", example_set='adv_test')
        model.set_test_dataset(dataset, examples)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        print("Start testing")
        trainer.test(model)
        resultb = trainer.tqdm_metrics

        print("Results: ")
        print("Regular results", resulta['avg_test_acc'])
        print("Adversarial results", resultb['avg_test_acc'])

        fname = str("results/fse/ind-result-imdb-"+str(u)+str(um)+"-"+str(p)+"-"+str(c)+"-"+str(t)+"-"+str(v)+".txt")        
        f = open(fname, "a")
        f.write(str(u)+"-"+str(um)+"-"+str(p)+"-"+str(c)+"-"+str(t)+"-"+str(v)+","+str(resulta['avg_test_acc'])+","+str(resultb['avg_test_acc']))
        f.write("\n")
        f.close()
        processor = None
        dataset = None
        examples = None
        model = None
        trainer = None
        
    else:
        print("Invalid data set selected!")

def test_iter_FireBERT_FSE(set='mnli', sample_pct=3):

    import gc
    import random
    gc.enable()

    print("Simple FireBERT_FSE tests")
    # create a FireBERT_FSE classifier for MNLI
    
    # negative number means abs number of samples, not percent
    u = random.randint(2,20)
    p = random.randint(2,15)
    c = random.randint(3,15)
    t = random.randint(3,20)
    b = 32
    random_bit = random.getrandbits(1)
    v = bool(random_bit)

    # prepare hyperparameters
    hparams = {
                'use_USE':True,
                'USE_method':"filter",
                'USE_multiplier':u,
                'stop_words':True,
                'perturb_words':p,
                'candidates_per_word':c,
                'total_alternatives':t,
                'match_pos':True,
                'batch_size':b, 
                'verbose':False, 
                'vote_avg_logits':v
            }
    if set == 'mnli':
        # now instantiate the MNLI model
        processor = MnliProcessor({'sample_percent':sample_pct})
        model = FireBERT_FSE(load_from='resources/models/MNLI/pytorch_model.bin', 
                            processor=processor, 
                            hparams=hparams)

        processor = model.get_processor()

        dataset, examples = processor.load_and_cache_examples("data/MNLI", example_set='dev')
        model.set_test_dataset(dataset, examples)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        print("MNLI: Start testing dev set")
        trainer.test(model)
        resulta = trainer.tqdm_metrics

        dataset, examples = processor.load_and_cache_examples("data/MNLI", example_set='adv_dev')
        model.set_test_dataset(dataset, examples)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        print("MNLI: Start testing adversarial dev set")
        trainer.test(model)
        resultb = trainer.tqdm_metrics

        fname = str("results/fse/mnli-hparams-results.txt")
        f = open(fname, "a")
        f.write(str(u)+"-"+str(p)+"-"+str(c)+"-"+str(t)+"-"+str(b)+"-"+str(v)+","+str(resulta['avg_test_acc'])+","+str(resultb['avg_test_acc']))
        f.write("\n")
        f.close()
        processor = None
        dataset = None
        examples = None
        model = None
        trainer = None
        gc.collect()  
    elif set == 'imdb':
        # now instantiate the IMDB model
        processor = ImdbProcessor({'sample_percent':sample_pct}) # negative number means abs number of samples, not percent

        model = FireBERT_FSE(load_from='resources/models/IMDB/pytorch_model.bin', 
                            processor=processor, 
                            hparams=hparams)

        processor = model.get_processor()

        dataset, examples = processor.load_and_cache_examples("data/IMDB", example_set='dev')
        model.set_test_dataset(dataset, examples)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        print("IMDB: Start testing dev set")
        trainer.test(model)
        resulta = trainer.tqdm_metrics

        dataset, examples = processor.load_and_cache_examples("data/IMDB", example_set='adv_dev')
        model.set_test_dataset(dataset, examples)

        trainer = pl.Trainer(gpus=(-1 if torch.cuda.is_available() else None))
        print("IMDB: Start testing adversarial dev set")
        trainer.test(model)
        resultb = trainer.tqdm_metrics

        fname = str("results/fse/imdb-hparams-results.txt")
        f = open(fname, "a")
        f.write(str(u)+"-"+str(p)+"-"+str(c)+"-"+str(t)+"-"+str(b)+"-"+str(v)+","+str(resulta['avg_test_acc'])+","+str(resultb['avg_test_acc']))
        f.write("\n")
        f.close()
        processor = None
        dataset = None
        examples = None
        model = None
        trainer = None
        gc.collect() 
    else:
        print("Invalid data set selected!")

if __name__ == "__main__":
    import pytorch_lightning as pl
    from processors import MnliProcessor, ImdbProcessor

    # Run iterations of imdb and mnli
    #test_iter_FireBERT_FSE('imdb', 3)
    #test_iter_FireBERT_FSE('mnli', 3)


    # The section and function below is optionally for 
    # verifying results
    hparams = {
            'use_USE':False,
            'USE_method':"filter",
            'USE_multiplier':15,
            'stop_words':True,
            'perturb_words':2,
            'candidates_per_word':15,
            'total_alternatives':15,
            'match_pos':True,
            'batch_size':32, 
            'verbose':False, 
            'vote_avg_logits':False
        }
    
    test_param_FireBERT_FSE(hparams, 'mnli')
    test_param_FireBERT_FSE(hparams, 'imdb')