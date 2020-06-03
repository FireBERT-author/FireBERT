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
"""SWITCH: Shared text perturbation component for FireBERT"""

import os
import argparse
import numpy as np
import nltk
import re
import gc
import numpy as np
from random import shuffle

# don't import tensorflow as is since we will have tf 2.0 and this will break things
# import tensorflow as tf
# do this instead to maintain compatibility with tf 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import tensorflow_hub as hub
import torch
import torch.nn as nn
from transformers import (BertConfig, BertModel, BertTokenizer)
from transformers.data.processors.utils import (InputExample)


    
#
# this will do all the work
#

class SWITCH:
    """ SWITCH (FireBERT)
    a class to facilitate discovery of important words for classification
    and perturbation thereof.
    """
    
    def __init__(self, hparams={}, model=None, tokenizer=None, perturb_text_b=None, device="cpu"):
        self.cos_nn, self.idx2word, self.word2idx, self.idx2norm = load_cos_nn()

        # merge passed hparams over default hparams
        hdict = self.get_default_hparams()
        hdict.update(hparams)
        self.hparams = argparse.Namespace(**hdict)

        # now do any intializations of internals related to hparams
        self.USE = None # to force intialization
        # gotta have stop words, otherwise occasionally we will perturb "and"
        # Stop-word list copied from TextFooler, see separate notice below
        self.all_stop_words = get_stopwords()
        # this is what makes them work, don't forget this line
        self.actualize_hparams()

        # hold on to the BERT model and tokenizer we got, we need it for gradients
        self.tokenizer = tokenizer
        self.model = model
        self.processor = model.processor

        # set the override first
        self.perturb_text_b = perturb_text_b
        if self.perturb_text_b is None:
            # but if we didn't get explicit instruction, ask the processor
            self.perturb_text_b = self.processor.get_text_to_perturb_is_b(None)

        # and what device we are supposed to use for Torch tensors
        self.device = device
        return

    # methods to set hparams on the fly
    def update_hparams(self, new_hparams):
        hdict = vars(self.hparams)
        hdict.update(new_hparams)
        self.hparams = argparse.Namespace(**hdict)
        self.actualize_hparams()

    def actualize_hparams(self):
        if self.hparams.use_USE:
            # load or build the semantic similarity module, but only once
            if self.USE is None:
                self.USE = USE("scratch/tf_cache")
            # method "rank" sort the results by use similarity and cuts, "filter" eliminates negatives scores but cuts randomly
            self.USE_method = self.hparams.USE_method
            # this is a factor for generating more candidates so we can score and prune them with USE
            self.USE_multiplier = self.hparams.USE_multiplier
            # print("SWITCH: Using USE")

        if self.hparams.stop_words:
            self.stop_words = self.all_stop_words
        else:
            self.stop_words = []

    #
    # here are some useful defaults
    #
    def get_default_hparams(self):
        return {
                'use_USE':True,
                'USE_method':"filter",          # filter cuts off negative similarity, rank deliver top scores
                'USE_multiplier':3,             # generate this times many more full alternatives to feed into USE
                'stop_words':True,              # filter out a long list of stop-words
                'perturb_words':2,              # how many words to replace
                'candidates_per_word':10,       # how many alternate words to pick from
                'total_alternatives':5,         # how many alternate samples to deliver back (max)
                'match_pos':True,               # use part-of-speech matching for replacements
                'leave_alone':0,                # don't perturb the top n most important words
                'random_out_of':0,              # if 0 , words are chosen by importance/stop_words/leave_alone
                'judge_bert':False              # if True, get the gradients for BERT output. If False, get them for model logits
                } 


    #
    # new APIs (v2.0)
    # Changes from 1.0: 
    # - no more words. SWITCH operates on BERT token ids now. 
    # - no more counter-fitted vectors. We are using the input embeddings of a tuned BERT model instead
    # - batches are now complete, and of predictable size - no more "we returned fewer than you asked for"
    # - more hparams passed to methods as arguments, that gives us more search flexibility
    #
    # - TBD: POS matching - not sure how we will do this with the new embeddings. We bascially need to make a new matrix. 
    #   We could base such a matrix on unigram or bigram POS ratings, anything else would get too big. 
    #   Alternatively, we could de-tokenize here to use NLTK, but I hate the thought. 
    #   if we record a new matrix, it should be done during fine-tuning.
    #

    def generate_alternative_tokens(self, token_ids, token_type_ids, attention_mask, disturb_id, hparams=None):
        """v2.0 API to generate altenate versions of a tokenized sample.

        Parameters:
        token_ids (torch.tensor)
            a pytorch batch tensor of tokenized samples
        token_type_ids (torch.tensor)
            would also be called segment_ids for BERT, and distinguishes text_a and text_b (usually generated by processor)
        attention_mask (torch.tensor)
            makes sure BERT pays no attention to padding (usually generated by processor)
        disturb_id (torch.tensor)
            the token_type_id that we will mess with. Could be the id for text_a or the one for text_b.
        hparams
            overrides for the hparams set at object creation

        Returns:
        (token_ids, token_type_ids, attention_mask)
            extended batch
        """
        pass

    def find_important_positions(self, token_ids, token_type_ids, attention_mask, disturb_id):
        """v2.0 internal method to find which tokens are most important to the classification.

        Parameters:
        token_ids (torch.tensor)
            a pytorch batch tensor of tokenized samples
        token_type_ids (torch.tensor)
            would also be called segment_ids for BERT, and distinguishes text_a and text_b (usually generated by processor)
        attention_mask (torch.tensor)
            makes sure BERT pays no attention to padding (usually generated by processor)
        disturb_id (torch.tensor)
            the token_type_id that we will mess with. Could be the id for text_a or the one for text_b.
        hparams
            overrides for the hparams set at object creation
        """

    def pick_most_similar_words_batch(self, token_ids, ret_count=10, match_pos=True):
        """ v2.0 internal method to produce plausible replacement tokens quickly.

        Parameters:
        token_ids (torch.tensor)
            todo: is this a list, or a compact tensor, or a sparse batch tensor, or a sparse, fully extended batch of samples
            tokens to find replacement for. non-zero token_ids will be expanded into replacements.
        """

        if match_pos:
            # todo
            pass

        pass

 


    #
    # this method works with the cosine nearest neighbor matrix to retrieve alternate words
    # it will match part-of-speech optionally
    # it shares a name with the TextFooler method, but works differently.
    #
    def pick_most_similar_words_batch(self, src_cfids, src_words, ret_count=10, match_pos=True):
    
        # if we are matching part-of-speech, get the pos info for the originals once
        pos_src = get_pos(src_words) if match_pos else []

        # get all the similar word index numbers. they are sorted in descending order.
        sim_order = self.cos_nn[src_cfids]
        #print("sim order", sim_order)

        # prepare a result to return
        sim_words = []

        # go through the sorted similarity matrix, one by one
        for idx, src_word in enumerate(src_words):

            # look up the word from the index
            sim_word_list = [self.idx2word[id] for id in sim_order[idx]]
            #print("sim word list", sim_word_list)

            # if desired, filter out words that don't match the part-of-speech of the original
            if match_pos:
                pos_new = get_pos(sim_word_list)
                sim_word_list = [sim_word_list[i] for i in range(len(sim_word_list)) if pos_new[i] == pos_src[idx]]
            
            # respect the maximum choice expected, and add the original so the list will never be empty
            sim_word_list = sim_word_list[:ret_count]
            sim_word_list.append(src_word)

            # add to results
            sim_words.append(sim_word_list)

        # done, get out of here
        return sim_words

    #
    # This method is for the vector plotting notebook
    # it merely returns a fixed number of replacement words (strings)
    #
    def get_replacement_words_for_display(self, word, count):
        # filter out words not in the counter-fitted vocabulary
        embeddings = self.word2idx.get(word, -1)
        if embeddings == -1:
            return []

        # ok, for every surviving word to be replaced, get some nearest neighbors
        replacements = self.pick_most_similar_words_batch([embeddings], [word], ret_count=count, match_pos=False)[0][:count]
        return replacements




    #
    # External API for SWITCH
    #

    def generate_candidates_from_example(self, example):
            
        # get the important words, also make a clean word list
        important_positions, important_token_positions, src = self.get_important_indices_from_example(example)
        # print("important positions: ", important_positions)
        # print("src:", src)

        # keep a clean string version for similarity scoring later
        sim_src = " ".join(src)

        # filter out words not in the counter-fitted vocabulary
        embeddings = [(self.word2idx.get(word, -1)) for word in src]

        # bunch of ugly code to filter the invalid positions and embeddings out at the same time
        # there is a more pythonic way to do this with zip and unzip, but this seemed more debuggable
        k = len(important_positions)
        count = 0
        get_words = max(self.hparams.perturb_words, self.hparams.random_out_of)
        ids = []
        positions = []
        words_to_perturb = []
        for i in range(k):
            # was the word at this position not in vocab, or a stop-word?
            # note: important_positions[i] == -1 should not really happen anymore
            if important_positions[i] != -1 and embeddings[important_positions[i]] != -1:
                ids.append(embeddings[important_positions[i]])
                positions.append(important_positions[i])
                words_to_perturb.append(src[important_positions[i]])

                # count the valid positions, stop when we reached what the caller asked for
                count += 1
                if count >= get_words:
                    break

        #print("important words: ", words_to_perturb)

        # Keep the front of the list. If desired, shuffle first. 
        if self.hparams.random_out_of > 0:
            shuffle(words_to_perturb)
        words_to_perturb = words_to_perturb[:self.hparams.perturb_words]


        # ok, for every surviving word to be replaced, get some nearest neighbors
        replacements = self.pick_most_similar_words_batch(ids, words_to_perturb)
        #print("replacements:", replacements)

        # Now use the new alternative words to randomly make replacement sentences
        results = []
        sim_results = []
        for k in range(self.hparams.total_alternatives if self.USE is None else self.hparams.total_alternatives*self.USE_multiplier):
            # start with the original sentence
            result = src.copy()

            # replace every important word with a random choice of alternative
            for i, position in enumerate(positions):
                result[position] = replacements[i][np.random.randint(len(replacements[i]))]

            # add it to the result list
            results.append(" ".join(result))
            if self.USE is not None:
                sim_results.append(" ".join(result))


        # score and trim by semantic similarity with original (with USE)
        if self.USE is not None:
            # score USE similarities
            semantic_sims = self.USE.semantic_sim([sim_src] * len(sim_results),sim_results)[0]
            # print("semantic similarities:", semantic_sims)

            # USE rank/filter
            if self.USE_method == "rank":
                # sort the results by semantic similarity to the original, best first
                idx = list(np.argsort(semantic_sims))[::-1]
            else :
                # filter out the indices with negative scores, otherwise don't touch it
                idx = [i for i in range(len(semantic_sims)) if semantic_sims[i] > 0]


            # if the original is among the alternatives, take it out.
            # print("best result:", sim_results[idx[0]])'
            if len(idx) > 0 and sim_results[idx[0]] == sim_src: 
                del idx[0]
                #print("after correction for identity:", sim_results[idx[0]])

            # take no more than the requested number
            idx = idx[:self.hparams.total_alternatives]
            # pick the surviving sentences 
            results = [results[i] for i in idx]
            

        # done
        return results


    def get_important_indices_from_example(self, example, input_ids=None, token_type_ids=None, attention_mask=None):
        # in entailment/neutral/contradiction scenarios, it might make sense
        # to not feed the premise into BERT when the supposed score is "neutral"

        # if we didn't get the tokenized features, make them here
        # FVE will pass them in, having done this already

        # print("SWITCH: get_imp: ex:",example)
        if input_ids is None:
            input_ids, attention_mask, token_type_ids, _ = self.processor.create_feature_tensors([example], self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)

        # now we have to establish where the thing to be perturbed starts
        if self.perturb_text_b:
            start_perturb = input_ids[0].tolist().index(102)+1 # look for sentence_end token, go one further
            text_to_perturb = example.text_b
        else:
            start_perturb = 1
            text_to_perturb = example.text_a
        
        # clean and split the text
        word_list = self.text_to_list(text_to_perturb)

        # in the next block, we will tokenize text_b word for word to figure out how it maps to tokens
        # prepare a token-position to word-position map
        word_positions = [] 
        # get the two special bos and eos token, squirrel the latter one away for later
        for i, word in enumerate(word_list):
            token_ids = self.tokenizer.encode(word, add_special_tokens = False)
            word_positions += [i]*len(token_ids) # for each word, add as many index positions as it has tokens
        word_positions.append(-1) # one for the eos token
        # ok we now have that map

        # back to the input_ids and other stuff we encoded from the whole example
        # get a tensor of the actual embeddings, and detach it from its computation graph
        bert = self.model.bert if hasattr(self.model,"bert") else self.model
        inputs_embeds = bert.get_input_embeddings()(input_ids).detach().to(self.device)
        inputs_embeds.requires_grad=True

        # get the model ready. remember whether it was in training mode
        training = self.model.training
        if training:
            self.model.eval()

        # run model forward and backward with gradients computed
        with torch.enable_grad():
            self.model.zero_grad()
            if self.hparams.judge_bert:
                #print("judging with bert")
                outputs = bert(input_ids=None,inputs_embeds=inputs_embeds,
                                    token_type_ids=token_type_ids, attention_mask=attention_mask)[0][0]
            else:
                outputs = self.model(input_ids=None,inputs_embeds=inputs_embeds,
                                    token_type_ids=token_type_ids, attention_mask=attention_mask, 
                                    extend=False)[0]
            outputs.backward(outputs)

        # restore the training status
        if training:
            self.model.train()
        
        # get the gradient lengths
        grad_norms = inputs_embeds.grad.norm(dim=2, p=2).cpu()[0]
        # set them to zero anywhere but in text_b
        #print(grad_norms)
        grad_norms.narrow(0,0,start_perturb).zero_()
        #print(grad_norms)
        # now argsort to get the indices of the longest grad vectors to the front
        important_token_positions = grad_norms.argsort(descending=True).tolist()
        important_token_positions = [i if grad_norms[i] != 0 else -1 for i in important_token_positions]
        # translate them back to word positions
        important_positions = [word_positions[i-start_perturb] \
                if i != -1 and (i-start_perturb) < len(word_positions) else -1 \
                for i in important_token_positions]

        #print ("SWITCH: gradients used successfully")

        # print("start:", start_perturb)
        # print("wpos",word_positions)
        # print("ipos",important_positions)
        # print("tpos", important_token_positions)
        # print(word_list)

        # filter results ({'leave_alone':n}, {'stop_words':True})
        count = self.hparams.leave_alone
        for i in range(len(important_positions)):
            # filter out stop_words
            word = word_list[important_positions[i]]
            if word in self.stop_words:
                important_token_positions[i] = -1
                important_positions[i] = -1
                continue

            # zero (well, -1) out the first n important items (for leave_alone)
            if important_positions[i] != -1 and count > 0:
                important_token_positions[i] = -1
                important_positions[i] = -1
                count -= 1
                continue

        return important_positions, important_token_positions, word_list
    
    
    
    #
    # parses one line from a sample corpus
    # returns a list of words, and an int label
    # MR is for the IMDB Movie Review corpus
    # adapted from TextFooler code (see below)
    #

    def text_to_list(self, line, clean=True, MR=True, encoding='utf8', lower=True, text_only=True):
        # parse out label and text data
        if text_only:
            text = line
            label = 0
        else:
            if MR:
                label,sep, text = line.partition(' ')
                label = int(label)
            else:
                label, sep, text = line.partition(',')
                label = int(label) - 1

        # clean and covert to lower case
        
        if clean:
            text = clean_str(text.strip()) if clean else text.strip()

        if lower:
            text = text.lower()
            
        # return text as list, label as int
        if text_only:
            return  text.split()
        else:
            text.split(), label



    def get_hparam(self, name):
        return self.hparams[name]

    def set_hparam(self, name, value):
        hp = {name:value}
        # todo: implement this here
        # and then:
        self.switch.set_hparam(name, value)



#
# Utility functions
#

def get_stopwords():
    return ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', \
            'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', \
            'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', \
            'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', \
            'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', \
            "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', \
            'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', \
            "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', \
            'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn',
            "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', \
            'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', \
            'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', \
            'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other',\
            'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", \
            'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', \
            'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', \
            'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', \
            'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', \
            "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', \
            'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', \
            'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', \
            "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']


def load_cos_nn(force_create=False):
    """ load_cos_nn (precomputed nearest neighbors per word)
    cos_nn.npy, word_norms.txt -> cos_nn, idx2word, word2idx, idx2norm if the files exists
    otherwise it creates them."""

    #print(os.getcwd())
    if os.path.exists("resources/firebert_cos_nn_100.npy") and os.path.exists("resources/firebert_word_norms.txt") and not force_create:
        #print('SWITCH: Loading pre-computed cosine neighborhood matrix and norms list ...')
        cos_nn = np.load("resources/firebert_cos_nn_100.npy")

        idx2word = []
        idx2norm = []
        word2idx= {}
        with open(os.path.join("resources/firebert_word_norms.txt"), "r", encoding="utf-8") as ifile:
            for i,line in enumerate(ifile):
                words = line.split()
                idx2word.append(words[0])
                idx2norm.append(float(words[1]))
                word2idx[words[0]] = i
                
    else:
        print("SWITCH: Creating neighborhood and word-norm files ...")
        print("SWITCH: Loading the cos sim matrix and vocabulary ...")
        cos_sim = load_cos_sim()
        idx2word, word2idx, idx2norm = load_vocab()

        print('SWITCH: Computing the cosine neighborhood matrix nand word norms (a one-time process)...')
        nns = []
        norms = []
        nn_count = 100

        for i, word in enumerate(idx2word):
            # make one list of words and their norm, in index order
            norms.append([word,idx2norm[i]])

            # and here a vector for each word, of its 10 nearest neighbors
            nn = np.argpartition(cos_sim[i,:], -nn_count-1)[-nn_count-1:]
            nn = sorted(nn, key=lambda x: cos_sim[i,x], reverse=True)[1:]
            nns.append(nn)

        # save cos nn file
        cos_nn = np.array(nns)
        print("SWITCH: Saving cos nn matrix ...")
        np.save("resources/firebert_cos_nn.npy_100", cos_nn)

        # save norms file
        print("SWITCH: Saving word norms ...")
        with open(os.path.join("resources/firebert_word_norms.txt"), 'w', encoding="utf-8") as ofile:
            for line in norms:
                ofile.write(line[0]+" "+str(line[1])+"\n")

        # get back some memory
        cos_sim = None
        nns = None
        norms = None
        gc.collect()
        print("SWITCH: Cos neighborhood matrix and norms done.")

    return cos_nn, idx2word, word2idx, idx2norm





"""
The following code was adapted from code by TextFooler at https://github.com/jind11/TextFooler,
a code repository in support of the paper:
Jin, Di, et al. "Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment."
arXiv preprint arXiv:1907.11932 (2019).
"""


#
# Helper class - Universal Sentence Encoder (Cer et al. 2019)
#


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path

        # QUESTIONABLE! Necessary for Mac, should not affect our TF-GPU environments
        if not torch.cuda.is_available():
            os.environ['KMP_DUPLICATE_LIB_OK']='True'

        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


#
# utilitiy loading functions
# load vocab: counter-fitted-vectors.txt -> idx2word, word2idx, idx2norm
#

def load_vocab():
    idx2word = []
    word2idx={}
    idx2norm=[]

    print("SWITCH: Building vocab and norms ...")
    with open("embeddings/counter-fitted-vectors.txt", 'r') as ifile:
        next_slot = 0
        for line in ifile:
            entries = line.strip().split()
            word = entries[0]
            if word not in idx2word: # todo: do we need this check? Might just as well have the last instance of the word
                # make the two-way mapping
                idx2word.append(word)
                word2idx[word] = next_slot

                # calculate the vector length, and keep it
                embeddings = [float(num) for num in entries[1:]]
                norm = np.linalg.norm(np.array(embeddings))
                idx2norm.append(norm)

                next_slot += 1

    return idx2word, word2idx, idx2norm



#
# string cleaning
#

def clean_str(string):
    """
    Tokenization/string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
    


#
# load_cos_sim
# cos_sim_counter_fitting.npy -> cos_sim if the former exists
# counter-fitted-vectors.txt -> cos_sim -> cos_sim_counter_fitting.npy otherwise
#

def load_cos_sim(force_create=False):
    if os.path.exists("cos_sim_counter_fitting.npy") and not force_create:
        # load pre-computed cosine similarity matrix if provided
        print('SWITCH: Load pre-computed cosine similarity matrix from {} ...'
                .format("cos_sim_counter_fitting.npy"))
        cos_sim = np.load("cos_sim_counter_fitting.npy")
    else:
        # calculate the cosine similarity matrix
        print('SWITCH: Computing the cosine similarity matrix ...')
        embeddings = []
        # todo: don't read that whole file again here
        with open("embeddings/counter-fitted-vectors.txt", 'r') as ifile:
            for line in ifile:
                words = line.strip().split()
                embedding = [float(num) for num in words[1:]]
                embeddings.append(embedding)

        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
        print("SWITCH: Saving cos sim matrix ..")
        np.save("cos_sim_counter_fitting.npy", cos_sim)

    print("SWITCH: Cos sim matrix finished!")
    return cos_sim


#
# Part-of-speech tagging##
#  
def get_pos(sent, tagset='default'):
    '''
    :param sent: list of word strings
    tagset: {'universal', 'default'}
    :return: list of pos tags.
    Universal (Coarse) Pos tags has  12 categories
        - NOUN (nouns)
        - VERB (verbs)
        - ADJ (adjectives)
        - ADV (adverbs)
        - PRON (pronouns)
        - DET (determiners and articles)
        - ADP (prepositions and postpositions)
        - NUM (numerals)
        - CONJ (conjunctions)
        - PRT (particles)
        - . (punctuation marks)
        - X (a catch-all for other categories such as abbreviations or foreign words)
    '''

    if len(sent) == 0:
        return []

    if tagset == 'default':
        word_n_pos_list = nltk.pos_tag(sent)
    elif tagset == 'universal':
        word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
    _, pos_list = zip(*word_n_pos_list)
    return pos_list




#
# testing and debugging code
#
            
def test_SWITCH():
    print("SWITCH tests")
    print()

    # Load pretrained model and tokenizer
    # set model task and training config
    model_name_or_path = 'bert-base-uncased'

    config = BertConfig.from_pretrained(
        model_name_or_path,
        num_labels=3,
        finetuning_task="MNLI",
        cache_dir=None
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_name_or_path,
        do_lower_case=True,
        cache_dir=None
    )
    bert_model = BertModel.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=None
    )

    print("Test: instantiating object")
    inst = SWITCH(tokenizer=tokenizer, model=bert_model)
    print("instance:", inst)
    print()
 
    print("Test: clean_str on a messy string:")
    s = "skldjfsdlkfjslkjg, mvdvawe,.mf /aweklf"
    print("source:", s)
    print("clean:", clean_str(s))
    print()

    print("Test: transform text into clean word list:")
    text = """The house itself feels like a mousetrap, which works for a narrative puzzle in which the parts keep shifting 
    as the wood-paneled walls close in. The overall sense of confinement is perfect for the aims of a private investigator, 
    Benoit Blanc, a honey-baked ham played by Daniel Craig with grandiose self-regard and a Southern accent that seems borrowed 
    from Kevin Spacey. There isn’t a butler in the parlor, but there is a rather too virtuous caretaker, Marta Cabrera 
    (Ana de Armas), who worked for the manor’s imperious patriarch Harlan Thrombey (Christopher Plummer), who suddenly 
    and rather flamboyantly croaks."""
    words = inst.text_to_list(text)
    print("source:", text)
    print("list:", words)
    
    print("Test: generate_candidates")
    data = inst.generate_candidates(words)
    print("alternatives:", data)
    print()
    
    print("Test: generate_candidates from text")
    data = inst.generate_candidates(text)
    print("alternatives:", data)
    print()
    print()
    print()

    text = "This movie is truly fun for the whole family. Adults and kids will totally enjoy it!"
    print("Test: generate_candidates from short text")
    print("text:", text)
    print("---------------------")
    data = inst.generate_candidates(text)
    print("alternatives:")
    for a in data:
        print(a)
    print()


    text = "The rule has data collection requirements which aid the EPA to realize their emission control goals."
    print("Test: generate_candidates from short text")
    print("text:", text)
    print("---------------------")
    data = inst.generate_candidates(text)
    print("alternatives:")
    for a in data:
        print(a)
    print()
    
    print("Tests complete.")

    return


def debug():
    from processors import MnliProcessor, ImdbProcessor
    from firebert_fse import FireBERT_FSE

    print("Simple SWITCH tests (through FSE)")
    # create a FireBERT_FVE classifier for MNNLI

    # prepare hyperparameters
    hparams = {
                'batch_size': 32, 
                'leave_alone':0, 
                'random_out_of':0, 
                'judge_bert':False
                }

    # now instantiate the model
    model = FireBERT_FSE(
                        load_from='resources/models/MNLI/pytorch_model.bin', 
                        processor=MnliProcessor(), 
                        hparams=hparams
                        )
    
    print()
    print()
    print()
    data = [
        {'premise':"Conceptually cream skimming has two basic dimensions - product and geography.",
        'hypothesis':"Product and geography are what make cream skimming work.",
        'label':"neutral"},

        {'premise':"He writes that it 's the first time he 's added such a track .",
        'hypothesis':"This is the first time he 's added such a track .",
        'label':"neutral"},

        {'premise':"The new rights are nice enough .",
        'hypothesis':"Everyone really likes the newest benefits .",
        'label':"neutral"},

        {'premise':"This site includes a list of all award winners and a searchable database of Government Executive articles.",
        'hypothesis':"The Government Executive articles housed on the website are not able to be searched.",
        'label':"contradiction"},

        {'premise':"yeah i i think my favorite restaurant is always been the one closest  you know the closest as long as it's it meets the minimum criteria you know of good food",
        'hypothesis':"My favorite restaurants are always at least a hundred miles away from my house.",
        'label':"contradiction"},

        {'premise':"Calcutta seems to be the only other production center having any pretensions to artistic creativity at all, but ironically you're actually more likely to see the works of Satyajit Ray or Mrinal Sen shown in Europe or North America than in India itself.",
        'hypothesis':"Most of Mrinal Sen's work can be found in European collections.",
        'label':"neutral"}
    ]
    
    def top_n(index, words, n):
        result = [words[i] for i in index if i != -1]
        result = [word for word in result if word not in model.switch.stop_words]
        return result[:n]

    for d in data:
        # make a "feature" tensor out of those
        example, input_ids, attention_mask, token_type_ids, label = \
            model.processor.make_single_example_with_features(d["premise"], d["hypothesis"], d["label"])

        # use SWITCH to figure out word importance within the list
        word_indices, token_indices, word_list = \
            model.switch.get_important_indices_from_example(example, input_ids, token_type_ids, attention_mask)

        print("Premise:",d["premise"])
        print("Original hypothesis:", d["hypothesis"], "(original label: ", d['label'], ")")

        print("Top 5 hypothesis words (new):", top_n(word_indices, word_list, 5))
        print()

    print()

    print()
    print("IMDB test")
    print()

    # prepare hyperparameters
    hparams = {
                'batch_size': 32, 
                'leave_alone':0, 
                'random_out_of':0, 
                'judge_bert':False,
                'perturb_words':2
                }

    # now instantiate the model
    model = FireBERT_FSE(
                        load_from='resources/models/IMDB/pytorch_model.bin', 
                        processor=ImdbProcessor(), 
                        hparams=hparams
                        )
    
    text = "This movie is truly fun for the whole family. Adults and kids will totally enjoy it!"
    label = 1
    
    # make a "feature" tensor out of those
    example, input_ids, attention_mask, token_type_ids, label = \
        model.processor.make_single_example_with_features(text, None, label)

 
    texts = model.switch.generate_candidates_from_example(example)
    print(text)
    for t in texts:
        print(":", t)
    print()


    # prepare new hyperparameters
    hparams = {
                'batch_size': 32, 
                'leave_alone':0, 
                'random_out_of':0, 
                'judge_bert':False,
                'perturb_words':5
                }

    # make the model use the new hparams
    model.update_hparams(hparams)

    texts = model.switch.generate_candidates_from_example(example)
    print(text)
    for t in texts:
        print(":", t)




if __name__ == "__main__":
    debug()
    #test_SWITCH()

