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
# processors

from transformers.data.processors.utils import (
    DataProcessor, 
    InputExample, 
    InputFeatures
)

from transformers import (
    BertTokenizer,
    glue_convert_examples_to_features as convert_examples_to_features
)

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import os
import argparse
import logging
import random

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    
    def __init__(self,hparams={}):
        super(MnliProcessor, self).__init__()

        self.logger = logging.getLogger(__name__)
        hdict = self.get_default_hparams()
        hdict.update(hparams)
        self.hparams = argparse.Namespace(**hdict)
        self.all_examples = []

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    
    def get_train_examples_from_original(self, data_dir):
        return self._create_examples_from_original(
            self._read_tsv(os.path.join(data_dir, "original", "multinli_1.0_train.txt")), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")
    
    def get_dev_examples_from_original(self, data_dir):
        return self._create_examples_from_original(
            self._read_tsv(os.path.join(data_dir, "original", "multinli_1.0_dev_matched.txt")), "dev_matched")    
    
    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")
    
    def get_test_examples_from_original(self, data_dir):
        return self._create_examples_from_original(
            self._read_tsv(os.path.join(data_dir, "original", "multinli_1.0_dev_mismatched.txt")), "dev_mismatched")        

    def get_adv_train_examples(self, data_dir):
        return self._create_examples_from_original(
            self._read_tsv(os.path.join(data_dir, "generated/mnli_adversarial_samples_for_train.txt")), "adv_train")
    
    def get_adv_dev_examples(self, data_dir):
        return self._create_examples_from_original(
            self._read_tsv(os.path.join(data_dir, "generated/mnli_adversarial_samples_for_dev.txt")), "adv_dev")

    def get_adv_test_examples(self, data_dir):
        return self._create_examples_from_original(
            self._read_tsv(os.path.join(data_dir, "generated/mnli_adversarial_samples_for_test.txt")), "adv_test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_text_to_perturb(self, example):
        return example.text_b

    def get_text_to_perturb_is_b(self, example):
        return True

    def get_default_hparams(self):
        hparams = {'num_labels': 3,
                    'max_seq_length': 128,
                    'task': 'mnli',
                    'output_mode': 'classification',
                    'sample_percent': 100,
                    'randomize': True
                }
        return hparams
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def create_feature_tensors(self, examples, device=None):
        features = self._create_features(examples)
        # Convert to Tensors
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device = device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long, device = device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long, device= device)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long, device=device)
        
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


    def create_feature_tensors_with_example(self, examples, device=None):
        features = self._create_features(examples)
        # Convert to Tensors
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device = device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long, device = device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long, device= device)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long, device=device)

        all_idxs = torch.tensor([i for i in range(len(examples))], dtype=torch.long)      
        
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_idxs


    
    def create_adversarial_examples(self, example, alternates):
        examples = []
        idx = []
        for (i, text_b) in enumerate(alternates):
            guid = "%s-%s" % (example.guid, i)
            label = example.label
            text_a = example.text_a
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            idx.append(i)

        return examples, idx

    # helper method to run create feature tensor for single example
    def make_single_example_with_features(self, text_a, text_b, label, device=None):
    
        example = InputExample("0", text_a, text_b, str(label))
        input_ids, attention_mask, token_type_ids, label = self.create_feature_tensors([example], device=device)
        idx = len(self.all_examples)
        self.all_examples.append(example)
        return example, input_ids, attention_mask, token_type_ids, label, idx


    def _create_features(self, examples):
        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.hparams.max_seq_length,
            output_mode=self.hparams.output_mode,
            pad_on_left=False,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        return features

    def _create_examples(self, lines, set_type):
        parenthesis_table = str.maketrans({'(': None, ')': None})
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if label == '-':
                continue
            # Remove '(' and ')' from the premise and hypothesis.
            text_a = text_a.translate(parenthesis_table)
            text_b = text_b.translate(parenthesis_table)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_from_original(self, lines, set_type):
        parentheses_table = str.maketrans({'(': None, ')': None})
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[0]
            if label == '-':
                continue
            # Remove '(' and ')' from the premises and hypotheses.
            text_a = text_a.translate(parentheses_table)
            text_b = text_b.translate(parentheses_table)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def load_and_cache_examples(self, data_dir, overwrite_cache=False, example_set='dev', use_original=False):
        # Load data features from cache or dataset file
        # or create the cache

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                example_set,
                "bert-base-uncased",
                str(self.hparams.max_seq_length),
                str(self.hparams.task) + ("-original" if use_original else ""),
            ),
        )
        if use_original:
            examples = (
                self.get_dev_examples_from_original(data_dir) if (example_set=='dev') else
                self.get_train_examples_from_original(data_dir) if (example_set=='train') else 
                self.get_test_examples_from_original(data_dir)
            )
        else:
            examples = (
                self.get_dev_examples(data_dir) if (example_set=='dev') else
                self.get_train_examples(data_dir) if (example_set=='train') else 
                self.get_adv_train_examples(data_dir) if (example_set=='adv_train') else
                self.get_adv_dev_examples(data_dir) if (example_set=='adv_dev') else
                self.get_adv_test_examples(data_dir) if (example_set=='adv_test') else
                self.get_test_examples(data_dir)
            )
        if os.path.exists(cached_features_file) and not overwrite_cache:
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            self.logger.info("Creating features from dataset file at %s", data_dir)
            features = self._create_features(examples)
            self.logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        if self.hparams.sample_percent != 100:
            value = self.hparams.sample_percent
            # random % subset
            print("FireBERT processor: Using sampled input, setting: ",
                    abs(value) if value != 0 else 1, 
                    "%" if value > 0 else "",\
                    "samples")
            all_data = list(zip(features, examples))
            if value > 0:
                num_samples = (len(examples) * value) // 100 # percent
            elif value < 0:
                num_samples = -value # absolute number
            else:
                num_samples = 1 # 0 means 1 sample

            if self.hparams.randomize:
                some_data = random.sample(all_data, num_samples)
            else:
                some_data = all_data[:num_samples]
            features, examples = zip(*some_data)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        all_idxs = torch.tensor([i for i in range(len(examples))], dtype=torch.long)
        assert (len(all_input_ids) == len(all_idxs))
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_idxs)

        return dataset, examples

   


    def _create_features(self, examples):
        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.hparams.max_seq_length,
            output_mode=self.hparams.output_mode,
            pad_on_left=False,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        return features

    def _create_examples(self, lines, set_type):
        parenthesis_table = str.maketrans({'(': None, ')': None})
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if label == '-':
                continue
            # Remove '(' and ')' from the premise and hypothesis.
            text_a = text_a.translate(parenthesis_table)
            text_b = text_b.translate(parenthesis_table)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_from_original(self, lines, set_type):
        parentheses_table = str.maketrans({'(': None, ')': None})
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[0]
            if label == '-':
                continue
            # Remove '(' and ')' from the premises and hypotheses.
            text_a = text_a.translate(parentheses_table)
            text_b = text_b.translate(parentheses_table)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


 #
 # IMDB starts here
 #

class ImdbProcessor(DataProcessor):
    """Processor for the IMDB MovieReview dataset (Stanford version)."""

    def __init__(self,hparams={}):
        super(ImdbProcessor, self).__init__()

        self.logger = logging.getLogger(__name__)
        hdict = self.get_default_hparams()
        hdict.update(hparams)
        self.hparams = argparse.Namespace(**hdict)
        self.all_examples = []


    def get_default_hparams(self):
        hparams = {'num_labels': 2,
                    'max_seq_length': 256,
                    'task': 'imdb',
                    'output_mode': 'classification',
                    'sample_percent': 100,
                    'randomize': True
                }
        return hparams

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            _read_corpus(os.path.join(data_dir, "imdb_train.txt"), MR=True, clean=False, shuffle=True), "train")

    def get_adv_train_examples(self, data_dir):
        return self._create_examples(
            _read_corpus(os.path.join(data_dir, "generated/imdb_adversarial_samples_for_train.txt")), "adv_train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            _read_corpus(os.path.join(data_dir, "imdb_dev.txt"), MR=True, clean=False, shuffle=True), "dev")

    def get_adv_dev_examples(self, data_dir):
        return self._create_examples(
            _read_corpus(os.path.join(data_dir, "generated/imdb_adversarial_samples_for_dev.txt")), "adv_dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            _read_corpus(os.path.join(data_dir, "imdb_test.txt"), MR=True, clean=False, shuffle=True), "test")

    def get_adv_test_examples(self, data_dir):
        return self._create_examples(
            _read_corpus(os.path.join(data_dir, "generated/imdb_adversarial_samples_for_test.txt")), "adv_test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def get_text_to_perturb(self, example):
        return example.text_a

    def get_text_to_perturb_is_b(self, example):
        return False


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_and_cache_examples(self, data_dir, overwrite_cache=False, example_set='dev'):
        # Load data features from cache or dataset file
        # or create the cache

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                example_set,
                "bert-base-uncased",
                str(self.hparams.max_seq_length),
                str(self.hparams.task),
            ),
        )
        examples = (
            self.get_adv_train_examples(data_dir) if (example_set=='adv_train') else
            self.get_adv_dev_examples(data_dir) if (example_set=='adv_dev') else
            self.get_adv_test_examples(data_dir) if (example_set=='adv_test') else
            self.get_dev_examples(data_dir) if (example_set=='dev') else
            self.get_train_examples(data_dir) if (example_set=='train') else 
            self.get_test_examples(data_dir)
        )
        if os.path.exists(cached_features_file) and not overwrite_cache:
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            self.logger.info("Creating features from dataset file at %s", data_dir)
            features = self._create_features(examples)
            self.logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        
        value = self.hparams.sample_percent 
        if value != 100:
            # random % subset
            print("FireBERT processor: Using sampled input, setting: ",
                    abs(value) if value != 0 else 1, 
                    "%" if value > 0 else "",\
                    "samples")
            all_data = list(zip(features, examples))
            num_samples = (len(examples) * self.hparams.sample_percent) // 100
            if self.hparams.randomize:
                some_data = random.sample(all_data, num_samples)
            else:
                some_data = all_data[:num_samples]
            features, examples = zip(*some_data)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        all_idxs = torch.tensor([i for i in range(len(examples))], dtype=torch.long)
        assert (len(all_input_ids) == len(all_idxs))
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_idxs)

        return dataset, examples

    def create_feature_tensors(self, examples, device=None):
        features = self._create_features(examples)
        # Convert to Tensors
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long, device=device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long, device=device)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long) 
        
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


    def create_feature_tensors_with_example(self, examples, device=None):
        features = self._create_features(examples)
        # Convert to Tensors
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device = device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long, device = device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long, device= device)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long, device=device)

        all_idxs = torch.tensor([i for i in range(len(examples))], dtype=torch.long)      
        
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_idxs
    

    def create_adversarial_examples(self, example, alternates):
        examples = []
        idx = []
        for i, text in enumerate(alternates):
            guid = "%s-%s" % (example.guid, i)
            label = example.label
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
            idx.append(i)
        return examples, idx
    

    # helper method to run create feature tensor for single example
    def make_single_example_with_features(self, text_a, text_b, label, device=None):
    
        example = InputExample("0", text_a, text_b, label)
        input_ids, attention_mask, token_type_ids, label = self.create_feature_tensors([example], device=device)
        idx = len(self.all_examples)
        self.all_examples.append(example)
    
        return example, input_ids, attention_mask, token_type_ids, label, idx


    def _create_features(self, examples):
        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.hparams.max_seq_length,
            output_mode=self.hparams.output_mode,
            pad_on_left=False,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        return features

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        
        texts, labels = lines
        examples = []
        for i, (text,label) in enumerate(zip(texts, labels)):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples


#
# helper functions from TextFooler
#

def _clean_str(string, TREC=False):
    import re
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
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
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def _read_corpus(path, clean=True, MR=True, encoding='utf8', shuffle=False, lower=True):
    data = []
    labels = []
    with open(path, encoding=encoding) as fin:
        for line in fin:
            if MR:
                label, sep, text = line.partition(' ')
                label = int(label)
            else:
                label, sep, text = line.partition(',')
                label = int(label) - 1
            if clean:
                text = _clean_str(text.strip()) if clean else text.strip()
            if lower:
                text = text.lower()
            labels.append(label)
            data.append(text)

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]

    return data, labels



#
# test
#

def test():
    import numpy
    
    import torch
    import pytorch_lightning as pl

    from processors import ImdbProcessor
    from bert_base_model import LightningBertForSequenceClassification

    # prepare hyperparameters

    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    weight_decay = 0.0
    adam_epsilon = 1e-8

    max_steps = -1 # if -1 then calculate number of training steps based on the length of the train set
    warmup_steps = 0

    num_train_epochs = 5
    batch_size = 16

    len_train_set = 40000

    if max_steps > 0:
        num_train_epochs = max_steps // (len_train_set // gradient_accumulation_steps) + 1
        num_training_steps = max_steps
    else:
        num_training_steps = len_train_set // gradient_accumulation_steps * num_train_epochs

    hparams = { 'learning_rate': learning_rate,
                'adam_epsilon': adam_epsilon,
                'weight_decay': weight_decay,
                'warmup_steps': warmup_steps,
                'num_training_steps': num_training_steps,
                'batch_size': batch_size,
            }

    model = LightningBertForSequenceClassification(processor=ImdbProcessor(), hparams=hparams)  
    processor = model.get_processor()
    train_dataset, _ = processor.load_and_cache_examples("data/IMDB", example_set='train')
    val_dataset, _ = processor.load_and_cache_examples("data/IMDB", example_set='dev')
    test_dataset, _ = processor.load_and_cache_examples("data/IMDB", example_set='test')

    model.set_train_dataset(train_dataset)
    model.set_val_dataset(val_dataset)
    model.set_test_dataset(test_dataset)


    from pytorch_lightning.logging import TensorBoardLogger

    save_root_path ='models/IMDB_on_lightning/'
    tensor_logger = TensorBoardLogger(save_dir= save_root_path + 'logs', version=10, name='imdb_finetuning')
    checkpoint_save_path = save_root_path + 'checkpoints/'

    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_save_path,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    amp_opt_level='O1' # https://nvidia.github.io/apex/amp.html#opt-levels
    max_grad_norm = 1.0

    trainer = pl.Trainer(default_save_path=checkpoint_save_path, logger=tensor_logger, gpus=(-1 if torch.cuda.is_available() else None),
                        max_epochs = num_train_epochs, amp_level=amp_opt_level, gradient_clip_val=max_grad_norm,
                        checkpoint_callback=checkpoint_callback)

    trainer.fit(model)




if __name__ == "__main__":
    import numpy as np
    import pytorch_lightning as pl
    from processors import MnliProcessor

    test()
    print("tests done")
