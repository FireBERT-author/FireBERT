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

import os
import argparse
import logging
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    BertConfig,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    glue_convert_examples_to_features as convert_examples_to_features
)

# this is ours
import processors

class LightningBertForSequenceClassification(pl.LightningModule):

    def __init__(self, load_from=None, processor=None, hparams={}):
        super(LightningBertForSequenceClassification, self).__init__()

        self.logger = logging.getLogger(__name__)


        self.train_dataset = None
        self.train_examples = None
        self.val_dataset = None
        self.val_examples = None
        self.test_dataset = None
        self.test_examples = None


        # merge passed hparams over default hparams
        hdict = self.get_default_hparams()
        hdict.update(processor.get_default_hparams())
        hdict.update(hparams)
        self.hparams = argparse.Namespace(**hdict)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_mode = "classification"

        # Load pretrained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )
        config =  BertConfig.from_pretrained(
            'bert-base-uncased',
            num_labels=self.hparams.num_labels,
            finetuning_task=self.hparams.task
        )
        self.bert = BertModel.from_pretrained( 
            'bert-base-uncased',
            config=config
        )
        self.processor = processor
        self.processor.set_tokenizer(self.tokenizer)
        
        self.config = self.bert.config
        
        # here are two things that are different between IMDB and MNLI
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.max_seq_length = self.hparams.max_seq_length

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.set_batch_size(self.hparams.batch_size)

        # load a specific saved model?
        if load_from is not None:
            checkpoint = torch.load(load_from, map_location=self.device)
            self.load_state_dict(checkpoint)
   

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, example_idx=None, extend=False):

        h, p = self.bert(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
        p = self.dropout(p)
        logits = self.classifier(p)

        return logits

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label, example_idx = batch
         
        # fwd
        y_hat = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, example_idx=example_idx)
        
        # loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))
        
        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}



    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label, example_idx = batch
        #print("input ids shape: ",input_ids.shape)

        # fwd
        y_hat = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, example_idx=example_idx)
        
        # print("y hat shape:",y_hat.shape)
        # print("label shape:",label.shape)
        # print("y hat view shape", (y_hat.view(-1, self.num_labels)).shape)
        # loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))
        
        # accuracy
        a, y_hat = torch.max(y_hat, dim=1)
        
        val_acc = accuracy_score(label.cpu(), y_hat.cpu())
        val_acc = torch.tensor(val_acc)
       
        val_f1 = f1_score(label.cpu(), y_hat.cpu(), average='macro')
        val_f1 = torch.tensor(val_f1)

        return {'val_loss': loss, 'val_acc': val_acc, 'val_f1': val_f1}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc, 'avg_val_f1': avg_val_f1}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

        
    def test_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label, example_idx = batch
        # fwd
        y_hat = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, example_idx=example_idx)
        # accuracy
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(label.cpu(), y_hat.cpu())
        test_f1 = f1_score(label.cpu(), y_hat.cpu(), average='macro')
        
        return {'test_acc': torch.tensor(test_acc), 'test_f1': torch.tensor(test_f1)}

    def test_end(self, outputs):
        
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_test_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc, 'avg_test_f1': avg_test_f1}
        return {'avg_test_acc': avg_test_acc, 'avg_test_f1': avg_test_f1, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        #return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.hparams.num_training_steps)
        
        return [optimizer], [scheduler]
                                                    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def set_train_dataset(self, train_dataset, examples=None):
        self.train_dataset = train_dataset
        self.train_examples = examples
        self.all_examples = examples

    def set_val_dataset(self, val_dataset, examples=None):
        self.val_dataset = val_dataset
        self.val_examples = examples
        self.all_examples = examples
        
    def set_test_dataset(self, test_dataset, examples=None):
        self.test_dataset = test_dataset
        self.test_examples = examples
        self.all_examples = examples
    
    #@pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    #@pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    #@pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)


    def get_default_hparams(self):
        gradient_accumulation_steps = 1
        learning_rate = 5e-5
        weight_decay = 0.0
        adam_epsilon = 1e-8

        max_steps = -1 # todo: consider removing "if" below
        warmup_steps = 0
        num_train_epochs = 1
        batch_size = 6 # 32?
        if max_steps > 0:
            num_train_epochs = max_steps // (len(train_dataset) // gradient_accumulation_steps) + 1
            num_training_steps = max_steps
        else:
            if self.train_dataset is not None:
                num_training_steps = len(train_dataset) // gradient_accumulation_steps * num_train_epochs
            else:
                num_training_steps = 0

        hparams = {'learning_rate': learning_rate,
                    'adam_epsilon': adam_epsilon,
                    'weight_decay': weight_decay,
                    'warmup_steps': warmup_steps,
                    'num_training_steps': num_training_steps,
                    'batch_size': batch_size,
                    }
        return hparams
    
    def get_processor(self):
        return self.processor



#
# simple tests
#


def test():
    print("Simple base class tests")
    # create a LightningBertForSequenceClassification classifier for MNNLI

    # prepare hyperparameters
    hparams = {'batch_size': 32}

    # now instantiate the model
    model = LightningBertForSequenceClassification(load_from='resources/models/MNLI/pytorch_model.bin', 
                        processor=MnliProcessor(), 
                        hparams=hparams)

    processor = model.get_processor()

    dataset, examples = processor.load_and_cache_examples("data/MNLI", example_set='dev')
    model.set_test_dataset(dataset, examples)

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer.test(model)
    trainer.tqdm_metrics

if __name__ == "__main__":
    import torch
    import pytorch_lightning as pl

    from processors import MnliProcessor

    test()



