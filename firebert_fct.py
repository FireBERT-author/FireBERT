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
# FCT as a subclass
# overrides extend_batch() used in training_step()
#

import torch
from torch import nn
from switch import SWITCH
from firebert_base import FireBERT_base

from transformers.data.processors.utils import InputExample



class FireBERT_FCT(FireBERT_base):


    def __init__(self, load_from=None, switch_model=None, perturb_text_b=None, processor=None, hparams=None):
        super(FireBERT_FCT, self).__init__(load_from=load_from, processor=processor, hparams=hparams)
        
        if switch_model == None:
            switch_model = self
            
        self.switch = SWITCH(hparams=hparams, perturb_text_b=perturb_text_b, model=switch_model,
                             tokenizer=switch_model.tokenizer, device=switch_model.device)

    #
    # here are some useful defaults
    #
    def get_default_hparams(self):
        d = FireBERT_base.get_default_hparams(self)
        d.update({
                # these are for SWITCH
                'use_USE': True,
                'USE_method': "filter",
                'USE_multiplier': 3,
                'stop_words': True,
                'perturb_words': 2,
                'candidates_per_word': 10,
                'total_alternatives': 5,
                'match_pos': True,
                'use_full_example': False
            })
        return d


    # this fills in the hook prepared in the base class
    def extend_batch_examples_train(self, batch):
        input_ids, attention_mask, token_type_ids, label, example_idx = batch

        for idx in example_idx:
            example = self.train_examples[idx]
            #print(example)
            if self.hparams.use_full_example:
                new_texts = self.switch.generate_candidates_from_example(example)
            else:
                text_a_example = InputExample(guid = example.guid,
                                              text_a = self.processor.get_text_to_perturb(example),
                                              text_b = None,
                                              label = example.label)
                new_texts = self.switch.generate_candidates_from_example(text_a_example)
                #new_texts = self.switch.generate_candidates()
            if new_texts != []:
                alternates, idxs = self.processor.create_adversarial_examples(example, new_texts)
                alt_input_ids, alt_attention_mask, alt_token_type_ids, alt_labels = \
                        self.processor.create_feature_tensors(alternates)
                alt_idxs = torch.tensor([idx for i in range(len(idxs))], dtype=torch.long)
                    
                input_ids = torch.cat((input_ids, alt_input_ids.to(self.device)), dim=0)
                attention_mask = torch.cat((attention_mask, alt_attention_mask.to(self.device)), dim=0)
                token_type_ids = torch.cat((token_type_ids, alt_token_type_ids.to(self.device)), dim=0)
                label = torch.cat((label, alt_labels.to(self.device)), dim=0)
                example_idx = torch.cat((example_idx, alt_idxs.to(self.device)), dim=0)

        return input_ids, attention_mask, token_type_ids, label, example_idx



#
# simple tests
#


def test_FireBERT_FCT():
    print("Simple FireBERT_FCT tests")
    # create a FireBERT_FCT classifier for MNNLI

    # load the right processor class
    processor = MnliProcessor({'sample_percent':3}) # negative number would means abs number of samples, not percent

    # prepare hyperparameters
    hparams = {'batch_size': 32, 'sample_percent':3}

    # now instantiate the model
    model = FireBERT_FCT(load_from='resources/models/MNLI/pytorch_model.bin', 
                        processor=processor, 
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

    test_FireBERT_FCT()

