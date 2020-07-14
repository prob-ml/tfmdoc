import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from transformers import DataCollatorForLanguageModeling, BertForMaskedLM

from datas import LineByLineTextDataset
from tokens import *


if __name__ == '__main__':
    seq_len = 110

    tokenizer = WordLevelTokenizer('./Synthetic/vocab.json')
    BertTokenizer = WordLevelBertTokenizer(tokenizer=tokenizer)

    dataset = LineByLineTextDataset(tokenizer=BertTokenizer,
                                    file_path="./synthetic_pretrain_X.txt",
                                    block_size=seq_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=BertTokenizer, mlm=True, mlm_probability=0.15, )
    data_collator.mask_tokens()

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(output_dir="./Synthetic/finetune_v6", overwrite_output_dir=True,
        num_train_epochs=2, per_device_train_batch_size=32, save_steps=10_000, )

    # Specify visible CUDA for the script, try to avoid encounder out of memory issue.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_NEW_MODEL = True

    if TRAIN_NEW_MODEL:

        from transformers import BertConfig

        config = BertConfig(vocab_size=len(BertTokenizer), max_position_embeddings=seq_len, num_attention_heads=4,
            num_hidden_layers=2, hidden_size=64, type_vocab_size=1, )

    else:

        # load a pre-trained model.
        from transformers import AutoConfig

        # config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

    model = BertForMaskedLM(config=config)

    print(f'The Bert model contains {model.num_parameters()} parameters.')

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset,
        prediction_loss_only=True, )

    trainer.train()