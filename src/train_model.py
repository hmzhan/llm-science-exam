
import os
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer
)
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)


os.environ['CUDA_VISIBLE_DEVICES'] = '0.1'
VERSION = 2
NUM_TRAIN_SAMPLES = 1_024
USE_PEFT = False
FREEZE_LAYERS = 18
FREEZE_EMBEDDINGS = True
MAX_INPUT = 256
MODEL = 'microsoft/deberta-v3-large'


class DataModule:
    def __init__(self):
        self.train_data = self._load_train_data()
        self.val_data = self._load_val_data()
        self.tokenized_train_data = self.tokenize_data(self.train_data, train=True)
        self.tokenized_val_data = self.tokenize_data(self.val_data, train=False)

    @staticmethod
    def _load_train_data():
        df_train = pd.read_csv('/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv')
        df_train = df_train.drop(columns="source")
        df_train = df_train.fillna('').sample(NUM_TRAIN_SAMPLES)
        return df_train

    @staticmethod
    def _load_val_data():
        return pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv')

    @staticmethod
    def _preprocess(example):
        option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
        index_to_option = {v: k for k, v in option_to_index.items()}

        first_sentence = ['[CLS] ' + example['context']] * 5
        second_sentence = [' #### ' + example['prompt'] + ' [SEP] ' + example[option] + ' [SEP]' for option in 'ABCDE']

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokenized_example = tokenizer(
            first_sentence,
            second_sentence,
            truncation='only_first',
            max_length=MAX_INPUT,
            add_special_tokens=False
        )
        tokenized_example['label'] = option_to_index[example['answer']]
        return tokenized_example

    def tokenize_data(self, data, train=True):
        dataset = Dataset.from_pandas(data)
        if train:
            dataset = dataset.remove_columns(['__index_level_0__'])
        tokenized_dataset = dataset.map(self._preprocess,
                                        remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
        return tokenized_dataset


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


class ModelModule:
    def __init__(self):
        self.llm = AutoModelForMultipleChoice.from_pretrained(MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        if USE_PEFT:
            self._adjust_model_peft()
        if FREEZE_EMBEDDINGS:
            self._adjust_model_embedding()
        if FREEZE_LAYERS > 0:
            self._adjust_model_layer()

    def _adjust_model_peft(self):
        peft_config = LoraConfig(
            r=8,
            lora_alpha=4,
            task_type=TaskType.SEQ_CLS,
            lora_dropout=0.1,
            bias='none',
            inference_model=False,
            target_modules=['query_proj', 'value_proj'],
            modules_to_save=['classifier', 'pooler']
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()

    def _adjust_model_embedding(self):
        for param in self.llm.deberta.embeddings.parameters():
            param.requires_grad = False

    def _adjust_model_layer(self):
        for layer in self.llm.deberta.encoder.layer[:FREEZE_LAYERS]:
            for param in layer.parameters():
                param.requires_grad = False


class TrainModule:
    def __init__(self):
        self.training_args = TrainingArguments(
            warmup_ratio=0.1,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            num_train_epochs=2,
            report_to='none',
            output_dir=f'./checkpoints_{VERSION}',
            overwrite_output_dir=True,
            fp16=True,
            gradient_accumulation_steps=8,
            logging_steps=25,
            evaluation_strategy='steps',
            eval_steps=25,
            save_strategy="steps",
            save_steps=25,
            load_best_model_at_end=False,
            metric_for_best_model='map@3',
            lr_scheduler_type='cosine',
            weight_decay=0.01,
            save_total_limit=2,
        )

    def fit(self, model, data):
        trainer = Trainer(
            model=model.llm,
            args=self.training_args,
            tokenizer=model.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=model.tokenizer),
            train_dataset=data.tokenized_train_data,
            eval_dataset=data.tokenized_val_data,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        trainer.save_model(f'model_v{VERSION}')

    @staticmethod
    def _map_at_3(predictions, labels):
        map_sum = 0
        pred = np.argsort(-1*np.array(predictions), axis=1)[:, :3]
        for x, y in zip(pred, labels):
            z = [1/i if y == j else 0 for i, j in zip([1, 2, 3], x)]
            map_sum += np.sum(z)
        return map_sum / len(predictions)

    def compute_metrics(self, p):
        predictions = p.predictions.tolist()
        labels = p.label_ids.tolist()
        return {'map@3': self._map_at_3(predictions, labels)}


class ValidationModule:
    def __init__(self):
        self.model = self._load_trained_model()
        self.test_data = self._load_test_data()
        self.tokenized_test_data = self._tokenize_test_data()
        self.trainer = Trainer(model=self.model)

    @staticmethod
    def _load_trained_model():
        if USE_PEFT:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=4,
                task_type=TaskType.SEQ_CLS,
                lora_dropout=0.1,
                bias='none',
                inference_model=False,
                target_modules=['query_proj', 'value_proj'],
                modules_to_save=['classifier', 'pooler']
            )
            model = AutoModelForMultipleChoice.from_pretrained(MODEL)
            model = get_peft_model(model, peft_config)
            checkpoint = torch.load(f'model_v{VERSION}/pytorch_model.bin')
            return model.load_state_dict(checkpoint)
        else:
            return AutoModelForMultipleChoice.from_pretrained(f'model_v{VERSION}')

    @staticmethod
    def _load_test_data():
        return pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv')

    @staticmethod
    def _preprocess(example):
        option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
        index_to_option = {v: k for k, v in option_to_index.items()}

        first_sentence = ['[CLS] ' + example['context']] * 5
        second_sentence = [' #### ' + example['prompt'] + ' [SEP] ' + example[option] + ' [SEP]' for option in 'ABCDE']

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokenized_example = tokenizer(
            first_sentence,
            second_sentence,
            truncation='only_first',
            max_length=MAX_INPUT,
            add_special_tokens=False
        )
        tokenized_example['label'] = option_to_index[example['answer']]
        return tokenized_example

    def _tokenize_test_data(self):
        tokenized_test_dataset = Dataset.from_pandas(self.test_data).map(
            self._preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E'])
        return tokenized_test_dataset

    def make_prediction(self):
        test_predictions = self.trainer.predict(self.tokenized_test_data).predictions
        predictions_as_ids = np.argsort(-test_predictions, 1)
        predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
        self.test_data['prediction'] = [
            ' '.join(row) for row in predictions_as_answer_letters[:, :3]
        ]

    def calculate_map_at_3(self):
        return self.MAP_at_3(self.test_data['prediction'].values, self.test_data['answer'].values)

    @staticmethod
    def precision_at_k(r, k):
        """Precision at k"""
        assert k <= len(r)
        assert k != 0
        return sum(int(x) for x in r[:k]) / k

    def MAP_at_3(self, predictions, true_items):
        """Score is mean average precision at 3"""
        U = len(predictions)
        map_at_3 = 0.0
        for u in range(U):
            user_preds = predictions[u].split()
            user_true = true_items[u]
            user_results = [1 if item == user_true else 0 for item in user_preds]
            for k in range(min(len(user_preds), 3)):
                map_at_3 += self.precision_at_k(user_results, k+1) * user_results[k]
        return map_at_3 / U
