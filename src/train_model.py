
import os
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallBack,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer
)
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy
)


os.environ['CUDA_VISIBLE_DEVICES'] = '0.1'


class DataModule:
    def __init__(self):
        self.train_data = self._load_train_data()
        self.val_data = self._load_val_data()

    @staticmethod
    def _load_train_data(self):
        df_train = pd.read_csv('/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv')
        df_train = df_train.drop(columns="source")
        df_train = df_train.fillna('').sample(NUM_TRAIN_SAMPLES)
        return df_train

    @staticmethod
    def _load_val_data(self):
        return pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv')


class ModelModule:
    def __init__(self, model):
        self.llm = AutoModelForMultipleChoice.from_pretrained(model)


class TrainModule:
    def __init__(self):
        self.training_args = TrainingArguments(
            warmup_ratio=0.1,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            num_train_epochs=2,
            report_to='none',
            output_dir=f'./checkpoints_{VER}',
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
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset_valid,
            compute_metrics=self.compute_metrics
        )

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

