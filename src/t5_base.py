import pandas as pd
from string import Template
from pathlib import Path

import warnings
warnings.simplefilter("ignore")

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


DATA_PATH = Path("/kaggle/input/kaggle-llm-science-exam")
MODEL_PATH = "/kaggle/input/flan-t5/pytorch/base/2"


class T5Simple:
    def __init__(self, model_path, data_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.test = pd.read_csv(data_path / "test.csv", index_col="id")
        self.submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='id')

    @staticmethod
    def format_input(df, idx):
        preamble = \
            'Answer the following question by outputting the letters A, B, C, D, and E ' \
            'in order of the most likely to be correct to the to least likely to be correct.'

        template = Template('$preamble\n\n$prompt\n\nA) $a\nB) $b\nC) $c\nD) $d\nE) $e')
        prompt = df.loc[idx, 'prompt']
        a = df.loc[idx, 'A']
        b = df.loc[idx, 'B']
        c = df.loc[idx, 'C']
        d = df.loc[idx, 'D']
        e = df.loc[idx, 'E']

        input_text = template.substitute(
            preamble=preamble, prompt=prompt, a=a, b=b, c=c, d=d, e=e)
        return input_text

    @staticmethod
    def post_process(predictions):
        valid = {'A', 'B', 'C', 'D', 'E'}
        # If there are no valid choices, return something and hope for partial credit
        if set(predictions).isdisjoint(valid):
            final_pred = 'A B C D E'
        else:
            final_pred = []
            for prediction in predictions:
                if prediction in valid:
                    final_pred += prediction
            # add remaining letters
            to_add = valid - set(final_pred)
            final_pred.extend(list(to_add))
            # put in space-delimited format
            final_pred = ' '.join(final_pred)
        return final_pred

    def make_submission(self):
        for idx in self.test.index:
            inputs = self.tokenizer(self.format_input(self.test, idx), return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.submission.loc[idx, 'prediction'] = self.post_process(answer)
        self.submission.to_csv("submission.csv")
