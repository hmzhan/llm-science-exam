import gc
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Optional, Union

import faiss
from faiss import write_index, read_index
from dataclasses import dataclass
from ..config import *
from .utils import *

from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy
)

import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")


class OpenBook:
    def __init__(self, prompt_data):
        self.prompt_data = pd.read_csv(Path(DATA_PATH) / prompt_data)
        self.model = SentenceTransformer(MODEL, device="cuda")
        self.model.max_seq_length = MAX_LENGTH
        self.model = self.model.half()

    def _encode_prompt(self):
        prompt_embeddings = self.model.encode(
            self.prompt_data.prompt.values,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
        _ = gc.collect()
        return prompt_embeddings

    def _get_top_3_pages(self):
        prompt_embeddings = self._encode_prompt()
        sentence_index = read_index(SENTENCE_INDEX_FILE)
        search_score, search_index = sentence_index.search(prompt_embeddings, 3)
        del sentence_index
        del prompt_embeddings
        _ = gc.collect()
        libc.malloc_trim(0)
        return search_score, search_index

    def load_wiki_files(self):
        search_score, search_index = self._get_top_3_pages()
        df = pd.read_parquet("/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet", columns=['id', 'file'])
        wiki_file_data = []
        for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
            scr_idx = idx
            _df = df.loc[scr_idx].copy()
            _df['prompt_id'] = i
            wiki_file_data.append(_df)
        wiki_file_data = pd.concat(wiki_file_data).reset_index(drop=True)
        wiki_file_data = wiki_file_data[['id', 'prompt_id', 'file']]\
            .drop_duplicates().sort_values(['file', 'id']).reset_index(drop=True)
        del df
        _ = gc.collect()
        return wiki_file_data

    @staticmethod
    def load_wiki_text(wiki_file_data):
        wiki_text_data = []
        for file in tqdm(wiki_file_data.file.unique(), total=len(wiki_file_data.file.unique())):
            _id = [str(i) for i in wiki_file_data[wiki_file_data['file'] == file]['id'].tolist()]
            _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=['id', 'text'])

            _df = _df[_df['id'].isin(_id)]
            wiki_text_data.append(_df)
            _ = gc.collect()
        wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
        _ = gc.collect()
        return wiki_text_data

    def process_wiki_text(self, wiki_file_data):
        wiki_text_data = self.load_wiki_text(wiki_file_data)
        return process_documents(wiki_text_data.text.values, wiki_text_data.id.values)

    def create_wiki_embedding(self, processed_wiki_data):
        # Get embeddings of the wiki text data
        wiki_data_embeddings = self.model.encode(
            processed_wiki_data.text,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()
        _ = gc.collect()
        return wiki_data_embeddings

    def encode_question_answers(self):
        self.prompt_data['answer_all'] = self.prompt_data.apply(
            lambda x: " ".join([x['A'], x['B'], x['C'], x['D'], x['E']]), axis=1)
        self.prompt_data['prompt_answer_stem'] = self.prompt_data['prompt'] + " " + self.prompt_data['answer_all']
        question_embeddings = self.model.encode(
            self.prompt_data.prompt_answer_stem.values,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        question_embeddings = question_embeddings.detach().cpu().numpy()
        return question_embeddings

    def create_prompt_context(self, wiki_file_data, processed_wiki_data, wiki_data_embeddings, question_embeddings):
        # Parameter to determine how many relevant sentences to include
        NUM_SENTENCES_INCLUDE = 3
        # List containing Question, Choices, Context
        prompt_contexts = []
        # List containing just Context
        contexts = []

        for r in self.prompt_data.itertuples():
            prompt_context = ""
            prompt_id = r.id
            prompt_context += "Question: " + self.prompt_data.prompt.iloc[prompt_id] + "\n"
            prompt_context += "Choices:\n"
            prompt_context += "(A) " + self.prompt_data.A.iloc[prompt_id] + "\n"
            prompt_context += "(B) " + self.prompt_data.B.iloc[prompt_id] + "\n"
            prompt_context += "(C) " + self.prompt_data.C.iloc[prompt_id] + "\n"
            prompt_context += "(D) " + self.prompt_data.D.iloc[prompt_id] + "\n"
            prompt_context += "(E) " + self.prompt_data.E.iloc[prompt_id] + "\n"

            prompt_indices = processed_wiki_data[processed_wiki_data['document_id'].isin(
                wiki_file_data[wiki_file_data['prompt_id'] == prompt_id]['id'].values)].index.values

            if prompt_indices.shape[0] > 0:
                prompt_context += "Context:\n"
                # Per Prompt Index
                prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
                prompt_index.add(wiki_data_embeddings[prompt_indices])
                context = ""
                # Get the top matches
                ss, ii = prompt_index.search(question_embeddings, NUM_SENTENCES_INCLUDE)
                for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
                    # Threshold on the score
                    if _s < 2:
                        context += processed_wiki_data.loc[prompt_indices]['text'].iloc[_i] + "\n"
                prompt_context += context
            contexts.append(context)
            prompt_contexts.append(prompt_context)

        return contexts


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


