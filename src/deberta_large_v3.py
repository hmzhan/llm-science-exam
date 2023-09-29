import os
import gc
import re

import torch
import ctypes

import pandas as pd
import numpy as np
import blingfire as bf

from __future__ import annotations
from tqdm.auto import tqdm
from faiss import write_index, read_index
from sentence_transformers import SentenceTransformer
from collections.abc import Iterable
from config import *


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:

    processed_documents = []
    for doc_id, doc in tqdm(zip(document_ids, document_ids), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (doc, 0, len(doc))
        row['doc_id'] = doc_id
        row['text'] = text
        row['offset'] = (start, end)
        processed_documents.append(row)
    df = pd.DataFrame(processed_documents)
    if df.shape[0] > 0:
        return df.sort_values(['doc_id', 'offset']).reset_index(drop=True)
    else:
        return df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row = {
                        'document_id': document_id,
                        'text': sentence,
                        'offset': abs_offsets
                    }
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)


def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values,
                        df.document_id.values,
                        df.offset.values,
                        filter_len,
                        disable_progress_bar)
    return df


model = SentenceTransformer(SIM_MODEL, device='cuda')
model.max_seq_length = MAX_LENGTH
model = model.half()


sentence_index = read_index("/kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index")


prompt_embeddings = model.encode(trn.prompt.values, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
_ = gc.collect()


search_score, search_index = sentence_index.search(prompt_embeddings, 3)


## Save memory - delete sentence_index since it is no longer necessary
libc = ctypes.CDLL("libc.so.6")
del sentence_index
del prompt_embeddings
_ = gc.collect()
libc.malloc_trim(0)




