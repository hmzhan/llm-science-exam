import gc
import pandas as pd
from pathlib import Path
from config import *
from sentence_transformers import SentenceTransformer
from faiss import write_index, read_index
from tqdm.auto import tqdm
from src.utils import *


class OpenBook:
    def __init__(self):
        self.train = pd.read_csv(Path(DATA_PATH) / "train.csv")
        self.sentence_index = read_index(SENTENCE_INDEX_FILE)
        self.model = SentenceTransformer(MODEL, device="cuda")
        self.model.max_seq_length = MAX_LENGTH
        self.model = self.model.half

    def encode_prompt(self):
        prompt_embeddings = self.model.encode(
            self.train.prompt.values,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).half()
        prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
        return prompt_embeddings

    def get_top_3_pages(self):
        prompt_embeddings = self.encode_prompt()
        search_score, search_index = self.sentence_index.search(prompt_embeddings, 3)
        _ = gc.collect()
        return search_score, search_index

    def load_wiki_files(self, search_score, search_index):
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

        wiki_text_data = []
        for file in tqdm(wiki_file_data.file.unique(), total=len(wiki_file_data.file.unique())):
            _id = [str(i) for i in wiki_file_data[wiki_file_data['file']==file]['id'].tolist()]
            _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=['id', 'text'])

            _df = _df[_df['id'].isin(_id)]
            wiki_text_data.append(_df)
            _ = gc.collect()
        wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
        _ = gc.collect()
        return wiki_text_data

    def split_doc_into_sentences(self):
        processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)
        # Get embeddings of the wiki text data
        wiki_data_embeddings = self.model.encode(
            processed_wiki_text_data.text,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).half()
        wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()
        _ = gc.collect()
        return wiki_data_embeddings

    def encode_question_answers(self):
        self.train['answer_all'] = self.train.apply(lambda x: " ".join([x['A'], x['B'], x['C'], x['D'], x['E']]),
                                                    axis=1)
        self.train['prompt_answer_stem'] = self.train['prompt'] + " " + self.train['answer_all']
        question_embeddings = self.model.encode(
            self.train.prompt_answer_stem.values,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).half()
        question_embeddings = question_embeddings.detach().cpu().numpy()
        return question_embeddings












