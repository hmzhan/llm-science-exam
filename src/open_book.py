import gc
from pathlib import Path
from config import *
from sentence_transformers import SentenceTransformer
from faiss import write_index, read_index
from src.utils import *


class OpenBook:
    def __init__(self):
        self.train = pd.read_csv(Path(DATA_PATH) / "train.csv")
        self.model = SentenceTransformer(MODEL, device="cuda")
        self.model.max_seq_length = MAX_LENGTH
        self.model = self.model.half()

    def _encode_prompt(self):
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

    def _get_top_3_pages(self):
        prompt_embeddings = self._encode_prompt()
        sentence_index = read_index(SENTENCE_INDEX_FILE)
        search_score, search_index = sentence_index.search(prompt_embeddings, 3)
        del sentence_index
        del prompt_embeddings
        _ = gc.collect()
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

    def create_prompt_context(self, wiki_file_data, processed_wiki_data, wiki_data_embeddings, question_embeddings):
        # Parameter to determine how many relevant sentences to include
        NUM_SENTENCES_INCLUDE = 3
        # List containing Question, Choices, Context
        prompt_contexts = []
        # List containing just Context
        contexts = []

        for r in self.train.itertuples():
            prompt_context = ""
            prompt_id = r.id
            prompt_context += "Question: " + self.train.prompt.iloc[prompt_id] + "\n"
            prompt_context += "Choices:\n"
            prompt_context += "(A) " + self.train.A.iloc[prompt_id] + "\n"
            prompt_context += "(B) " + self.train.B.iloc[prompt_id] + "\n"
            prompt_context += "(C) " + self.train.C.iloc[prompt_id] + "\n"
            prompt_context += "(D) " + self.train.D.iloc[prompt_id] + "\n"
            prompt_context += "(E) " + self.train.E.iloc[prompt_id] + "\n"

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

        return prompt_contexts
