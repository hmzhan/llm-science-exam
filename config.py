import os

MODEL = '/kaggle/input/sentencetransformers-allminilml6v2/sentence-transformers_all-MiniLM-L6-v2'
DEVICE = 0
MAX_LENGTH = 384
BATCH_SIZE = 16

WIKI_PATH = "/kaggle/input/wikipedia-20230701"
wiki_files = os.listdir(WIKI_PATH)

SENTENCE_INDEX_FILE = "/kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index"
DATA_PATH = "/kaggle/input/kaggle-llm-science-exam/"
