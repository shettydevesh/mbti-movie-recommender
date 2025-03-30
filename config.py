import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Model configurations
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-roberta-large-v1"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-electra-base"

# Dataset path
DATASET_PATH = DATA_DIR / "movies_dataset.csv"

# FAISS index path
FAISS_INDEX_PATH = INDEX_DIR / "roberta_index.idx"

# Default weights
DEFAULT_GENRE_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_RELEVANCE_WEIGHT = 0.7
DEFAULT_POPULARITY_WEIGHT = 0.3

# Streamlit configurations
STREAMLIT_TITLE = "MBTI Movie Recommender"
STREAMLIT_ICON = "🎬"
