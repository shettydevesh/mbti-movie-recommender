# MBTI Movie Recommender 🎬

MBTI Movie Recommender is an AI-powered system that suggests movies based on Myers-Briggs Type Indicator (MBTI) personality types. The system leverages advanced natural language processing techniques to match movie characteristics with personality preferences.

<!---[MBTI Movie Recommender Architecture](assets/architecture.png) --->

## Features

- **Personality-Based Recommendations**: Get movie suggestions tailored to your MBTI personality type
- **AI-Powered Matching**: Uses state-of-the-art transformer models for semantic matching
- **Two-Stage Ranking**: Fast vector search with FAISS followed by precise cross-encoder reranking
- **Streamlit Web Interface**: User-friendly web application for exploring recommendations

## Architecture

The recommendation system uses a two-stage approach:

1. **First Stage**: Fast approximate nearest neighbor search using FAISS
   - Creates embeddings for movies using genres and keywords
   - Creates a query embedding from MBTI type preferences
   - Efficiently retrieves top candidate matches

2. **Second Stage**: Precise reranking with a cross-encoder model
   - Compares query directly to each candidate movie
   - Provides more accurate relevance scores
   - Combines with popularity metrics for final ranking

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository
```bash
git clone https://github.com/devesh-shetty/mbti-movie-recommender.git
cd mbti-movie-recommender
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Prepare your movie dataset
   - Place your movie dataset as `movies_dataset.csv` in the `data` directory
   - Ensure it has the required columns: title, genres, keywords

## Usage

### Web Interface

To launch the Streamlit web application:

```bash
streamlit run app.py
```

Then open your browser and navigate to http://localhost:8501

### Python API

You can also use the recommendation system directly in your Python code:

```python
from models.mbti_recommendations import MBTI_Recommendation
from constants import mbti_mapping

# Initialize the recommendation system
recommender = MBTI_Recommendation(dataset="data/movies_dataset.csv")

# Get recommendations for a specific MBTI type
meta_data = mbti_mapping["INTJ"]
recommendations = recommender.recommend(meta_data, top_n=10)

# Display results
print(recommendations[["title", "final_score", "genres"]])
```

## Dataset Requirements

The recommendation system requires a movie dataset with at least the following columns:

- `title`: Movie title
- `genres`: Comma-separated list of genres
- `keywords`: Comma-separated list of keywords

Optional but recommended columns:
- `original_title`: Original movie title (if different)
- `overview`: Movie description
- `poster_path`: Path to movie poster
- `popularity`: Popularity metric
- `revenue`: Box office revenue
- `tagline`: Movie tagline

## Technical Details

### AI Models

- **Embedding Model**: `sentence-transformers/all-roberta-large-v1` for semantic embeddings
- **Reranker Model**: `cross-encoder/ms-marco-electra-base` for precise relevance scoring

### Key Components

- **SentenceTransformer**: Generates embeddings for movies and MBTI preferences
- **CrossEncoder**: Performs precise scoring between query and candidate movies
- **FAISS**: Efficient similarity search for retrieving candidates
- **Streamlit**: Interactive web interface for user interaction

### Recommendation Algorithm

1. **Query Processing**: Convert MBTI preferences into a semantic query
2. **Candidate Retrieval**: Use FAISS to find potentially relevant movies
3. **Reranking**: Score candidates with the cross-encoder model
4. **Final Scoring**: Combine semantic relevance with popularity metrics
5. **Result Presentation**: Return and display top-ranked movies

## Configuration

The system is configurable through `config.py`:

```python
# Model configurations
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-roberta-large-v1"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-electra-base"

# Default weights
DEFAULT_GENRE_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_RELEVANCE_WEIGHT = 0.7
DEFAULT_POPULARITY_WEIGHT = 0.3
```

## Performance Optimization

- **FAISS Integration**: Fast similarity search using Facebook AI Similarity Search
- **Batch Processing**: Efficient processing of movie embeddings
- **GPU Acceleration**: Automatically uses GPU if available
- **Index Caching**: Save and reuse FAISS indices for faster startup

## Project Structure

```
mbti-movie-recommender/
├── app.py                 # Streamlit web interface
├── config.py              # Configuration settings
├── constants/
│   ├── __init__.py
│   └── variables.py       # MBTI personality type mappings
├── models/
│   ├── __init__.py
│   └── mbti_recommendations.py   # Core recommendation engine
├── utils/
│   ├── __init__.py
│   ├── mbti_data.py       # MBTI data management
│   └── utilities.py       # Utility functions
├── data/                  # Directory for datasets
├── index/                 # Directory for FAISS indices
└── README.md              # Documentation
```

## Dependencies

Main dependencies include:
- streamlit
- pandas
- numpy
- faiss-cpu (or faiss-gpu)
- sentence-transformers
- torch
- tqdm
- requests
- Pillow

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MBTI framework provided inspiration for personality-based recommendations
- Built with [Sentence Transformers](https://www.sbert.net/) by UKPLab
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Web interface developed with [Streamlit](https://streamlit.io/)

---

Made with ❤️ by Devesh Shetty
