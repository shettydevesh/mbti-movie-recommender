# MBTI Movie Recommender 🎬

MBTI Movie Recommender is an AI-powered system that suggests movies based on Myers-Briggs Type Indicator (MBTI) personality types. The system leverages advanced natural language processing techniques to match movie characteristics with personality preferences.

<!-- ![MBTI Movie Recommender Demo](assets/demo.png) -->

## Features

- **Personality-Based Recommendations**: Get movie suggestions tailored to your MBTI personality type
- **AI-Powered Matching**: Uses state-of-the-art transformer models for semantic matching
- **Two-Stage Ranking**: Fast vector search with FAISS followed by precise cross-encoder reranking
- **Streamlit Web Interface**: User-friendly web application for exploring recommendations
- **Command Line Interface**: Quick access to recommendations via CLI

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

<!-- ![Architecture Diagram](assets/architecture.png) -->

## Installation

### Prerequisites

- Python 3.8+
- pip

### Option 1: Install from PyPI

```bash
pip install mbti-movie-recommender
```

### Option 2: Install from Source

```bash
git clone https://github.com/yourusername/mbti-movie-recommender.git
cd mbti-movie-recommender
pip install -e .
```

### Option 3: Docker

```bash
docker pull yourusername/mbti-movie-recommender
docker run -p 8501:8501 yourusername/mbti-movie-recommender
```

## Usage

### Web Interface

To launch the Streamlit web application:

```bash
cd mbti-movie-recommender
streamlit run streamlit_app.py
```

Then open your browser and navigate to http://localhost:8501

### Command Line Interface

Get movie recommendations directly from the command line:

```bash
# Basic usage
mbti-recommend --mbti INTJ

# Advanced options
mbti-recommend --mbti ENFP --top-n 15 --output recommendations.csv
```

### Python API

```python
from models.mbti_recommendations import MBTI_Recommendation
from constants import mbti_mapping

# Initialize the recommendation system
recommender = MBTI_Recommendation(dataset="movies_dataset.csv")

# Get recommendations for a specific MBTI type
meta_data = mbti_mapping["INTJ"]
recommendations = recommender.recommend(meta_data, top_n=10)

# Display results
print(recommendations[["title", "final_score", "genres"]])
```

## Dataset

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

You can use a custom dataset or scrape one from sources like TMDB.

## Configuration

The system is highly configurable. Key settings can be found in `config.py`:

- Model selection
- Weights for different features
- Paths for data and indices
- Default parameters

## Project Structure

```
mbti-movie-recommender/
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
├── streamlit_app.py       # Streamlit web interface
├── cli.py                 # Command line interface
├── config.py              # Configuration settings
├── setup.py               # Installation script
└── README.md              # Documentation
```

## Performance Optimization

- **FAISS Integration**: Fast similarity search using Facebook AI Similarity Search
- **Batch Processing**: Efficient processing of movie embeddings
- **GPU Acceleration**: Automatically uses GPU if available
- **Index Caching**: Save and reuse FAISS indices for faster startup

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MBTI framework provided inspiration for personality-based recommendations
- Built with [Sentence Transformers](https://www.sbert.net/) by UKPLab
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Web interface developed with [Streamlit](https://streamlit.io/)

---

Made with ❤️ by Devesh Shetty