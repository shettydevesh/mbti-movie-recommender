import streamlit as st
import pandas as pd
import os
import uuid
import logging
from pathlib import Path
import sys

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from utils import (
    setup_logging,
    load_dataset,
    fetch_movie_poster,
    format_movie_info,
    save_user_preferences,
)
from constants import mbti_mapping
from models import MBTI_Recommendation
from utils import MBTIData
import config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.STREAMLIT_TITLE,
    page_icon=config.STREAMLIT_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .mbti-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .movie-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "recommender" not in st.session_state:
    st.session_state.recommender = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None


# Helper functions
def load_recommender():
    """Load the recommendation model."""
    try:
        dataset_path = config.DATASET_PATH

        # Check if the dataset exists, otherwise use a default path
        if not dataset_path.exists():
            default_path = Path("movies_dataset.csv")
            if default_path.exists():
                dataset_path = default_path
            else:
                st.error(
                    "Movie dataset not found. Please place 'movies_dataset.csv' in the data directory."
                )
                return None

        # Load the recommender
        recommender = MBTI_Recommendation(
            dataset=str(dataset_path),
            embedding_model=config.DEFAULT_EMBEDDING_MODEL,
            reranker_model=config.DEFAULT_RERANKER_MODEL,
            index_path=str(config.FAISS_INDEX_PATH),
        )
        return recommender
    except Exception as e:
        logger.error(f"Error loading recommender: {str(e)}")
        st.error(f"Error loading recommender: {str(e)}")
        return None


def get_recommendations(mbti_type, top_n=10):
    """Get movie recommendations for the given MBTI type."""
    if st.session_state.recommender is None:
        st.session_state.recommender = load_recommender()
        if st.session_state.recommender is None:
            return None

    try:
        meta_data = mbti_mapping.get(mbti_type, None)
        if meta_data is None:
            st.error(f"No data found for MBTI type {mbti_type}")
            return None

        # Get recommendations
        recommendations = st.session_state.recommender.recommend(
            meta_data=meta_data,
            top_n=top_n,
            rerank_topk=20,
            relevance_weight=config.DEFAULT_RELEVANCE_WEIGHT,
            popularity_weight=config.DEFAULT_POPULARITY_WEIGHT,
        )
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        st.error(f"Error getting recommendations: {str(e)}")
        return None


# Sidebar
with st.sidebar:
    st.title("MBTI Movie Recommender")
    # st.image(
    #     "https://raw.githubusercontent.com/username/mbti-movie-recommender/main/assets/logo.png",
    #     use_container_width=True,
    # )

    # MBTI type selection
    st.subheader("Select Your MBTI Type")

    # Group MBTI types
    mbti_groups = MBTIData.get_mbti_groups()
    mbti_descriptions = MBTIData.get_mbti_descriptions()

    selected_group = st.selectbox(
        "Personality Group", options=list(mbti_groups.keys()), index=0
    )

    selected_type = st.selectbox(
        "MBTI Type",
        options=mbti_groups[selected_group],
        index=0,
        format_func=lambda x: f"{x} - {mbti_descriptions[x].split(' - ')[0]}",
    )

    # Number of recommendations
    num_recommendations = st.slider(
        "Number of recommendations", min_value=5, max_value=20, value=10, step=5
    )

    # Get recommendations button
    if st.button("Get Recommendations"):
        with st.spinner("Analyzing personality preferences..."):
            recommendations = get_recommendations(selected_type, num_recommendations)
            if recommendations is not None:
                st.session_state.recommendations = recommendations
                st.success(f"Found {len(recommendations)} movies for {selected_type}!")

    # About section
    st.subheader("About")
    st.markdown(
        """
    This app recommends movies based on your MBTI personality type using AI-powered 
    semantic matching and personalized recommendations.
    
    Built with:
    - Sentence Transformers
    - FAISS vector search
    - Streamlit
    """
    )

    # Footer
    st.markdown(
        """
    <div class="footer">
        © 2025 Devesh Shetty<br>
        MIT License
    </div>
    """,
        unsafe_allow_html=True,
    )

# Main content
st.markdown(
    '<div class="main-header">MBTI Movie Recommender</div>', unsafe_allow_html=True
)

# MBTI type information
if selected_type:
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display MBTI type card
        st.markdown(
            f'<div class="sub-header">{selected_type} - {mbti_descriptions[selected_type].split(" - ")[0]}</div>',
            unsafe_allow_html=True,
        )

        # Display genres and keywords
        meta_data = mbti_mapping.get(selected_type, {})
        genres = meta_data.get("genres", [])
        keywords = meta_data.get("keywords", [])

        st.markdown('<div class="mbti-card">', unsafe_allow_html=True)
        st.markdown(
            f"**Description**: {mbti_descriptions[selected_type].split(' - ')[1]}"
        )
        st.markdown("**Preferred Genres**")
        for genre in genres:
            st.markdown(f"- {genre.title()}")

        st.markdown("**Personality Keywords**")
        for keyword in keywords:
            st.markdown(f"- {keyword.title()}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # MBTI description and explanation
        st.markdown("### How We Match Movies to Your Personality")
        st.markdown(
            """
        The recommendation engine analyzes your MBTI personality type's preferences and matches them with movies using:
        
        1. **AI-Powered Semantic Matching**: We use advanced natural language models to understand the deeper meaning behind movie genres and keywords.
        
        2. **Personality Alignment**: Your MBTI profile provides insights into the stories, themes, and character arcs that might resonate with you.
        
        3. **Two-Stage Ranking**: First, we find potential matches using fast vector search, then we re-rank them using a sophisticated cross-encoder model.
        
        This creates a personalized recommendation experience that goes beyond simple genre matching.
        """
        )

# Movie recommendations
if st.session_state.recommendations is not None:
    st.markdown(
        '<div class="sub-header">Your Personalized Movie Recommendations</div>',
        unsafe_allow_html=True,
    )

    # Display recommendations in a grid
    num_cols = 2
    recommendations = st.session_state.recommendations.to_dict(orient="records")

    for i in range(0, len(recommendations), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i + j
            if idx < len(recommendations):
                movie = recommendations[idx]
                with cols[j]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                    # Movie poster
                    if "poster_path" in movie and movie["poster_path"]:
                        try:
                            poster = fetch_movie_poster(movie["poster_path"])
                            if poster:
                                st.image(poster, width=200)
                        except Exception as e:
                            st.warning("Poster not available")

                    # Movie info
                    st.markdown(format_movie_info(movie))

                    # Add to favorites button
                    if st.button(f"Add to Favorites", key=f"fav_{idx}"):
                        movie_title = movie.get("title", "Unknown")
                        save_user_preferences(
                            st.session_state.user_id, selected_type, [movie_title]
                        )
                        st.success(f"Added {movie_title} to your favorites!")

                    st.markdown("</div>", unsafe_allow_html=True)
else:
    # Welcome message when no recommendations yet
    st.markdown(
        """
    ### Welcome to the MBTI Movie Recommender!
    
    This application helps you discover movies that match your personality type based on the
    Myers-Briggs Type Indicator (MBTI). 
    
    To get started:
    1. Select your MBTI personality type from the sidebar
    2. Click "Get Recommendations"
    3. Explore your personalized movie suggestions
    
    The system uses advanced AI to understand the connection between personality traits and
    movie preferences, creating a unique recommendation experience just for you.
    """
    )

    # Display example image
    # st.image(
    #     "https://raw.githubusercontent.com/username/mbti-movie-recommender/main/assets/example.png",
    #     use_container_width=True,
    #     caption="Example recommendations for INTJ personality type",
    # )
