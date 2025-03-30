import streamlit as st
import pandas as pd
import os
import uuid
import logging
from pathlib import Path
import sys
import random

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

# CSS for styling - updated for new design
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .personality-keywords {
        font-size: 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .personality-description {
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: center;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
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
        padding: 0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .movie-poster-container {
        position: relative;
    }
    .movie-year {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .movie-content {
        padding: 1rem;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    .movie-title {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .genre-tag {
        display: inline-block;
        background-color: #f0f2f6;
        color: #333;
        padding: 3px 10px;
        border-radius: 15px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.8rem;
    }
    .movie-overview {
        margin-top: 0.8rem;
        margin-bottom: 1rem;
        flex-grow: 1;
    }
    .match-reason {
        border-top: 1px solid #eee;
        padding-top: 0.8rem;
        margin-top: 0.5rem;
    }
    .match-label {
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    .match-explanation {
        font-style: italic;
        color: #555;
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


def get_match_explanation(mbti_type, movie):
    """Generate a personalized explanation for why a movie matches the MBTI type."""
    # Dictionary of explanation templates for each MBTI type
    explanations = {
        "INTJ": [
            "Appeals to your strategic mindset with complex plot development and innovative concepts.",
            "Features a methodical protagonist solving intricate problems, aligning with your analytical approach.",
            "Explores visionary ideas and systematic thinking that resonate with your long-term perspective.",
        ],
        "INTP": [
            "Follows a brilliant mind exploring complex problems and theories, appealing to INTP's love for abstract thinking.",
            "Showcases logical problem-solving and the life of a misunderstood genius, resonating with INTP's intellectual approach.",
            "Explores the nature of consciousness and theoretical concepts, providing the theoretical depth INTPs crave.",
            "Presents fascinating logical puzzles and abstract concepts that engage your analytical curiosity.",
        ],
        "ENTJ": [
            "Features strong leadership themes and strategic decision-making that align with your decisive nature.",
            "Showcases efficiency and structured approaches to overcoming obstacles.",
            "Appeals to your appreciation for ambition and objective problem-solving in high-stakes situations.",
        ],
        "ENTP": [
            "Full of creative possibilities and inventive problem-solving that appeal to your innovative thinking.",
            "Features witty dialogue and unconventional approaches that match your love of debate and mental challenges.",
            "Balances humor with complex ideas, engaging your curiosity and love for theoretical exploration.",
        ],
        # Add more for other types...
    }

    # Get explanations for the given MBTI type, or use a generic one if not found
    type_explanations = explanations.get(
        mbti_type,
        [
            "Aligns with your personality preferences and cognitive style.",
            "Features themes and characters that resonate with your MBTI type.",
            "Presents a narrative and perspective that matches your personality preferences.",
        ],
    )

    # Select a random explanation from the list
    return random.choice(type_explanations)


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
if st.session_state.recommendations is not None:
    # Get type information
    type_name = mbti_descriptions[selected_type].split(" - ")[0]
    type_description = mbti_descriptions[selected_type].split(" - ")[1]

    # Get keywords (personality traits)
    keywords = mbti_mapping.get(selected_type, {}).get("keywords", [])
    formatted_keywords = " • ".join([keyword.capitalize() for keyword in keywords[:5]])

    # Display header with MBTI type information
    st.markdown(
        f'<div class="main-header">Movies for {selected_type}: {type_name}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="personality-keywords">{formatted_keywords}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="personality-description">{type_description} Here are some movies that might resonate with your personality type.</div>',
        unsafe_allow_html=True,
    )

    # Display recommendations in a grid
    num_cols = 3  # Show 3 movies per row
    recommendations = st.session_state.recommendations.to_dict(orient="records")

    for i in range(0, len(recommendations), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i + j
            if idx < len(recommendations):
                movie = recommendations[idx]
                with cols[j]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                    # Movie poster with year overlay
                    st.markdown(
                        '<div class="movie-poster-container">', unsafe_allow_html=True
                    )
                    if "poster_path" in movie and movie["poster_path"]:
                        try:
                            poster = fetch_movie_poster(movie["poster_path"])
                            if poster:
                                st.image(poster, use_container_width=True)
                        except Exception as e:
                            st.image(
                                "https://via.placeholder.com/300x450?text=No+Poster",
                                use_container_width=True,
                            )
                    else:
                        st.image(
                            "https://via.placeholder.com/300x450?text=No+Poster",
                            use_container_width=True,
                        )

                    # Year (can be added if available in your data)
                    release_year = (
                        movie.get("release_date", "").split("-")[0]
                        if movie.get("release_date", "")
                        else "N/A"
                    )
                    st.markdown(
                        f'<div class="movie-year">{release_year}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Movie content
                    st.markdown('<div class="movie-content">', unsafe_allow_html=True)

                    # Title
                    title = movie.get("title", "Unknown Title")
                    st.markdown(
                        f'<div class="movie-title">{title}</div>',
                        unsafe_allow_html=True,
                    )

                    # Genres as tags
                    genres = movie.get("genres", [])
                    if isinstance(genres, list) and genres:
                        genre_html = ""
                        for genre in genres[:3]:  # Limit to 3 genres
                            genre_html += (
                                f'<span class="genre-tag">{genre.title()}</span>'
                            )
                        st.markdown(genre_html, unsafe_allow_html=True)

                    # Overview
                    overview = movie.get("overview", "No overview available")
                    if isinstance(overview, str):
                        truncated_overview = (
                            overview[:150] + "..." if len(overview) > 150 else overview
                        )
                    else:
                        truncated_overview = "No overview available"
                    st.markdown(
                        f'<div class="movie-overview">{truncated_overview}</div>',
                        unsafe_allow_html=True,
                    )

                    # Match explanation
                    match_explanation = get_match_explanation(selected_type, movie)
                    st.markdown(
                        f"""
                        <div class="match-reason">
                            <div class="match-label">Why it matches your type:</div>
                            <div class="match-explanation">{match_explanation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)
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
