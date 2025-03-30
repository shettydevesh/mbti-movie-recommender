import logging
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import requests
from io import BytesIO
from PIL import Image
import json

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging settings for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler("mbti_recommender.log"),  # Log to file
        ],
    )


def load_dataset(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load and validate the movie dataset.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame or None if loading fails
    """
    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)

        # Basic validation
        required_columns = ["title", "genres", "keywords"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in dataset")
                return None

        logger.info(f"Successfully loaded dataset with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None


def fetch_movie_poster(
    poster_path: str, api_key: Optional[str] = None
) -> Optional[Image.Image]:
    """
    Fetch movie poster from TMDB API.

    Args:
        poster_path: Path to the poster from TMDB
        api_key: TMDB API key (optional)

    Returns:
        PIL Image object or None if fetching fails
    """
    if not poster_path:
        logger.warning("No poster path provided")
        return None

    # If poster_path is already a full URL, use it directly
    if poster_path.startswith("http"):
        url = poster_path
    else:
        # Otherwise, construct TMDB URL
        base_url = "https://image.tmdb.org/t/p/w500"
        url = f"{base_url}{poster_path}"

        # Add API key if provided
        if api_key:
            url += f"?api_key={api_key}"

    try:
        logger.info(f"Fetching poster from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Create image from response
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        logger.error(f"Error fetching poster: {str(e)}")
        return None


def format_movie_info(movie: Dict[str, Any]) -> str:
    """
    Format movie information for display.

    Args:
        movie: Dictionary containing movie data

    Returns:
        Formatted movie information as a string
    """
    # Extract and format fields
    title = movie.get("title", "Unknown Title")
    original_title = movie.get("original_title", "")
    tagline = movie.get("tagline", "")
    overview = movie.get("overview", "No overview available")

    # Format genres and keywords
    genres = movie.get("genres", [])
    if isinstance(genres, list) and genres:
        genres_str = ", ".join(genres)
    else:
        genres_str = "Unknown"

    keywords = movie.get("keywords", [])
    if isinstance(keywords, list) and keywords:
        keywords_str = ", ".join(keywords)
    else:
        keywords_str = "None"

    # Handle revenue
    revenue = movie.get("revenue", 0)
    if revenue and revenue > 0:
        revenue_str = f"${revenue:,}"
    else:
        revenue_str = "Unknown"

    # Format the output
    info = []
    info.append(f"**{title}**")

    if original_title and original_title != title:
        info.append(f"Original Title: {original_title}")

    if tagline:
        info.append(f"*{tagline}*")

    info.append(f"\n{overview}\n")

    info.append(f"**Genres:** {genres_str}")
    info.append(f"**Keywords:** {keywords_str}")
    info.append(f"**Revenue:** {revenue_str}")

    # Add match percentage
    match_score = movie.get("final_score", 0)
    if match_score:
        match_percentage = f"{match_score * 100:.1f}%"
        info.append(f"**Match:** {match_percentage}")

    return "\n\n".join(info)


def save_user_preferences(
    user_id: str,
    mbti_type: str,
    favorite_movies: List[str],
    file_path: str = "user_preferences.json",
) -> bool:
    """
    Save user preferences to a JSON file.

    Args:
        user_id: Unique identifier for the user
        mbti_type: User's MBTI personality type
        favorite_movies: List of user's favorite movies
        file_path: Path to save the preferences

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create data structure
        user_data = {
            "user_id": user_id,
            "mbti_type": mbti_type,
            "favorite_movies": favorite_movies,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Load existing data if file exists
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {"users": []}
        else:
            data = {"users": []}

        # Update or add user data
        user_exists = False
        for i, user in enumerate(data["users"]):
            if user.get("user_id") == user_id:
                data["users"][i] = user_data
                user_exists = True
                break

        if not user_exists:
            data["users"].append(user_data)

        # Save data
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved preferences for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving user preferences: {str(e)}")
        return False


def get_user_preferences(
    user_id: str, file_path: str = "user_preferences.json"
) -> Optional[Dict[str, Any]]:
    """
    Get user preferences from a JSON file.

    Args:
        user_id: Unique identifier for the user
        file_path: Path to the preferences file

    Returns:
        Dictionary with user preferences or None if not found
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Preferences file {file_path} not found")
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        for user in data.get("users", []):
            if user.get("user_id") == user_id:
                logger.info(f"Found preferences for user {user_id}")
                return user

        logger.warning(f"No preferences found for user {user_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return None
