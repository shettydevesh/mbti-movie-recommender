import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from typing import Dict, List, Tuple, Union, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MBTI_Recommendation:
    """
    Movie recommendation system based on MBTI personality types.

    This class uses semantic embeddings from transformer models to find movies
    that match the preferences of different MBTI personality types.
    """

    def __init__(
        self,
        dataset: str,
        embedding_model: str = "sentence-transformers/all-roberta-large-v1",
        reranker_model: str = "cross-encoder/ms-marco-electra-base",
        index_path: str = "index/roberta_index.idx",
        query_weights: Dict[str, float] = None,
    ):
        """
        Initialize the MBTI movie recommendation system.

        Args:
            dataset: Path to the movie dataset CSV file
            embedding_model: Sentence transformer model for embeddings
            reranker_model: Cross encoder model for reranking
            index_path: Path to save/load the FAISS index
            query_weights: Weights for different features in the query
        """
        self.df = pd.read_csv(dataset)
        self.device = self._get_device()
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.index_path = Path(index_path)

        # Ensure index directory exists
        self.index_path.parent.mkdir(exist_ok=True, parents=True)

        # Set default weights if not provided
        self.query_weight = query_weights or {"genres": 0.7, "keywords": 0.3}

        # Load models
        logger.info(f"Loading embedding model: {embedding_model}")
        self.roberta_model = SentenceTransformer(embedding_model, device=self.device)

        logger.info(f"Loading reranker model: {reranker_model}")
        self.cross_encoder = CrossEncoder(reranker_model, device=self.device)

        # Preprocess data
        original_length = len(self.df)
        self.df = self.preprocess_data()
        logger.info(f"Dataset size before preprocessing: {original_length}")
        logger.info(f"Dataset size after preprocessing: {len(self.df)}")

        # Load or create FAISS index
        self.faiss_index = None
        self._prepare_data()

    def _get_device(self) -> torch.device:
        """Determine the appropriate device (CPU or CUDA)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA for computation")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for computation")
        return device

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the movie dataset.

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing dataset")
        # Remove rows with missing genres or keywords
        df = self.df[~self.df["genres"].isnull()]
        df = df[~df["keywords"].isnull()]

        # Convert string lists to actual lists
        df["genres"] = df["genres"].apply(
            lambda x: [element.strip().lower() for element in x.split(",")]
        )
        df["keywords"] = df["keywords"].apply(
            lambda x: [element.strip().lower() for element in x.split(",")]
        )

        return df

    def compute_field_embeddings(
        self,
        text_lists: List[List[str]],
        field_name: str,
        model: SentenceTransformer,
        dimensions: int,
    ) -> np.ndarray:
        """
        Compute embeddings for a list of text lists.

        Args:
            text_lists: List of text lists to encode
            field_name: Name of the field (for logging)
            model: SentenceTransformer model to use
            dimensions: Dimensionality of embeddings

        Returns:
            Array of embeddings
        """
        try:
            final_embeddings = []
            for texts in tqdm(text_lists, desc=f"Encoding {field_name}"):
                if not texts:
                    # Use zero vector for empty text lists
                    final_embeddings.append(np.zeros(dimensions))
                else:
                    # Encode text list and compute mean embedding
                    embeddings = model.encode(
                        texts,
                        show_progress_bar=False,
                        batch_size=128,
                        device=self.device,
                    )
                    final_embeddings.append(np.mean(embeddings, axis=0))
            return np.array(final_embeddings)
        except Exception as e:
            logger.error(f"Error computing {field_name} embeddings: {str(e)}")
            raise

    def _create_embeddings(
        self, df: pd.DataFrame, model: SentenceTransformer, weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Create weighted embeddings for movies.

        Args:
            df: DataFrame with movie data
            model: SentenceTransformer model to use
            weights: Weights for different features

        Returns:
            Weighted embeddings
        """
        # Get embedding dimension
        test_embeddings = model.encode(
            ["test"], show_progress_bar=False, batch_size=1, device=self.device
        )
        embedding_dim = test_embeddings.shape[1]
        logger.info(f"Embedding dimension: {embedding_dim}")

        # Compute embeddings for each field
        logger.info("Computing genre embeddings")
        genres_embeddings = self.compute_field_embeddings(
            df["genres"].tolist(), "Genre", model, embedding_dim
        )

        logger.info("Computing keyword embeddings")
        keywords_embeddings = self.compute_field_embeddings(
            df["keywords"].tolist(), "Keywords", model, embedding_dim
        )

        # Create weighted embeddings
        logger.info(f"Creating weighted embeddings with weights: {weights}")
        weighted_embeddings = (
            genres_embeddings * weights["genres"]
            + keywords_embeddings * weights["keywords"]
        )

        return weighted_embeddings

    def _load_faiss_index(self, path: Union[str, Path]) -> Optional[faiss.Index]:
        """
        Load FAISS index from disk.

        Args:
            path: Path to the index file

        Returns:
            FAISS index or None if not found
        """
        path = Path(path)
        if path.exists():
            logger.info(f"Loading existing FAISS index from {path}")
            return faiss.read_index(str(path))
        logger.info(f"No existing index found at {path}")
        return None

    def _save_faiss_index(self, index: faiss.Index, path: Union[str, Path]) -> None:
        """
        Save FAISS index to disk.

        Args:
            index: FAISS index to save
            path: Path to save the index
        """
        path = Path(path)
        logger.info(f"Saving FAISS index to {path}")
        faiss.write_index(index, str(path))

    def _prepare_data(self) -> None:
        """Prepare data by loading or creating FAISS index."""
        # Try to load existing index
        self.faiss_index = self._load_faiss_index(self.index_path)

        if self.faiss_index is None:
            logger.info("Creating new FAISS index")
            # Create embeddings
            self.roberta_embeddings = self._create_embeddings(
                self.df, self.roberta_model, self.query_weight
            )
            logger.info(
                f"Created embeddings with shape {self.roberta_embeddings.shape}"
            )

            # Create and populate index
            self.faiss_index = faiss.IndexFlatIP(self.roberta_embeddings.shape[1])
            self.faiss_index.add(np.array(self.roberta_embeddings, dtype=np.float32))

            # Save index
            self._save_faiss_index(self.faiss_index, self.index_path)

    def batch_predict(
        self, encoder: CrossEncoder, pairs: List[List[str]], batch_size: int = 32
    ) -> np.ndarray:
        """
        Batch prediction with cross-encoder.

        Args:
            encoder: CrossEncoder model
            pairs: List of text pairs to encode
            batch_size: Batch size for prediction

        Returns:
            Array of scores
        """
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            results.extend(encoder.predict(batch))

        return np.array(results)

    def recommend(
        self,
        meta_data: Dict[str, List[str]],
        top_n: int = 10,
        rerank_topk: int = 20,
        relevance_weight: float = 0.7,
        popularity_weight: float = 0.3,
    ) -> pd.DataFrame:
        """
        Recommend movies based on MBTI preferences.

        Args:
            meta_data: Dictionary with genres and keywords for the MBTI type
            top_n: Number of top movies to return
            rerank_topk: Number of candidates to rerank
            relevance_weight: Weight for semantic relevance
            popularity_weight: Weight for movie popularity

        Returns:
            DataFrame with top movie recommendations
        """
        try:
            # Convert meta_data to DataFrame format
            query_df = pd.DataFrame([meta_data])

            # Create embeddings for the query
            query_embeddings = self._create_embeddings(
                query_df, self.roberta_model, self.query_weight
            )

            # Search FAISS index
            logger.info(f"Searching for top {rerank_topk} candidates")
            distances, indices = self.faiss_index.search(
                np.array(query_embeddings, dtype=np.float32), rerank_topk
            )

            # Get candidate movies
            candidates = self.df.iloc[indices[0]].copy()

            # Prepare texts for reranking
            query_text = " ".join(
                [", ".join(meta_data["genres"]), ", ".join(meta_data["keywords"])]
            )

            candidates_texts = []
            for _, row in candidates.iterrows():
                candidate_text = " ".join(
                    [", ".join(row["genres"]), ", ".join(row["keywords"])]
                )
                candidates_texts.append(candidate_text)

            # Create pairs for cross-encoder
            pairs = [[query_text, text] for text in candidates_texts]

            # Rerank with cross-encoder
            logger.info("Reranking candidates with cross-encoder")
            scores = self.batch_predict(self.cross_encoder, pairs)

            # Normalize FAISS distances
            max_distance = np.max(distances[0])
            if max_distance > 0:
                faiss_norm = 1 - distances[0] / max_distance
            else:
                faiss_norm = np.zeros_like(distances[0])

            # Combine scores (0.7 weight to cross-encoder, 0.3 to FAISS)
            candidates["similarity_score"] = 0.7 * scores + 0.3 * faiss_norm

            # Add popularity if available
            if "popularity" in candidates.columns:
                max_popularity = candidates["popularity"].max()
                if max_popularity > 0:
                    candidates["normalized_popularity"] = (
                        candidates["popularity"] / max_popularity
                    )
                else:
                    candidates["normalized_popularity"] = 0
            else:
                # Default popularity if not available
                candidates["normalized_popularity"] = 0.5

            # Create final score with relevance and popularity
            candidates["final_score"] = (
                relevance_weight * candidates["similarity_score"]
                + popularity_weight * candidates["normalized_popularity"]
            )

            # Sort and return top results
            candidates.sort_values(by=["final_score"], ascending=False, inplace=True)

            # Select relevant columns
            result_columns = [
                "title",
                "original_title",
                "revenue",
                "tagline",
                "poster_path",
                "popularity",
                "final_score",
                "overview",
                "genres",
                "keywords",
            ]

            # Filter columns that exist in the DataFrame
            available_columns = [
                col for col in result_columns if col in candidates.columns
            ]

            return candidates.head(top_n)[available_columns]

        except Exception as e:
            logger.error(f"Error in recommendation: {str(e)}")
            raise
