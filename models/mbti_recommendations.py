import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import OneHotEncoder
import os
from typing import Dict
from tqdm import tqdm
from constant import mbti_mapping


class MBTI_Recommendation:
    def __init__(self, dataset):
        self.df = pd.read_csv(dataset)
        self.device = self.get_device()
        self.roberta_model = SentenceTransformer(
            "sentence-transformers/all-roberta-large-v1", device=self.device
        )
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-electra-base", device=self.device
        )
        self.query_weight = {"genres": 0.7, "keywords": 0.3}
        original_length = len(self.df)
        self.df = self.preprocess_data()
        print(f"The size of the dataset before preprocessing: {original_length}")
        print(f"The size of the dataset after preprocessing: {len(self.df)}")
        self.faiss_index = None
        self._prepare_data()

    def load_faiss_index(self, save_path):
        if os.path.exists(save_path):
            return faiss.read_index(save_path)
        return None

    def save_faiss_index(self, index, save_path):
        faiss.write_index(index, save_path)

    def get_device(self):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def preprocess_data(self):
        self.df = self.df[~self.df["genres"].isnull()]
        self.df = self.df[~self.df["keywords"].isnull()]
        self.df["genres"] = self.df["genres"].apply(
            lambda x: [element.lower() for element in x.split(",")]
        )
        self.df["keywords"] = self.df["keywords"].apply(
            lambda x: [element.lower() for element in x.split(",")]
        )
        return self.df

    def compute_field_embeddings(self, text_lists, field_name, model, dimensions):
        try:
            final_embeddings = []
            for texts in tqdm(text_lists, desc=f"Encoding Started for {field_name}"):
                if not texts:
                    final_embeddings.append(np.zeros(dimensions))
                else:
                    embeddings = model.encode(
                        texts,
                        show_progress_bar=False,
                        batch_size=128,
                        device=self.device,
                    )
                    final_embeddings.append(np.mean(embeddings, axis=0))
            return np.array(final_embeddings)
        except Exception as e:
            print(e)

    def _create_embeddings(self, df, model, weights):
        test_embeddings = model.encode(
            ["text"], show_progress_bar=True, batch_size=32, device=self.device
        )
        embedding_dim = test_embeddings.shape[1]

        genres_embeddings = self.compute_field_embeddings(
            df["genres"].tolist(), "Genre", model, embedding_dim
        )
        keywords_embeddings = self.compute_field_embeddings(
            df["keywords"].tolist(), "Keywords", model, embedding_dim
        )

        weighted_embeddings = (
            genres_embeddings * weights["genres"]
            + keywords_embeddings * weights["keywords"]
        )
        return weighted_embeddings

    def _prepare_data(self):
        self.faiss_index = self.load_faiss_index("roberta_index.idx")
        print(f"Index value: {self.faiss_index}")
        if self.faiss_index is None:
            print("Creating new index")
            self.roberta_embeddings = self._create_embeddings(
                self.df, self.roberta_model, self.query_weight
            )
            print("Embeddings have been generated")
            self.faiss_index = faiss.IndexFlatIP(self.roberta_embeddings.shape[1])
            self.faiss_index.add(np.array(self.roberta_embeddings, dtype=np.float32))
            self.save_faiss_index(self.faiss_index, "roberta_index.idx")

    def batch_predict(self, encoder, pairs, batch_size=32):
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            results.extend(encoder.predict(batch))
        final_results = np.array([np.mean(score) for score in results])
        return final_results

    def recommend_ab(self, meta_data, top_n=10, rerank_topk=20):
        try:
            query_df = pd.DataFrame([meta_data])
            query_embeddings = self._create_embeddings(
                query_df, self.roberta_model, self.query_weight
            )
            distances, indices = self.faiss_index.search(query_embeddings, rerank_topk)
            candidates = self.df.iloc[indices[0]].copy()
            query_text = f"{meta_data['genres']} {meta_data['keywords']}"
            candidates_texts = []
            for _, row in candidates.iterrows():
                candidates_texts.append(f"{row['genres']} {row['keywords']}")
            pairs = [[query_text, text] for text in candidates_texts]
            scores = self.batch_predict(self.cross_encoder, pairs)
            faiss_norm = (
                1 - distances[0] / np.max(distances)
                if np.max(distances) > 0
                else np.zeros_like(distances[0])
            )
            combined_score = 0.7 * scores + 0.3 * faiss_norm
            candidates["score"] = combined_score
            if "popularity" in candidates.columns:
                max_popularity = candidates["popularity"].max()
                if max_popularity > 0:
                    candidates["normalized_popularity"] = (
                        candidates["popularity"] / max_popularity
                    )
                else:
                    candidates["normalized_popularity"] = 0
            else:
                # If popularity doesn't exist in your dataset yet, you'll need to add it
                # For now, we'll use a placeholder with neutral values
                candidates["normalized_popularity"] = 0.5  # Default mid-range value

            # Step 2: Create a balanced final score incorporating both relevance and popularity
            # You can adjust these weights based on your prioritization
            relevance_weight = 0.7  # 70% weight to semantic relevance
            popularity_weight = 0.3  # 30% weight to popularity

            candidates["final_score"] = (
                relevance_weight * candidates["score"]
                + popularity_weight * candidates["normalized_popularity"]
            )

            # Step 3: Sort by the final blended score
            candidates.sort_values(by=["final_score"], ascending=False, inplace=True)
            top_results = candidates.head(top_n)
            return top_results[
                [
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
            ]
        except Exception as e:
            print(e)


if __name__ == "__main__":
    recommender = MBTI_Recommendation("movies_dataset.csv")
    meta_data = mbti_mapping["ESTP"]
    recommendations = recommender.recommend_ab(meta_data, top_n=10)
    print(recommendations.to_dict(orient="records"))
