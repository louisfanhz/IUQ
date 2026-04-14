import os
import numpy as np
import json
import wikipedia
from tqdm import tqdm
from typing import List, Union
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

from .factscore_utils import DocDB, RetrievalEasy, Retrieval


class FactScoreEvaluator:
    def __init__(
        self, 
        db_path: str="./dataset/factscore/enwiki-20230401.db",
        ref_doc_retrieval_k: int = 5,
        retrieval_type: str="gtr"
    ):
        self.retrieval_type = retrieval_type
        if self.retrieval_type == "easy":
            self.retrieval = RetrievalEasy(DocDB(db_path))
        elif self.retrieval_type == "gtr":
            dir_path = os.path.dirname(db_path)
            cache_path = os.path.join(dir_path, "retrieval-cache.json")
            embed_cache_path = os.path.join(dir_path, "retrieval-embed-cache.pkl")
            self.retrieval = Retrieval(DocDB(db_path), cache_path, embed_cache_path)
        else:
            raise ValueError(f"Invalid retrieval type: {self.retrieval_type}")


class LongFactEvaluator:
    def __init__(
        self, 
        db_path: str, 
        ref_doc_retrieval_k: int = 5,
        text_encoder: str="all-MiniLM-L6-v2"
    ) -> None:
        wikipedia.set_lang('en')
        self.text_encoder = SentenceTransformer("sentence-transformers/" + text_encoder).cuda().eval()
        self.text_chunks = {}
        self.text_embeddings = {}

        with open(db_path, "r") as f:
            longfact_dict = json.load(f)
            dataset = [{"topic": entry["prompt"], "wiki_entity": entry["wiki_entity"]} for longfact_file in longfact_dict.values() 
                                                                            for entry in longfact_file]
        for entry in tqdm(dataset, desc="Initializing longfact dataset"):
            wiki_page = self._get_wiki_page(entry["wiki_entity"])
            # wiki_page = wikipedia.search("Solomon R. Guggenheim Museum")
            self._make_text_embeddings(entry["topic"], entry["wiki_entity"], wiki_page)

    def _get_wiki_page(self, entity_name):
        try:
            page = wikipedia.page(entity_name, auto_suggest=False)
            return page.content
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error, multiple articles found: {e.options}")
        except wikipedia.exceptions.PageError:
            print(f"Page not found for {entity_name}")

    def _make_text_embeddings(self, topic: str, wiki_title: str, text: str) -> np.ndarray:
        try:
            chunks = self._create_chunks(wiki_title, text)
        except Exception as e:
            print(f"Error creating chunks for {wiki_title}: {e}")
            raise e
        self.text_chunks[topic] = chunks
        
        embeddings = self.text_encoder.encode(chunks, convert_to_numpy=True, device=self.text_encoder.device)
        self.text_embeddings[topic] = embeddings

    def _create_chunks(self, wiki_title: str, text: str, max_chunk_size: int = 256) -> List[str]:
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
        
            # If adding this sentence would exceed max_chunk_size, create a new chunk
            if current_chunk and current_word_count + sentence_words > max_chunk_size:
                chunk_text = wiki_title + ": " + " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = wiki_title + ": " + " ".join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks

    def get_query_embeddings(self, retrieval_query):
        if isinstance(retrieval_query, str):
            query_vectors = self.text_encoder.encode([retrieval_query],
                                                    convert_to_numpy=True,
                                                    device=self.text_encoder.device)[0]
        elif isinstance(retrieval_query, list) and len(retrieval_query) > 0:
            query_vectors = self.text_encoder.encode(retrieval_query,
                                                    convert_to_numpy=True,
                                                    device=self.text_encoder.device) 
        else:
            raise ValueError("Invalid retrieval query")
        return query_vectors

    def retrieve_relevant_passages(self, topic: str, query: Union[str, List[str]], k: int = 5) -> str:
        if topic not in self.text_embeddings:
            raise ValueError(f"{topic} should be in the LongFactEvaluator dataset, but is not.")
            
        # query_embedding = self.text_encoder.encode(query, convert_to_numpy=True, device=self.text_encoder.device)
        query_embeddings = self.get_query_embeddings(query)
        passage_embeddings = self.text_embeddings[topic]
        scores = np.inner(query_embeddings, passage_embeddings)
        indices = np.argsort(-scores, axis=-1)
        # indices = np.argsort(-scores, axis=-1)[:k]
        if indices.ndim > 1:
            # In most cases with batched claims evaluation, the ranked passages for multiple claims contain same passages in k<5
            indices = indices[0, :k]
        elif indices.ndim == 1:
            indices = indices[:k]

        ranked_passages = [self.text_chunks[topic][i] for i in indices]
        return "\n".join(ranked_passages)