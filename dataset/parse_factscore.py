# Adapted from FactScore.retrieval: 
# https://github.com/shmsw25/FActScore/blob/main/factscore/retrieval.py

import json
import time
import os

import sqlite3
import numpy as np
import pickle as pkl
import json
import datasets
from pprint import pprint
import sys

from rank_bm25 import BM25Okapi

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        
        if len(cursor.fetchall())==0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print (f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        
        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text)==str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip())>0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset+MAX_LENGTH])
                            offset += MAX_LENGTH
                
                psgs = [tokenizer.decode(tokens) for tokens in passages if np.sum([t not in [0, 2] for t in tokens])>0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        if not results:
            print(f"WARNING: {title} does not exist in '{self.db_path}'!!!")
            return None
        assert len(results)==1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = results[0][0]
        # results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        # assert len(results)>0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results

class Retrieval(object):

    def __init__(self, retrieval_type="gtr-t5-large", batch_size=None):
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type=="bm25" or retrieval_type.startswith("gtr-")
        
        self.encoder = None
        self.add_n = 0

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None

    def get_bm25_passages(self, query, passages, k):
        bm25 = BM25Okapi([psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages])
        self.add_n_embed += 1
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_gtr_passages(self, retrieval_query, passages, k):
        if self.encoder is None:
            self.load_encoder()
        inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
        passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device)
        self.add_n_embed += 1
        query_vectors = self.encoder.encode([retrieval_query], 
                                            batch_size=self.batch_size,
                                            device=self.encoder.device)[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]
    
    def get_most_similar_passages(self, atom, knowledge_source, k=5):
        topic = knowledge_source['title']
        retrieval_query = topic + " " + atom.strip()
        # cache_key = topic + "#" + retrieval_query
        
        passages = [{"title": topic, "text": para} for para in knowledge_source['text'].split(SPECIAL_SEPARATOR)]
        assert len(passages) > 0, f"`topic` in your data ({topic}) is likely to be not a valid title."
    
        if self.retrieval_type=="bm25":
            most_similar_psgs = self.get_bm25_passages(retrieval_query, passages, k)
        else:
            most_similar_psgs = self.get_gtr_passages(retrieval_query, passages, k)
        assert len(most_similar_psgs) in [k, len(passages)]
        self.add_n += 1
        
        return most_similar_psgs

def fetch_and_save_relevant_passages(src_db_path, dst_db_path, knowledge_source_path):
    """
    fetch the data used by factscore from wiki catalog and save as a separate .db file.

    Parameters
    ----------
    src_db_path : str
        Path to the factscore downloaded .db file. For example "./data/enwiki-20230401.db".
    dst_db_path : str
        Path to the new .db file (can be non-existent). For example "./data/factscore_data.db".
    knowledge_source_path: str
        Path to the document titles used by factscore. For example "./data/unlabeled/prompt_entities.txt".
        
    """
    conn_src = sqlite3.connect(src_db_path, check_same_thread=False)
    conn_dst = sqlite3.connect(dst_db_path, check_same_thread=False)
    c_src = conn_src.cursor()
    c_dst = conn_dst.cursor()

    with open(knowledge_source_path) as f:
        titles = f.read().splitlines()
    titles = tuple(titles)

    c_src.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = c_src.fetchall()
    assert len(table_names) == 1, f"more than one table exists in '{src_db_path}', check the download"
    
    # fetch relevant passages
    c_src.execute(f"SELECT * FROM {table_names[0][0]} WHERE title IN {titles}")
    passages = c_src.fetchall()
    # assert len(passages) == len(titles), f"some entries in '{knowledge_source_path}' do not exist in '{src_db_path}'"
    if len(passages) != len(titles):
        c_src.execute(f"SELECT title FROM {table_names[0][0]} WHERE title IN {titles}")
        fetched_titles = set([t[0] for t in c_src.fetchall()])
        src_titles = set(titles)
        print(f"WARNING: {src_titles - fetched_titles} do not exist in '{src_db_path}'!!!")
    
    # create new table at the destination
    # cursor.execute(f"CREATE TABLE biographies AS SELECT * FROM {table_names[0]} WHERE title IN {titles}")
    c_dst.execute("CREATE TABLE documents (title PRIMARY KEY, text);")
    c_dst.executemany("INSERT INTO documents VALUES (?,?)", passages)

    # connection.commit()
    conn_dst.commit()
    conn_dst.close()
    conn_src.close()

def generate_dataset(db_path,
                     prompt_entities_path,
                     save_copy=False):
    # db_path = "./dataset/factscore/factscore_knowledge_sources.db"
    # db_path = os.path.join(args.dataset_root_path, args.dataset, "factscore_knowledge_sources.db")
    
    db = DocDB(db_path)
    # data_entry = db.get_text_from_title("Kang Ji-hwan")
    # print(len(data_entry))
    # pprint(data_entry, sort_dicts=False)

    # retrieval = Retrieval(db, cache_path, embed_cache_path, batch_size=batch_size)

    with open(prompt_entities_path, "r") as f:
        topics = f.read().splitlines()
    # for topic in topics:
    #     passages = db.get_text_from_title(topic)
    #     if not passages:
    #         continue
    #     entry = {passages[0]['title']: []}
    #     for p in passages:
    #         entry[p['title']].append(p['text'])
    #     assert len(entry) == 1
    #     data_entries.update(entry)

    data_entries = {}
    for topic in topics:
        document = db.get_text_from_title(topic)
        if document:
            document = document.replace(SPECIAL_SEPARATOR, "")
            data_entries[topic] = document

    # assert data_entries, f"No data collected from {db_path}"

    def load_data():
        counter = 0
        for topic in data_entries.keys():
            yield {'topic': topic, 
                   'prompt_text': f"Tell me a bio of {topic}.",
                   'reference_text': data_entries[topic]}

            counter += 1

    # topic = "Suthida"
    # atom = "she complete military training courses and was appointed the the commander of the Special Operations Unit."
    # passages_raw = retrieval.db.get_text_from_title(topic)
    # passages = retrieval.get_passages(topic, atom, k=5)
    # pprint(passages_raw)
    # pprint(passages)

    dataset = datasets.Dataset.from_generator(load_data)

    # save_path = db_path[:db_path.rfind("/")]
    # json_path = os.path.join(save_path, "fs_dataset_parsed.json")
    # with open(json_path, "w") as f:
    #     json.dump(data_entries, f)

    # dataset = datasets.load_dataset('json', data_files=json_path, split='train')
    # dataset = datasets.Dataset.from_dict(data_entries)

    if save_copy:
        save_path = db_path[:db_path.rfind("/")]
        dataset.save_to_disk(os.path.join(save_path, "fs_dataset_parsed_copy.hf"))

    return dataset

# dataset = generate_dataset(None, "./dataset/factscore/data/unlabeled/prompt_entities.txt")
# retrieval = Retrieval(batch_size=256)
# print(dataset[1])
# pprint(retrieval.get_most_similar_passages(atom="she was titled as the first class", knowledge_source=dataset[0]))
