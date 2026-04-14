import json
import random
import os
import datasets

from rich import print as rprint

def select_random_entries(directory, num_select_entries, flatten=True):
    # Get all jsonl files in the directory
    jsonl_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    selected_entries = {}
    
    for file_name in jsonl_files:
        file_path = os.path.join(directory, file_name)
        entries = []
        
        # Read the jsonl file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        # Randomly select entries
        if num_select_entries == -1:
            selected_entries[file_name] = entries
        else:
            selected = random.sample(entries, num_select_entries)
            selected_entries[file_name] = selected

    if flatten:
        return [{"prompt": entry["prompt"]} for entries in selected_entries.values() for entry in entries]

    return selected_entries

def save_entries_to_jsonl(entries, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

# def generate_dataset(longfact_dir, 
#                      evenly_select_from_topics=True,
#                      num_select_entries=5,
#                      seed=42):
#     original_state = random.getstate()
#     random.seed(seed)

#     # longfact_dir = "./dataset/longfact/longfact-objects_gpt4_01-12-2024_noduplicates"
#     # output_file = "./dataset/longfact/longfact-objects_selected_entities.jsonl"
    
#     # Randomly select n prompts for each topic from longfact prompt sets
#     if evenly_select_from_topics:
#         selected_entries = select_random_entries(longfact_dir, num_select_entries=num_select_entries)
#     else:
#         selected_entries = select_random_entries(longfact_dir, num_select_entries=-1)
#     # save_entries_to_jsonl(selected_entries, output_file)

#     rprint(selected_entries)

#     def load_data():
#         for entry in selected_entries:
#             yield {'topic': entry["prompt"], 
#                    'prompt_text': entry["prompt"]}

#     dataset = datasets.Dataset.from_generator(load_data)

#     # revert back random state
#     random.setstate(original_state)

#     return dataset

def generate_dataset(data_path):
    with open(data_path, "r") as f:
        longfact_dict = json.load(f)
        data = [{"prompt": entry["prompt"]} for longfact_file in longfact_dict.values() for entry in longfact_file]

    def load_data():
        for entry in data:
            yield {'topic': entry["prompt"], 
                   'prompt_text': entry["prompt"]}

    dataset = datasets.Dataset.from_generator(load_data)

    return dataset