import json
import time
from tqdm import tqdm
from typing import Dict, Any
from .api import (
    BatchRequestCollector, get_batch_inference, load_batch_results, 
)


class GenerationPhase:
    
    PROMPT_PREFIX = "Answer the following question in plain text, without any additional formatting:\n\n{prompt}"
    
    def __init__(self, args):
        self.args = args
        self.model_name = args.model.model
        self.provider = args.model.provider
        self.max_completion_tokens = args.data.max_completion_tokens

        self.num_samples = args.data.num_gen_samples

        self.collector = BatchRequestCollector(
            model_name=args.model.model,
            provider=args.model.provider,
            cache_dir="./batch_cache",
        )
    
    def prepare_batch(self, dataset) -> str:
        """Prepare batch requests for all topics in dataset."""
        print(f"\n{'-'*60}")
        print("GENERATION PHASE - Preparing Batch Requests")
        print(f"{'-'*60}\n")
        
        for sample in tqdm(dataset, desc="Preparing generation requests"):
            topic = sample["topic"]
            prompt = sample["prompt_text"]
            formatted_prompt = self.PROMPT_PREFIX.format(prompt=prompt)
            
            # Add most likely generation request
            self.collector.add_request(
                prompt=formatted_prompt,
                params={"temperature": 0.0, "max_tokens": self.max_completion_tokens},
                metadata={"topic": topic, "type": "most_likely", "original_prompt": prompt}
            )
            
            # Add diverse generation requests
            for i in range(self.num_samples):
                self.collector.add_request(
                    prompt=formatted_prompt,
                    params={
                        "max_tokens": self.max_completion_tokens,
                        "temperature": 1.0,
                        "top_p": 1.0,
                    },
                    metadata={"topic": topic, "type": "diverse", "index": i, "original_prompt": prompt}
                )
        
        print(f"\nTotal requests prepared: {self.collector.num_requests}")
        
        # Submit batch
        batch_name = f"generation_{self.model_name.replace('/', '_')}_{int(time.time())}"
        batch_id = self.collector.submit_batch(batch_name)
        
        print(f"\n{'-'*60}")
        print(f"Batch submitted successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"Check status: python pipeline.py --check-status {batch_id}")
        print(f"Process results: python pipeline.py --process-results {batch_id} --phase-for-results generate")
        print(f"{'-'*60}\n")
        
        return batch_id
    
    def process_results(self, batch_id, state) -> Dict[str, Any]:
        """Process batch results and save to generations cache."""
        print(f"\n{'-'*60}")
        print("GENERATION PHASE - Processing Results")
        print(f"{'-'*60}\n")
        
        # Load results
        results = load_batch_results(batch_id, self.provider)
        
        # Load request map
        batch_handler = get_batch_inference(self.provider, cache_dir=f"./batch_cache/{self.provider}")
        
        # Find the request map file
        request_map = {}
        for f in batch_handler.cache_dir.glob("*_request_map.json"):
            with open(f, 'r') as file:
                map_content = json.load(file)
                # Check if any custom_id matches our results
                if any(cid in map_content for cid in results.keys()):
                    request_map = map_content
                    break
        
        if not request_map:
            print("Warning: Could not find request map, using raw results")
        
        # Organize results by topic
        generations_by_topic = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for custom_id, result in results.items():
            if custom_id not in request_map:
                continue
            
            usage = (result.raw_response or {}).get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            
            metadata = request_map[custom_id].get("metadata", {})
            topic = metadata.get("topic")
            gen_type = metadata.get("type")
            
            if not topic:
                continue
            
            if topic not in generations_by_topic:
                generations_by_topic[topic] = {
                    "generation_prompt": metadata.get("original_prompt", ""),
                    "most_likely_generation": None,
                    "diverse_generations": [None] * self.num_samples
                }
            
            if result.error:
                print(f"Error for {topic} ({gen_type}): {result.error}")
                continue
            
            response = result.response
            if gen_type == "most_likely":
                generations_by_topic[topic]["most_likely_generation"] = response
            elif gen_type == "diverse":
                idx = metadata.get("index", 0)
                generations_by_topic[topic]["diverse_generations"][idx] = response
        
        # Save to cache
        generations_cache = state.get_generations_cache()
        
        valid_count = 0
        for topic, gen_data in generations_by_topic.items():
            # Check if all generations are valid
            if gen_data["most_likely_generation"] and all(gen_data["diverse_generations"]):
                valid_count += 1
            else:
                print(f"Warning: Incomplete generations for topic: {topic}")
            generations_cache[topic] = gen_data
        
        generations_cache.sync()
        generations_cache.to_json()
        
        print(f"\nProcessed {valid_count} valid topic generations")
        print(f"Results saved to: {state.generations_path}")
        
        return {
            "valid_topics": valid_count,
            "total_topics": len(generations_by_topic),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }