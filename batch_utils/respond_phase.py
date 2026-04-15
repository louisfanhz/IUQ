import os
import re
import json
import time
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from prompts import responder_prompts, uncertainty_metrics_prompts
from schemas import Claim, ClaimAnalysis, GenerationSample, TopicResult
from .api import (
    BatchRequestCollector, get_batch_inference, load_batch_results, 
)


class RespondPhase:    
    def __init__(self, args):        
        self.args = args
        self.model_name = args.model.model
        self.provider = args.model.provider
        self.max_completion_tokens = args.data.max_completion_tokens
        self.num_ans_per_question = args.data.num_ans_per_question

        self.collector = BatchRequestCollector(
            model_name=self.model_name,
            provider=self.provider,
            cache_dir="./batch_cache"
        )
        
    def prepare_batch(self, state) -> str:
        """Prepare batch requests for responding."""
        print(f"\n{'-'*60}")
        print("RESPOND PHASE - Preparing Batch Requests")
        print(f"{'-'*60}\n")
        
        analysis_cache = state.get_analysis_cache()
        generations_cache = state.get_generations_cache()
        
        for topic in tqdm(analysis_cache.cache.keys(), desc="Preparing respond requests"):
            topic_res = TopicResult(**analysis_cache[topic])
            gen_data = generations_cache[topic]
            topic_context = gen_data["generation_prompt"]
            
            for ga in topic_res.gen_analysis:
                for claim_idx, claim in enumerate(ga.claims):
                    for ca_idx, ca in enumerate(claim.claim_analysis):
                        # Generate answer requests
                        prompt = responder_prompts["respond"].format(
                            context=topic_context, question=ca.question
                        )
                        
                        for res_idx in range(self.num_ans_per_question):
                            self.collector.add_request(
                                prompt=prompt,
                                params={
                                    "max_tokens": self.max_completion_tokens,
                                    "temperature": 1.0,
                                    "top_p": 1.0,
                                },
                                logprobs=False,
                                metadata={
                                    "topic": topic,
                                    "gen_idx": ga.gen_idx,
                                    "claim_idx": claim_idx,
                                    "ca_idx": ca_idx,
                                    "res_idx": res_idx,
                                    "type": "answer",
                                    "claim": claim.content,
                                    "question": ca.question
                                },
                                custom_id_prefix="respond"
                            )
        
        print(f"\nTotal requests prepared: {self.collector.num_requests}")
        
        batch_name = f"respond_{self.model_name.replace('/', '_')}_{int(time.time())}"
        batch_id = self.collector.submit_batch(batch_name)
        
        print(f"\n{'-'*60}")
        print(f"Batch submitted successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"Check status: python pipeline.py --check-status {batch_id}")
        print(f"Process results: python pipeline.py --process-results {batch_id} --phase-for-results respond")
        print(f"{'-'*60}\n")
        
        return batch_id
    
    def process_results(self, batch_id, state) -> Dict[str, Any]:
        """Process respond phase results."""
        print(f"\n{'-'*60}")
        print("RESPOND PHASE - Processing Results")
        print(f"{'-'*60}\n")
        
        results = load_batch_results(batch_id, self.provider)
        
        # Load request map
        batch_handler = get_batch_inference(self.provider, cache_dir=f"./batch_cache/{self.provider}")
        request_map = {}
        for f in batch_handler.cache_dir.glob("*_request_map.json"):
            with open(f, 'r') as file:
                map_content = json.load(file)
                if any(cid in map_content for cid in results.keys()):
                    request_map = map_content
                    break
        
        # Organize answers
        answers_by_topic = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for custom_id, result in results.items():
            if custom_id not in request_map:
                continue
            
            usage = (result.raw_response or {}).get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            
            metadata = request_map[custom_id].get("metadata", {})
            if metadata.get("type") != "answer":
                continue
            
            topic = metadata.get("topic")
            gen_idx = metadata.get("gen_idx")
            claim_idx = metadata.get("claim_idx")
            ca_idx = metadata.get("ca_idx")
            res_idx = metadata.get("res_idx")
            
            key = (topic, gen_idx, claim_idx, ca_idx)
            if key not in answers_by_topic:
                answers_by_topic[key] = []
            
            if result.response:
                answer_data = {
                    "text": result.response,
                    "tokens": result.tokens,
                    "logprobs": result.logprobs,
                    "res_idx": res_idx
                }
                answers_by_topic[key].append(answer_data)
        
        # Update analysis cache
        analysis_cache = state.get_analysis_cache()
        topics_updated = set()
        
        for (topic, gen_idx, claim_idx, ca_idx), answers in answers_by_topic.items():
            if topic not in analysis_cache.cache:
                continue
            
            topic_res = TopicResult(**analysis_cache[topic])
            
            # Find the right claim analysis
            for ga in topic_res.gen_analysis:
                if ga.gen_idx != gen_idx:
                    continue
                
                if claim_idx < len(ga.claims):
                    claim = ga.claims[claim_idx]
                    if ca_idx < len(claim.claim_analysis):
                        ca = claim.claim_analysis[ca_idx]
                        
                        # Sort answers by res_idx
                        answers.sort(key=lambda x: x.get("res_idx", 0))
                        
                        # Format answers
                        ca.answers = [{
                            "text": a["text"],
                            "contradiction": None
                        } for a in answers]
            
            analysis_cache[topic] = topic_res.model_dump()
            topics_updated.add(topic)
        
        analysis_cache.sync()
        analysis_cache.to_json()
        
        print(f"Processed answers for {len(topics_updated)} topics")
        print(f"Results saved to: {state.analysis_path}")
        
        return {
            "topics_updated": len(topics_updated),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }


class FaithfulnessEvaluationPhase:    
    def __init__(self, args):
        self.args = args
        self.model_name = args.model.model
        self.provider = args.model.provider
        self.max_completion_tokens = args.data.max_completion_tokens

        self.collector = BatchRequestCollector(
            model_name=self.model_name,
            provider=self.provider,
            cache_dir="./batch_cache"
        )
    
    def prepare_batch(self, state) -> str:
        print(f"\n{'-'*60}")
        print("FAITHFULNESS EVALUATION - Preparing Batch Requests")
        print(f"{'-'*60}\n")
        
        analysis_cache = state.get_analysis_cache()
        
        for topic in tqdm(analysis_cache.cache.keys(), desc="Preparing faithfulness evaluation requests"):
            topic_res = TopicResult(**analysis_cache[topic])
            
            for ga in topic_res.gen_analysis:
                all_claims = ga.all_claims
                
                for claim_idx, claim in enumerate(ga.claims):
                    # Build context: claims from claim_idx down to 0 (reversed slice)
                    context = "\n".join(all_claims[claim_idx::-1])
                    
                    for ca_idx, ca in enumerate(claim.claim_analysis):
                        if ca.answers is None:
                            continue
                        
                        for ans_idx, answer in enumerate(ca.answers):
                            answer_text = answer["text"] if isinstance(answer, dict) else answer
                            
                            prompt = responder_prompts["contradiction"].format(
                                statement=answer_text, context=context
                            )
                            
                            self.collector.add_request(
                                prompt=prompt,
                                params={"temperature": 0.0, "max_tokens": self.max_completion_tokens},
                                metadata={
                                    "topic": topic,
                                    "gen_idx": ga.gen_idx,
                                    "claim_idx": claim_idx,
                                    "ca_idx": ca_idx,
                                    "ans_idx": ans_idx,
                                    "type": "contradiction"
                                },
                                custom_id_prefix="contradiction"
                            )
        
        print(f"\nTotal contradiction requests: {self.collector.num_requests}")
        
        batch_name = f"contradiction_{self.model_name.replace('/', '_')}_{int(time.time())}"
        batch_id = self.collector.submit_batch(batch_name)
        
        print(f"\n{'-'*60}")
        print(f"Contradiction batch submitted: {batch_id}")
        print(f"Check status: python pipeline.py --check-status {batch_id}")
        print(f"Process results: python pipeline.py --process-results {batch_id} --phase-for-results measure-contradiction")
        print(f"{'-'*60}\n")
        
        return batch_id
    
    def process_results(self, batch_id, state) -> Dict[str, Any]:
        print(f"\n{'-'*60}")
        print("FAITHFULNESS EVALUATION - Processing Results")
        print(f"{'-'*60}\n")
        
        results = load_batch_results(batch_id, self.provider)
        
        # Load request map
        batch_handler = get_batch_inference(self.provider, cache_dir=f"./batch_cache/{self.provider}")
        request_map = {}
        for f in batch_handler.cache_dir.glob("*_request_map.json"):
            with open(f, 'r') as file:
                map_content = json.load(file)
                if any(cid in map_content for cid in results.keys()):
                    request_map = map_content
                    break
        
        # Parse contradiction results
        faithfulness_map = {}  # (topic, gen_idx, claim_idx, ca_idx, ans_idx) -> float
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for custom_id, result in results.items():
            if custom_id not in request_map:
                continue
            
            usage = (result.raw_response or {}).get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            
            metadata = request_map[custom_id].get("metadata", {})
            if metadata.get("type") != "contradiction":
                continue
            
            topic = metadata.get("topic")
            gen_idx = metadata.get("gen_idx")
            claim_idx = metadata.get("claim_idx")
            ca_idx = metadata.get("ca_idx")
            ans_idx = metadata.get("ans_idx")
            
            if result.error:
                print(f"Error for {topic} gen={gen_idx} claim={claim_idx} ca={ca_idx} ans={ans_idx}: {result.error}")
                continue
            
            response = (result.response or "").strip()
            
            # Extract percentage: take the last sentence, find a number
            sentences = re.split(r'[.!?]+\s*', response)
            sentences = [s for s in sentences if s]
            if sentences:
                response = sentences[-1]
            
            percentage_match = re.search(r'(\d+)[.,%]?', response)
            if percentage_match:
                percentage = int(percentage_match.group(1))
                if 0 <= percentage <= 100:
                    faithfulness_map[(topic, gen_idx, claim_idx, ca_idx, ans_idx)] = percentage / 100.0
                else:
                    print(f"WARNING: Out of range contradiction {percentage} for {topic} gen={gen_idx} claim={claim_idx}, response={response}")
            else:
                print(f"WARNING: Could not parse response of contradiction for {topic} gen={gen_idx} claim={claim_idx}, due to missing percentage: {response}")

            if not faithfulness_map.get((topic, gen_idx, claim_idx, ca_idx, ans_idx), None):
                faithfulness_map[(topic, gen_idx, claim_idx, ca_idx, ans_idx)] = 0.0
        
        # Update analysis cache
        analysis_cache = state.get_analysis_cache()
        topics_updated = set()
        
        for (topic, gen_idx, claim_idx, ca_idx, ans_idx), contrdt_val in faithfulness_map.items():
            if topic not in analysis_cache.cache:
                continue
            
            topic_res = TopicResult(**analysis_cache[topic])
            
            for ga in topic_res.gen_analysis:
                if ga.gen_idx != gen_idx:
                    continue
                if claim_idx < len(ga.claims):
                    claim = ga.claims[claim_idx]
                    if ca_idx < len(claim.claim_analysis):
                        ca = claim.claim_analysis[ca_idx]
                        if ca.answers and ans_idx < len(ca.answers):
                            ca.answers[ans_idx]["contradiction"] = contrdt_val
            
            analysis_cache[topic] = topic_res.model_dump()
            topics_updated.add(topic)
        
        analysis_cache.sync()
        analysis_cache.to_json()
        
        print(f"Updated contradiction for {len(faithfulness_map)} answers across {len(topics_updated)} topics")
        return {
            "answers_updated": len(faithfulness_map),
            "topics_updated": len(topics_updated),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }
