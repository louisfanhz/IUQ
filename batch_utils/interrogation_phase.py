import json
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from schemas import AtomicClaims, CorrectnessResult, Claim, ClaimAnalysis, GenerationSample, TopicResult
from prompts import interrogator_prompts, evaluator_prompts
from .api import (
    BatchRequestCollector, get_batch_inference, load_batch_results, 
)

class InterrogationPhase:    
    def __init__(self, args):
        self.args = args
        self.model_name = args.model.model
        self.judge_model_name = args.model.judge
        self.dataset_name = args.data.dataset
        self.provider = args.model.provider
        self.judge_provider = "openai"
        self.max_completion_tokens = args.data.max_completion_tokens

        self.num_question_per_claim = args.data.num_question_per_claim

        self.collector = BatchRequestCollector(
            model_name=self.model_name,
            provider=self.provider,
            cache_dir="./batch_cache"
        )
        self.supportness_collector = BatchRequestCollector(
            model_name=self.model_name,
            provider=self.provider,
            cache_dir="./batch_cache"
        )
        self.correctness_collector = BatchRequestCollector(
            model_name=self.judge_model_name,
            provider=self.judge_provider,
            cache_dir="./batch_cache"
        )

    def prepare_claim_extraction_batch(self, state) -> str:
        """Prepare batch requests for interrogation."""
        print(f"\n{'-'*60}")
        print("INTERROGATION PHASE - Preparing Batch Requests")
        print(f"{'-'*60}\n")
        
        generations_cache = state.get_generations_cache()
        
        # Phase 1: Extract atomic claims
        print("Preparing atomic claim extraction requests...")
        for topic in tqdm(generations_cache.cache.keys(), desc="Claim extraction"):
            gen_data = generations_cache[topic]
            context = gen_data["generation_prompt"]
            
            for gen_idx, gen in enumerate(gen_data["diverse_generations"]):
                if gen == "invalid":
                    continue
                
                usr_prompt = interrogator_prompts["extract_ac_user_prompt_strict"].format(
                    context=context, text=gen
                )
                
                self.collector.add_request(
                    prompt=usr_prompt,
                    params={"temperature": 0.0, "max_tokens": self.max_completion_tokens},
                    system_prompt=interrogator_prompts["extract_ac_system_prompt"],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "atomic_claims",
                            "schema": AtomicClaims.model_json_schema(),
                        },
                    },
                    metadata={
                        "topic": topic,
                        "gen_idx": gen_idx,
                        "type": "extract_claims",
                        "context": context,
                        "generation": gen
                    },
                    custom_id_prefix="extract_claims"
                )
        
        print(f"\nTotal requests prepared: {self.collector.num_requests}")
        
        # Submit batch
        batch_name = f"claim_extraction_{self.model_name.replace('/', '_')}_{int(time.time())}"
        batch_id = self.collector.submit_batch(batch_name)
        
        print(f"\n{'-'*60}")
        print(f"Batch submitted successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"Check status: python pipeline.py --check-status {batch_id}")
        print(f"Process results: python pipeline.py --process-results {batch_id} --phase-for-results interrogate")
        print(f"{'-'*60}\n")
        
        return batch_id

    
    def process_claims_results(self, batch_id, state) -> Dict[str, Any]:
        """Process claim extraction results."""
        print(f"\n{'-'*60}")
        print("INTERROGATION PHASE - Processing Claim Extraction Results")
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
        
        # Organize claims by topic and gen_idx
        claims_by_topic = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for custom_id, result in results.items():
            if custom_id not in request_map:
                continue
            
            usage = (result.raw_response or {}).get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            
            metadata = request_map[custom_id].get("metadata", {})
            if metadata.get("type") != "extract_claims":
                continue
            
            topic = metadata.get("topic")
            gen_idx = metadata.get("gen_idx")
            
            if not topic or gen_idx is None:
                continue
            
            if topic not in claims_by_topic:
                claims_by_topic[topic] = {}
            
            if result.error:
                print(f"Error for {topic} gen_idx={gen_idx}: {result.error}")
                continue
            
            response = result.response or ""
            if not response:
                print(f"  Warning: no response from claim extraction for {topic} generation {gen_idx+1}")
                continue

            cleaned_claims = json.loads(response)["atomic_claims"]
            
            claims_by_topic[topic][gen_idx] = {
                "claims": cleaned_claims,
                "context": metadata.get("context"),
                "generation": metadata.get("generation")
            }
        
        # Save to analysis cache
        analysis_cache = state.get_analysis_cache()
        
        for topic, gen_claims in claims_by_topic.items():
            gen_analysis = []
            for gen_idx in sorted(gen_claims.keys()):
                claims_data = gen_claims[gen_idx]
                claims = [Claim(content=c, correctness=None, supportness_score=None, claim_analysis=[]) 
                         for c in claims_data["claims"]]
                
                gen_analysis.append(GenerationSample(
                    gen_idx=gen_idx,
                    all_claims=claims_data["claims"],
                    all_questions=[],
                    claims=claims
                ))
            
            topic_result = TopicResult(gen_analysis=gen_analysis)
            analysis_cache[topic] = topic_result.model_dump()
        
        analysis_cache.sync()
        analysis_cache.to_json()
        
        print(f"Processed claims for {len(claims_by_topic)} topics")
        print(f"Results saved to: {state.analysis_path}")
        
        return {
            "topics_processed": len(claims_by_topic),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }
    
    def prepare_questions_batch(self, state) -> str:
        """Prepare batch for question generation (after claims are extracted)."""
        print(f"\n{'-'*60}")
        print("INTERROGATION PHASE - Preparing Question Generation Requests")
        print(f"{'-'*60}\n")
        
        analysis_cache = state.get_analysis_cache()
        generations_cache = state.get_generations_cache()
        
        for topic in tqdm(analysis_cache.cache.keys(), desc="Question generation"):
            topic_res = TopicResult(**analysis_cache[topic])
            gen_data = generations_cache[topic]
            context = gen_data["generation_prompt"]
            
            for ga in topic_res.gen_analysis:
                for claim_idx, claim in enumerate(ga.claims):
                    for q_idx in range(self.num_question_per_claim):
                        usr_prompt = interrogator_prompts["q_from_single_claim_user_prompt"].format(
                            context=context, claim=claim.content
                        )
                        
                        self.collector.add_request(
                            prompt=usr_prompt,
                            params={
                                "max_tokens": self.max_completion_tokens,
                                "temperature": 1.0,
                                "top_p": 1.0,
                            },
                            system_prompt=interrogator_prompts.get("q_from_single_claim_system_prompt"),
                            metadata={
                                "topic": topic,
                                "gen_idx": ga.gen_idx,
                                "claim_idx": claim_idx,
                                "q_idx": q_idx,
                                "type": "generate_question"
                            },
                            custom_id_prefix="generate_question"
                        )
        
        print(f"\nTotal requests prepared: {self.collector.num_requests}")
        
        batch_name = f"question_{self.model_name.replace('/', '_')}_{int(time.time())}"
        batch_id = self.collector.submit_batch(batch_name)
        
        print(f"\n{'-'*60}")
        print(f"Batch submitted successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"{'-'*60}\n")
        
        return batch_id
    
    def process_questions_results(self, batch_id, state) -> Dict[str, Any]:
        """Process question generation results."""
        print(f"\n{'-'*60}")
        print("INTERROGATION PHASE - Processing Question Generation Results")
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
        
        # Organize questions
        questions_by_topic = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for custom_id, result in results.items():
            if custom_id not in request_map:
                continue
            
            usage = (result.raw_response or {}).get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            
            metadata = request_map[custom_id].get("metadata", {})
            if metadata.get("type") != "generate_question":
                continue
            
            topic = metadata.get("topic")
            gen_idx = metadata.get("gen_idx")
            claim_idx = metadata.get("claim_idx")
            q_idx = metadata.get("q_idx")
            
            if topic not in questions_by_topic:
                questions_by_topic[topic] = {}
            if gen_idx not in questions_by_topic[topic]:
                questions_by_topic[topic][gen_idx] = {}
            if claim_idx not in questions_by_topic[topic][gen_idx]:
                questions_by_topic[topic][gen_idx][claim_idx] = []
            
            if result.response:
                questions_by_topic[topic][gen_idx][claim_idx].append(result.response)
        
        # Update analysis cache
        analysis_cache = state.get_analysis_cache()
        
        for topic, gen_questions in questions_by_topic.items():
            if topic not in analysis_cache.cache:
                continue
            
            topic_res = TopicResult(**analysis_cache[topic])
            
            for ga in topic_res.gen_analysis:
                if ga.gen_idx not in gen_questions:
                    continue
                
                claim_questions = gen_questions[ga.gen_idx]
                all_questions = []
                
                for claim_idx, claim in enumerate(ga.claims):
                    if claim_idx in claim_questions:
                        # Remove duplicates
                        unique_qs = list(dict.fromkeys(claim_questions[claim_idx]))
                        claim.claim_analysis = [ClaimAnalysis(question=q) for q in unique_qs]
                        all_questions.extend(unique_qs)
                
                ga.all_questions = all_questions
            
            analysis_cache[topic] = topic_res.model_dump()
        
        analysis_cache.sync()
        analysis_cache.to_json()
        
        print(f"Processed questions for {len(questions_by_topic)} topics")
        
        return {
            "topics_processed": len(questions_by_topic),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }

    def _init_ref_evaluator(self):
        """Initialize the reference evaluator for local passage retrieval (lazy init)."""
        if hasattr(self, '_ref_eval'):
            return
        if self.dataset_name == "factscore":
            from .evaluator import FactScoreEvaluator
            self._ref_eval = FactScoreEvaluator(
                db_path=self.args.data.factscore_db_path,
                ref_doc_retrieval_k=self.args.data.ref_doc_retrieval_k,
                retrieval_type="gtr",
            )
        elif self.dataset_name == "longfact":
            from .evaluator import LongFactEvaluator
            self._ref_eval = LongFactEvaluator(
                db_path=self.args.data.longfact_data_path,
                ref_doc_retrieval_k=self.args.data.ref_doc_retrieval_k
            )
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not supported for correctness evaluation")

    def _get_reference(self, topic: str, claim: str) -> str:
        """Retrieve reference passage for a claim using local retrieval."""
        if self.dataset_name == "factscore":
            return self._ref_eval.retrieval.get_passages(topic, claim, k=self.args.data.ref_doc_retrieval_k)
        elif self.dataset_name == "longfact":
            return self._ref_eval.retrieve_relevant_passages(topic, claim, k=self.args.data.ref_doc_retrieval_k)

    def prepare_correctness_batch(self, state) -> str:
        """Prepare batch for correctness evaluation."""
        print(f"\n{'-'*60}")
        print("INTERROGATION EVAL - Preparing Correctness Batch")
        print(f"Eval model: {self.judge_model_name}")
        print(f"{'-'*60}\n")
        
        self._init_ref_evaluator()
        
        analysis_cache = state.get_analysis_cache()
        
        for topic in tqdm(analysis_cache.cache.keys(), desc="Correctness requests"):
            topic_res = TopicResult(**analysis_cache[topic])
            
            for ga in topic_res.gen_analysis:
                for claim_idx, claim in enumerate(ga.claims):
                    reference = self._get_reference(topic, claim.content)
                    
                    sys_prompt = evaluator_prompts["eval_claims_from_reference_system_prompt"]
                    usr_prompt = evaluator_prompts["eval_claims_from_reference_user_prompt_single"].format(
                        reference=reference, claim=claim.content
                    )
                    
                    self.correctness_collector.add_request(
                        prompt=usr_prompt,
                        params={"temperature": 0.0, "max_tokens": 500},
                        system_prompt=sys_prompt,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "correctness",
                                "schema": CorrectnessResult.model_json_schema(),
                            },
                        },
                        metadata={
                            "topic": topic,
                            "gen_idx": ga.gen_idx,
                            "claim_idx": claim_idx,
                            "type": "correctness"
                        },
                        custom_id_prefix="correctness"
                    )
        
        print(f"\nTotal correctness requests: {self.correctness_collector.num_requests}")
        
        batch_name = f"correctness_{self.judge_model_name.replace('/', '_')}_{int(time.time())}"
        batch_id = self.correctness_collector.submit_batch(batch_name)
        
        print(f"\nCorrectness batch submitted: {batch_id}")
        return batch_id
    
    def process_correctness_results(self, batch_id, state) -> Dict[str, Any]:
        """Process correctness evaluation results and update claims."""
        print(f"\n{'-'*60}")
        print("INTERROGATION EVAL - Processing Correctness Results")
        print(f"{'-'*60}\n")
        
        results = load_batch_results(batch_id, "openai")
        
        # Load request map
        batch_handler = get_batch_inference("openai", cache_dir="./batch_cache/openai")
        request_map = {}
        for f in batch_handler.cache_dir.glob("*_request_map.json"):
            with open(f, 'r') as file:
                map_content = json.load(file)
                if any(cid in map_content for cid in results.keys()):
                    request_map = map_content
                    break
        
        # Organize correctness results: (topic, gen_idx, claim_idx) -> "correct"/"incorrect"
        correctness_map = {}
        
        for custom_id, result in results.items():
            if custom_id not in request_map:
                continue
            metadata = request_map[custom_id].get("metadata", {})
            if metadata.get("type") != "correctness":
                continue
            
            topic = metadata.get("topic")
            gen_idx = metadata.get("gen_idx")
            claim_idx = metadata.get("claim_idx")
            
            if result.error:
                print(f"Error for {topic} gen_idx={gen_idx} claim_idx={claim_idx}: {result.error}")
                continue
            
            response = result.response or ""
            if not response:
                print(f"Warning: empty response for {topic} gen_idx={gen_idx} claim_idx={claim_idx}")
                continue
            
            correctness = json.loads(response)["correctness"]
            # if correctness == "not_enough_information":
            if correctness != "correct":
                correctness = "incorrect"
            
            correctness_map[(topic, gen_idx, claim_idx)] = correctness
        
        # Update analysis cache
        analysis_cache = state.get_analysis_cache()
        topics_updated = set()
        
        for (topic, gen_idx, claim_idx), correctness in correctness_map.items():
            if topic not in analysis_cache.cache:
                continue
            
            topic_res = TopicResult(**analysis_cache[topic])
            for ga in topic_res.gen_analysis:
                if ga.gen_idx == gen_idx and claim_idx < len(ga.claims):
                    ga.claims[claim_idx].correctness = correctness
            
            analysis_cache[topic] = topic_res.model_dump()
            topics_updated.add(topic)
        
        analysis_cache.sync()
        analysis_cache.to_json()
        
        print(f"Updated correctness for {len(correctness_map)} claims across {len(topics_updated)} topics")
        return {"claims_updated": len(correctness_map), "topics_updated": len(topics_updated)}

    
    def prepare_supportness_batch(self, state) -> str:
        """Prepare batch for supported score evaluation (uses main model)."""
        print(f"\n{'-'*60}")
        print("INTERROGATION EVAL - Preparing Supported Score Batch")
        print(f"Model: {self.model_name} ({self.provider})")
        print(f"{'-'*60}\n")
        
        analysis_cache = state.get_analysis_cache()
        generations_cache = state.get_generations_cache()
        
        for topic in tqdm(analysis_cache.cache.keys(), desc="Supported score requests"):
            topic_res = TopicResult(**analysis_cache[topic])
            gen_data = generations_cache[topic]
            all_gens = gen_data["diverse_generations"]
            
            for ga in topic_res.gen_analysis:
                for claim_idx, claim in enumerate(ga.claims):
                    for psg_idx, passage in enumerate(all_gens):
                        if passage == "invalid":
                            continue
                        
                        usr_prompt = evaluator_prompts["from_generations_user_prompt_strict"].format(
                            passage=passage, claim=claim.content
                        )
                        
                        self.supportness_collector.add_request(
                            prompt=usr_prompt,
                            params={"temperature": 0.0, "max_tokens": 500},
                            metadata={
                                "topic": topic,
                                "gen_idx": ga.gen_idx,
                                "claim_idx": claim_idx,
                                "psg_idx": psg_idx,
                                "n_passages": len(all_gens),
                                "type": "supported"
                            },
                            custom_id_prefix="supportness"
                        )
        
        print(f"\nTotal supported score requests: {self.supportness_collector.num_requests}")
        
        batch_name = f"supported_{self.model_name.replace('/', '_')}_{int(time.time())}"
        batch_id = self.supportness_collector.submit_batch(batch_name)
        
        print(f"\nSupported score batch submitted: {batch_id}")
        return batch_id
    

    def process_supportness_results(self, batch_id, state) -> Dict[str, Any]:
        """Process supported score results and update claims."""        
        print(f"\n{'-'*60}")
        print("INTERROGATION EVAL - Processing Supported Score Results")
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
        
        # Collect support votes: (topic, gen_idx, claim_idx) -> list of True/False
        support_votes = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for custom_id, result in results.items():
            if custom_id not in request_map:
                continue
            
            usage = (result.raw_response or {}).get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            
            metadata = request_map[custom_id].get("metadata", {})
            if metadata.get("type") != "supported":
                continue
            
            topic = metadata.get("topic")
            gen_idx = metadata.get("gen_idx")
            claim_idx = metadata.get("claim_idx")
            
            if result.error:
                continue
            
            response = (result.response or "").lower().strip()
            is_supported = "true" in response or "yes" in response
            
            key = (topic, gen_idx, claim_idx)
            if key not in support_votes:
                support_votes[key] = []
            support_votes[key].append(is_supported)
        
        # Average votes to get supported_score per claim
        analysis_cache = state.get_analysis_cache()
        topics_updated = set()
        
        for (topic, gen_idx, claim_idx), votes in support_votes.items():
            if topic not in analysis_cache.cache:
                continue
            
            supported_score = np.mean(votes).item()
            
            topic_res = TopicResult(**analysis_cache[topic])
            for ga in topic_res.gen_analysis:
                if ga.gen_idx == gen_idx and claim_idx < len(ga.claims):
                    ga.claims[claim_idx].supportness_score = supported_score
            
            analysis_cache[topic] = topic_res.model_dump()
            topics_updated.add(topic)
        
        analysis_cache.sync()
        analysis_cache.to_json()
        
        print(f"Updated supported_score for {len(support_votes)} claims across {len(topics_updated)} topics")
        return {
            "claims_updated": len(support_votes),
            "topics_updated": len(topics_updated),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }
