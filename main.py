"""
IUQ (Interrogation-based Uncertainty Quantification) Pipeline
using regular (non-batch) API requests.

Supports OpenAI and TogetherAI providers.
No dataset required — starts from a single example prompt.
"""

import os
import re
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from omegaconf import OmegaConf
from rich import print as rprint

from schemas import AtomicClaims, Claim, ClaimAnalysis, GenerationSample, TopicResult
from prompts import interrogator_prompts, evaluator_prompts, responder_prompts
from utils.api import chat_completion

GENERATION_PREFIX = "Answer the following question in plain text, without any additional formatting:\n\n{prompt}"

TOGETHER_MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2-VL-72B-Instruct",
    "google/gemma-2-27b-it",
    "google/gemma-3n-E4B-it",
    "google/gemma-4-31B-it",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "moonshotai/Kimi-K2.5",
    "MiniMaxAI/MiniMax-M2.7",
]

PHASE_ORDER = [
    "generate",
    "claim_extraction",
    "question_generation",
    "supportness",
    "respond",
    "faithfulness",
]


# ---------------------------------------------------------------------------
# Intermediate-result cache (mirrors batch pipeline's CacheFileManager + PipelineState)
# ---------------------------------------------------------------------------

class PipelineCache:
    """JSON-backed cache that persists generations, analysis, and phase status.

    Files written under *result_dir*:
        {prefix}_manifest.json      – phase completion status
        {prefix}_generations.json   – output of Phase 1
        {prefix}_analysis.json      – TopicResult, updated after Phases 2-6
    """

    def __init__(self, result_dir: str, prefix: str, config: OmegaConf):
        self.dir = Path(result_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

        self.manifest_path = self.dir / f"{prefix}_manifest.json"
        self.generations_path = self.dir / f"{prefix}_generations.json"
        self.analysis_path = self.dir / f"{prefix}_analysis.json"

        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self._manifest = {
                "created_at": time.time(),
                "config": OmegaConf.to_container(config, resolve=True),
                "phases": {p: "pending" for p in PHASE_ORDER},
            }
            self._save_manifest()

    # -- manifest helpers ---------------------------------------------------

    def phase_status(self, phase: str) -> str:
        return self._manifest["phases"][phase]

    def is_phase_done(self, phase: str) -> bool:
        return self._manifest["phases"][phase] == "completed"

    def mark_phase_done(self, phase: str):
        self._manifest["phases"][phase] = "completed"
        self._manifest[f"{phase}_completed_at"] = time.time()
        self._save_manifest()

    def _save_manifest(self):
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    # -- generations --------------------------------------------------------

    def save_generations(self, data: Dict[str, Any]):
        self._write_json(self.generations_path, data)

    def load_generations(self) -> Optional[Dict[str, Any]]:
        return self._read_json(self.generations_path)

    # -- analysis (TopicResult) ---------------------------------------------

    def save_analysis(self, topic_result: TopicResult):
        self._write_json(self.analysis_path, topic_result.model_dump())

    def load_analysis(self) -> Optional[TopicResult]:
        data = self._read_json(self.analysis_path)
        if data is not None:
            return TopicResult(**data)
        return None

    # -- generic JSON helpers -----------------------------------------------

    @staticmethod
    def _write_json(path: Path, obj: Any):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None


# ---------------------------------------------------------------------------
# Phase 1: Generation
# ---------------------------------------------------------------------------

def phase_generate(topic: str, prompt: str, args) -> Dict[str, Any]:
    """Generate 1 greedy + N stochastic completions for the input prompt."""
    rprint(f"\n[bold]{'='*60}[/bold]")
    rprint("[bold]PHASE 1: Generation[/bold]")
    rprint(f"[bold]{'='*60}[/bold]\n")

    model = args.model.model
    provider = args.model.provider
    max_tokens = args.data.max_completion_tokens
    num_samples = args.data.num_gen_samples

    formatted = GENERATION_PREFIX.format(prompt=prompt)
    messages = [{"role": "user", "content": formatted}]

    rprint("Generating most likely response (temperature=0)...")
    result = chat_completion(provider=provider, model=model, messages=messages,
                             temperature=0.0, max_tokens=max_tokens)
    most_likely = result["response"]
    rprint(f"  Done. ({len(most_likely.split())} words)")

    diverse: List[str] = []
    for i in range(num_samples):
        rprint(f"Generating diverse sample {i+1}/{num_samples} (temperature=1.0)...")
        result = chat_completion(provider=provider, model=model, messages=messages,
                                 temperature=1.0, max_tokens=max_tokens, top_p=1.0)
        diverse.append(result["response"])
        rprint(f"  Done. ({len(diverse[-1].split())} words)")

    generations = {
        "generation_prompt": prompt,
        "most_likely_generation": most_likely,
        "diverse_generations": diverse,
    }

    rprint(f"\n[green]Generation complete: 1 greedy + {num_samples} diverse samples[/green]")
    return generations


# ---------------------------------------------------------------------------
# Phase 2: Claim Extraction
# ---------------------------------------------------------------------------

def phase_extract_claims(generations: Dict, args) -> TopicResult:
    """Extract atomic claims from each diverse generation using structured output."""
    rprint(f"\n[bold]{'='*60}[/bold]")
    rprint("[bold]PHASE 2: Claim Extraction[/bold]")
    rprint(f"[bold]{'='*60}[/bold]\n")

    model = args.model.model
    provider = args.model.provider
    max_tokens = args.data.max_completion_tokens
    context = generations["generation_prompt"]

    gen_analysis: List[GenerationSample] = []
    for gen_idx, gen_text in enumerate(generations["diverse_generations"]):
        rprint(f"Extracting claims from generation {gen_idx+1}/{len(generations['diverse_generations'])}...")

        usr_prompt = interrogator_prompts["extract_ac_user_prompt_strict"].format(
            context=context, text=gen_text
        )
        messages = [
            {"role": "system", "content": interrogator_prompts["extract_ac_system_prompt"]},
            {"role": "user", "content": usr_prompt},
        ]

        result = chat_completion(provider=provider, model=model, messages=messages,
                                 temperature=0.0, max_tokens=max_tokens,
                                 response_format=AtomicClaims)

        if not result["response"]:
            rprint(f"  Warning: no response from claim extraction for generation {gen_idx+1}")
            continue

        cleaned = json.loads(result["response"])["atomic_claims"]

        claims = [Claim(content=c, correctness=None, supportness_score=None, claim_analysis=[])
                  for c in cleaned]

        gen_analysis.append(GenerationSample(
            gen_idx=gen_idx, all_claims=cleaned, all_questions=[], claims=claims,
        ))
        rprint(f"  Extracted {len(cleaned)} claims")

    topic_result = TopicResult(gen_analysis=gen_analysis)
    total_claims = sum(len(ga.claims) for ga in gen_analysis)
    rprint(f"\n[green]Claim extraction complete: {total_claims} claims from "
           f"{len(gen_analysis)} generations[/green]")
    return topic_result


# ---------------------------------------------------------------------------
# Phase 3: Question Generation
# ---------------------------------------------------------------------------

def phase_generate_questions(topic_result: TopicResult, context: str, args) -> TopicResult:
    """Generate probing questions from each claim."""
    rprint(f"\n[bold]{'='*60}[/bold]")
    rprint("[bold]PHASE 3: Question Generation[/bold]")
    rprint(f"[bold]{'='*60}[/bold]\n")

    model = args.model.model
    provider = args.model.provider
    max_tokens = args.data.max_completion_tokens
    num_q = args.data.num_question_per_claim

    total_questions = 0
    for ga in topic_result.gen_analysis:
        rprint(f"Processing generation {ga.gen_idx+1}...")
        for claim in ga.claims:
            questions: List[str] = []
            for _ in range(num_q):
                usr_prompt = interrogator_prompts["q_from_single_claim_user_prompt"].format(
                    context=context, claim=claim.content
                )
                messages = [
                    {"role": "system", "content": interrogator_prompts["q_from_single_claim_system_prompt"]},
                    {"role": "user", "content": usr_prompt},
                ]
                result = chat_completion(provider=provider, model=model, messages=messages,
                                         temperature=1.0, max_tokens=max_tokens, top_p=1.0)
                questions.append(result["response"])

            unique_qs = list(dict.fromkeys(questions))
            claim.claim_analysis = [ClaimAnalysis(question=q) for q in unique_qs]
            ga.all_questions.extend(unique_qs)
            total_questions += len(unique_qs)

    rprint(f"\n[green]Question generation complete: {total_questions} questions[/green]")
    return topic_result


# ---------------------------------------------------------------------------
# Phase 4: Supportness Evaluation
# ---------------------------------------------------------------------------

def phase_evaluate_supportness(topic_result: TopicResult, generations: Dict, args) -> TopicResult:
    """For each claim, check whether it is supported by each diverse generation."""
    rprint(f"\n[bold]{'='*60}[/bold]")
    rprint("[bold]PHASE 4: Supportness Evaluation[/bold]")
    rprint(f"[bold]{'='*60}[/bold]\n")

    model = args.model.model
    provider = args.model.provider
    all_gens = generations["diverse_generations"]

    claims_evaluated = 0
    for ga in topic_result.gen_analysis:
        rprint(f"Processing generation {ga.gen_idx+1}...")
        for claim in ga.claims:
            votes: List[bool] = []
            for passage in all_gens:
                usr_prompt = evaluator_prompts["from_generations_user_prompt_strict"].format(
                    passage=passage, claim=claim.content
                )
                messages = [{"role": "user", "content": usr_prompt}]
                result = chat_completion(provider=provider, model=model, messages=messages,
                                         temperature=0.0, max_tokens=500)
                response = result["response"].lower().strip()
                votes.append("true" in response or "yes" in response)

            claim.supportness_score = float(np.mean(votes))
            claims_evaluated += 1

    rprint(f"\n[green]Supportness evaluation complete: {claims_evaluated} claims[/green]")
    return topic_result


# ---------------------------------------------------------------------------
# Phase 5: Respond (generate answers to questions)
# ---------------------------------------------------------------------------

def phase_respond(topic_result: TopicResult, context: str, args) -> TopicResult:
    """For each question, generate multiple stochastic answers."""
    rprint(f"\n[bold]{'='*60}[/bold]")
    rprint("[bold]PHASE 5: Respond[/bold]")
    rprint(f"[bold]{'='*60}[/bold]\n")

    model = args.model.model
    provider = args.model.provider
    max_tokens = args.data.max_completion_tokens
    num_ans = args.data.num_ans_per_question

    total_answers = 0
    for ga in topic_result.gen_analysis:
        rprint(f"Processing generation {ga.gen_idx+1}...")
        for claim in ga.claims:
            for ca in claim.claim_analysis:
                prompt = responder_prompts["respond"].format(
                    context=context, question=ca.question
                )
                messages = [{"role": "user", "content": prompt}]

                answers: List[Dict[str, Any]] = []
                for _ in range(num_ans):
                    result = chat_completion(
                        provider=provider, model=model, messages=messages,
                        temperature=1.0, max_tokens=max_tokens, top_p=1.0,
                        logprobs=False,
                    )
                    answers.append({"text": result["response"], "contradiction": None})
                    total_answers += 1
                ca.answers = answers

    rprint(f"\n[green]Respond phase complete: {total_answers} answers[/green]")
    return topic_result


# ---------------------------------------------------------------------------
# Phase 6: Faithfulness Evaluation (contradiction measurement)
# ---------------------------------------------------------------------------

def phase_evaluate_faithfulness(topic_result: TopicResult, args) -> TopicResult:
    """Measure contradiction between each answer and the accumulated claim context."""
    rprint(f"\n[bold]{'='*60}[/bold]")
    rprint("[bold]PHASE 6: Faithfulness Evaluation[/bold]")
    rprint(f"[bold]{'='*60}[/bold]\n")

    model = args.model.model
    provider = args.model.provider
    max_tokens = args.data.max_completion_tokens

    total_evaluated = 0
    for ga in topic_result.gen_analysis:
        all_claims = ga.all_claims
        rprint(f"Processing generation {ga.gen_idx+1}...")

        for claim_idx, claim in enumerate(ga.claims):
            context = "\n".join(all_claims[claim_idx::-1])

            for ca in claim.claim_analysis:
                if ca.answers is None:
                    continue
                for answer in ca.answers:
                    prompt = responder_prompts["contradiction"].format(
                        statement=answer["text"], context=context
                    )
                    messages = [{"role": "user", "content": prompt}]
                    result = chat_completion(provider=provider, model=model,
                                             messages=messages, temperature=0.0,
                                             max_tokens=max_tokens)

                    response = result["response"].strip()
                    sentences = [s for s in re.split(r'[.!?]+\s*', response) if s]
                    if sentences:
                        response = sentences[-1]

                    match = re.search(r'(\d+)[.,%]?', response)
                    if match:
                        pct = int(match.group(1))
                        answer["contradiction"] = pct / 100.0 if 0 <= pct <= 100 else 0.0
                    else:
                        rprint(f"[yellow]  Warning: could not parse contradiction: {response}[/yellow]")
                        answer["contradiction"] = 0.0
                    total_evaluated += 1

    rprint(f"\n[green]Faithfulness evaluation complete: {total_evaluated} answers[/green]")
    return topic_result


# ---------------------------------------------------------------------------
# UQ Score Computation
# ---------------------------------------------------------------------------

def compute_uq_scores(topic_result: TopicResult) -> List[Dict[str, Any]]:
    """Compute IUQ = supportness * impact for every claim."""
    rprint(f"\n[bold]{'='*60}[/bold]")
    rprint("[bold]IUQ Score Summary[/bold]")
    rprint(f"[bold]{'='*60}[/bold]\n")

    all_scores: List[Dict[str, Any]] = []

    for ga in topic_result.gen_analysis:
        impacts = ga.gather_impacts()
        supportness = ga.gather_supportness_score()
        faithfulness = ga.gather_claim_level_faithfulness()

        for claim_idx, claim in enumerate(ga.claims):
            sup = supportness[claim_idx]
            faith = faithfulness[claim_idx]
            imp = float(impacts[claim_idx])
            iuq = sup * imp if sup is not None else None

            score = {
                "gen_idx": ga.gen_idx,
                "claim": claim.content,
                "supportness_score": sup,
                "faithfulness_score": faith,
                "impact": imp,
                "Interrogative Uncertainty": float(iuq) if iuq is not None else None,
            }
            all_scores.append(score)

            # claim_short = claim.content[:80] + ("..." if len(claim.content) > 80 else "")
            # if iuq is not None:
            #     rprint(f"  Gen {ga.gen_idx} | Claim faithfulness={faith:.3f} | Claim uncertainty={iuq:.3f} | {claim_short}")
            # else:
            #     rprint(f"  Gen {ga.gen_idx} | Claim faithfulness=N/A   | Claim uncertainty=N/A   | {claim_short}")

    return all_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cli_args):
    config = OmegaConf.load("config.yaml")

    if cli_args.model:
        config.model.model = cli_args.model
    if cli_args.num_gen_samples is not None:
        config.data.num_gen_samples = cli_args.num_gen_samples
    if cli_args.num_question_per_claim is not None:
        config.data.num_question_per_claim = cli_args.num_question_per_claim
    if cli_args.num_ans_per_question is not None:
        config.data.num_ans_per_question = cli_args.num_ans_per_question
    if cli_args.max_completion_tokens is not None:
        config.data.max_completion_tokens = cli_args.max_completion_tokens

    model_name = config.model.model
    if model_name in TOGETHER_MODELS:
        config.model.provider = "togetherai"
    elif "gpt" in model_name:
        config.model.provider = "openai"
    else:
        raise ValueError(f"Unknown model: {model_name}. Add it to TOGETHER_MODELS or use a gpt-* model.")

    prompt = cli_args.prompt
    topic = cli_args.topic or prompt[:50]
    result_dir = cli_args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    args = config
    safe_name = re.sub(r'[^\w\-]', '_', topic[:30])
    cache = PipelineCache(result_dir, safe_name, config)

    rprint(f"\n[bold]IUQ Pipeline — Regular API Requests[/bold]")
    rprint(f"  Model:    {args.model.model}")
    rprint(f"  Provider: {args.model.provider}")
    rprint(f"  Topic:    {topic}")
    rprint(f"  Prompt:   {prompt}")
    rprint(f"  Samples:  {args.data.num_gen_samples} diverse generations")
    rprint(f"  Q/claim:  {args.data.num_question_per_claim}")
    rprint(f"  Ans/Q:    {args.data.num_ans_per_question}")
    rprint(f"  Output:   {result_dir}")
    rprint()
    rprint("[dim]Note: correctness evaluation is skipped (requires a reference database).[/dim]")

    # --- Phase 1: Generate -------------------------------------------------
    if cache.is_phase_done("generate"):
        rprint("\n[dim]Phase 1 (Generation): already cached — skipping.[/dim]")
        generations = cache.load_generations()
    else:
        generations = phase_generate(topic, prompt, args)
        cache.save_generations(generations)
        cache.mark_phase_done("generate")

    # --- Phase 2: Claim Extraction -----------------------------------------
    if cache.is_phase_done("claim_extraction"):
        rprint("[dim]Phase 2 (Claim Extraction): already cached — skipping.[/dim]")
        topic_result = cache.load_analysis()
    else:
        topic_result = phase_extract_claims(generations, args)
        cache.save_analysis(topic_result)
        cache.mark_phase_done("claim_extraction")

    # --- Phase 3: Question Generation --------------------------------------
    if cache.is_phase_done("question_generation"):
        rprint("[dim]Phase 3 (Question Generation): already cached — skipping.[/dim]")
        topic_result = cache.load_analysis()
    else:
        topic_result = phase_generate_questions(topic_result, generations["generation_prompt"], args)
        cache.save_analysis(topic_result)
        cache.mark_phase_done("question_generation")

    # --- Phase 4: Supportness Evaluation -----------------------------------
    if cache.is_phase_done("supportness"):
        rprint("[dim]Phase 4 (Supportness): already cached — skipping.[/dim]")
        topic_result = cache.load_analysis()
    else:
        topic_result = phase_evaluate_supportness(topic_result, generations, args)
        cache.save_analysis(topic_result)
        cache.mark_phase_done("supportness")

    # --- Phase 5: Respond --------------------------------------------------
    if cache.is_phase_done("respond"):
        rprint("[dim]Phase 5 (Respond): already cached — skipping.[/dim]")
        topic_result = cache.load_analysis()
    else:
        topic_result = phase_respond(topic_result, generations["generation_prompt"], args)
        cache.save_analysis(topic_result)
        cache.mark_phase_done("respond")

    # --- Phase 6: Faithfulness Evaluation ----------------------------------
    if cache.is_phase_done("faithfulness"):
        rprint("[dim]Phase 6 (Faithfulness): already cached — skipping.[/dim]")
        topic_result = cache.load_analysis()
    else:
        topic_result = phase_evaluate_faithfulness(topic_result, args)
        cache.save_analysis(topic_result)
        cache.mark_phase_done("faithfulness")

    # --- UQ Scores ---------------------------------------------------------
    uq_scores = compute_uq_scores(topic_result)

    # Save final combined output
    output = {
        "topic": topic,
        "prompt": prompt,
        "config": OmegaConf.to_container(args, resolve=True),
        "generations": generations,
        "analysis": topic_result.model_dump(),
        "uq_scores": uq_scores,
    }

    output_path = os.path.join(result_dir, f"{safe_name}_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    rprint(f"\n[bold green]Pipeline complete! Results saved to: {output_path}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides config.yaml)")
    parser.add_argument("--prompt", type=str,
                        default="Tell me a bio of Bernie Sanders",
                        help="Input prompt to analyze")
    parser.add_argument("--topic", type=str, default=None,
                        help="Topic label (default: first 50 chars of prompt)")
    parser.add_argument("--result-dir", type=str, default="./results_synchronous",
                        help="Directory to save results")
    parser.add_argument("--num-gen-samples", type=int, default=None)
    parser.add_argument("--num-question-per-claim", type=int, default=None)
    parser.add_argument("--num-ans-per-question", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)

    cli_args = parser.parse_args()

    main(cli_args)
