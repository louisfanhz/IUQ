import os
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from omegaconf import OmegaConf
from rich import print as rprint

from batch_utils import *

class PhaseStatus:
    PENDING = "pending"
    BATCH_SUBMITTED = "batch_submitted"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"

class PipelineState:
    """Single JSON manifest that tracks pipeline progress.
    
    tracks which batch IDs are active and the processed batch results
    """
    
    PHASE_ORDER = [
        "generate",
        "claim_extraction",
        "interrogation",
        "respond",
        "faithfulness_evaluation",
    ]
    
    def __init__(self, model_name: str, dataset: str, config: OmegaConf, cache_dir: str = "./results"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        save_names = model_name.split('/')
        if len(save_names) == 2:
            model_dir = save_names[0]
            safe_model = save_names[1]
        else:
            model_dir = "others"
            safe_model = model_name

        if not os.path.exists(self.cache_dir / model_dir):
            os.makedirs(self.cache_dir / model_dir)
        self.manifest_path = self.cache_dir / model_dir / f"{safe_model}_{dataset}_pipeline.json"
        self.generations_path = self.cache_dir / model_dir / f"{safe_model}_{dataset}_generations"
        self.analysis_path = self.cache_dir / model_dir / f"{safe_model}_{dataset}_analysis_results"
        self.baseline_path = self.cache_dir / model_dir / f"{safe_model}_{dataset}_baseline_results"
        
        if self.manifest_path.exists():
            self._data = json.loads(self.manifest_path.read_text())
        else:
            self._data = self._create_manifest(model_name, dataset, config)
            self.save()
    
    def _create_manifest(self, model_name: str, dataset: str, config: OmegaConf) -> Dict:
        return {
            "created_at": time.time(),
            "model": model_name,
            "dataset": dataset,
            "config": OmegaConf.to_container(config, resolve=True),
            "phases": {phase: {"status": PhaseStatus.PENDING} for phase in self.PHASE_ORDER},
        }
    
    @property
    def config(self) -> Dict:
        return self._data["config"]
    
    @property
    def dataset(self) -> str:
        return self._data["dataset"]
    
    def current_phase(self) -> Optional[str]:
        """Return the first phase that is not completed or skipped."""
        for phase in self.PHASE_ORDER:
            status = self._data["phases"][phase]["status"]
            if status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED):
                return phase
        return None
    
    def phase_info(self, phase: str) -> Dict:
        return self._data["phases"][phase]
    
    def update_phase(self, phase: str, **kwargs):
        self._data["phases"][phase].update(kwargs)
        self.save()
    
    def get_provider_for_batch(self, batch_id: str) -> Optional[str]:
        """Look up which provider a batch was submitted to."""
        for phase in self.PHASE_ORDER:
            info = self._data["phases"][phase]
            if info.get("batch_id") == batch_id:
                return info.get("provider")
        return None
    
    def save(self):
        tmp = self.manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.manifest_path)
    
    def get_generations_cache(self) -> CacheFileManager:
        return CacheFileManager(cache_path=str(self.generations_path))
    
    def get_analysis_cache(self) -> CacheFileManager:
        return CacheFileManager(cache_path=str(self.analysis_path))
    
    # def get_baseline_cache(self) -> CacheFileManager:
    #     return CacheFileManager(cache_path=str(self.baseline_path))

def load_dataset(args):
    """Load the dataset."""
    dataset_name = args.data.dataset
    num_topics = args.data.num_topics
    seed = args.seed

    if dataset_name == "factscore":
        from dataset.parse_factscore import generate_dataset
        dataset = generate_dataset(
            db_path=args.data.factscore_db_path,
            prompt_entities_path=args.data.factscore_prompt_entities_path,
            save_copy=False
        ).shuffle(seed=seed).select(range(num_topics))
    elif dataset_name == "longfact":
        from dataset.parse_longfact import generate_dataset
        dataset = generate_dataset(
            data_path=args.data.longfact_data_path
        ).shuffle(seed=seed).select(range(num_topics))
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")
    
    return dataset

def show_status(state: PipelineState):
    STATUS_MARKERS = {
        PhaseStatus.PENDING: "[ ]",
        PhaseStatus.BATCH_SUBMITTED: "⏳",
        PhaseStatus.COMPLETED: "✅",
        PhaseStatus.SKIPPED: "😶",
        PhaseStatus.ERROR: "❌",
    }

    rprint(f"\n[bold]Pipeline Status[/bold]")
    rprint(f"  Model:   {state._data['model']}")
    rprint(f"  Dataset: {state.dataset}")
    rprint()

    for phase in state.PHASE_ORDER:
        info = state.phase_info(phase)
        status = info["status"]
        marker = STATUS_MARKERS.get(status, "[?]")
        line = f"  {marker} {phase}: {status}"
        if "batch_id" in info:
            line += f"  (batch: {info['batch_id']})"
        rprint(line)

    current = state.current_phase()
    if current:
        rprint(f"\n  Next: [bold]{current}[/bold]")
    else:
        rprint(f"\n  [green]All phases complete![/green]")

def submit_phase(args, phase_name: str, state: PipelineState, dataset=None):
    if phase_name == "generate":
        phase = GenerationPhase(args)
        return [phase.prepare_batch(dataset)], [args.model.provider]
    elif phase_name == "claim_extraction":
        phase = InterrogationPhase(args)
        return [phase.prepare_claim_extraction_batch(state)], [args.model.provider]
    elif phase_name == "interrogation":
        phase = InterrogationPhase(args)

        q_batch_id, q_provider = phase.prepare_questions_batch(state), args.model.provider
        s_batch_id, s_provider = phase.prepare_supportness_batch(state), args.model.provider
        c_batch_id, c_provider = phase.prepare_correctness_batch(state), "openai"
        
        return [q_batch_id, s_batch_id, c_batch_id], [q_provider, s_provider, c_provider]
    elif phase_name == "respond":
        phase = RespondPhase(args)
        return phase.prepare_batch(state), args.model.provider
    elif phase_name == "faithfulness_evaluation":
        phase = FaithfulnessEvaluationPhase(args)
        return phase.prepare_batch(state), args.model.provider
    else:
        raise ValueError(f"Unknown phase: {phase_name}")

def process_phase(args, phase_name: str, batch_id: List[str], state: PipelineState):
    if len(batch_id) == 1:
        batch_id = batch_id[0]

    if phase_name == "generate":
        phase = GenerationPhase(args)
        return [phase.process_results(batch_id, state)]
    elif phase_name == "claim_extraction":
        phase = InterrogationPhase(args)
        return [phase.process_claims_results(batch_id, state)]
    elif phase_name == "interrogation":
        phase = InterrogationPhase(args)

        q_result = phase.process_questions_results(batch_id[0], state)
        s_result = phase.process_supportness_results(batch_id[1], state)
        c_result = phase.process_correctness_results(batch_id[2], state)

        return [q_result, s_result, c_result]
    elif phase_name == "respond":
        phase = RespondPhase(args)
        return phase.process_results(batch_id, state)
    elif phase_name == "faithfulness_evaluation":
        phase = FaithfulnessEvaluationPhase(args)
        return phase.process_results(batch_id, state)
    else:
        raise ValueError(f"Unknown phase: {phase_name}")

def next_step(state: PipelineState, args: OmegaConf):
    """Advance the pipeline by one step.
    
    - If current phase is pending: submit its batch.
    - If current phase has a submitted batch: check status, process if complete.
    """
    phase_name = state.current_phase()
    if phase_name is None:
        rprint("[green]All phases complete![/green]")
        return

    info = state.phase_info(phase_name)

    if info["status"] == PhaseStatus.PENDING:
        dataset = None
        if phase_name == "generate":
            dataset = load_dataset(args)
            rprint(f"Loaded {len(dataset)} topics from {state.dataset} dataset")

        batch_id, provider = submit_phase(args, phase_name, state, dataset)
        state.update_phase(phase_name,
            status=PhaseStatus.BATCH_SUBMITTED,
            batch_id=batch_id,
            provider=provider,
            submitted_at=time.time(),
        )
        rprint(f"\n[bold]Submitted[/bold] {phase_name}  batch_id={batch_id}")
        rprint(f"Run [cyan]python pipeline.py --next[/cyan] to check status and advance.")

    elif info["status"] == PhaseStatus.BATCH_SUBMITTED:
        batch_id = info["batch_id"]
        provider = info["provider"]

        batch_ids = batch_id if isinstance(batch_id, list) else [batch_id]
        providers = provider if isinstance(provider, list) else [provider]

        flat_pairs = []
        for bid, pvd in zip(batch_ids, providers):
            if isinstance(bid, list):
                flat_pairs.extend((b, pvd) for b in bid)
            else:
                flat_pairs.append((bid, pvd))

        status = [check_batch_status(bid, pvd) for bid, pvd in flat_pairs]
        batch_status = [s["status"] for s in status]
        if all(s == BatchStatus.COMPLETED.value for s in batch_status):
            result = process_phase(args, phase_name, batch_ids, state)
            state.update_phase(phase_name,
                status=PhaseStatus.COMPLETED,
                completed_at=time.time(),
                result_summary=result,
            )

            next_phase = state.current_phase()
            if next_phase:
                rprint(f"\n[green]{phase_name} complete.[/green]  Next: [bold]{next_phase}[/bold]")
                rprint(f"Run [cyan]python pipeline.py --next[/cyan] to submit {next_phase}.")
            else:
                rprint(f"\n[green]{phase_name} complete. All phases done![/green]")
                batch_cache_dir = Path("./batch_cache")
                if batch_cache_dir.exists():
                    shutil.rmtree(batch_cache_dir)
                    rprint("[yellow]Cleaned up batch_cache directory.[/yellow]")
        else:
            rprint(f"\nBatch [bold]{batch_ids}[/bold] status: {batch_status}")
            rprint(f"Run [cyan]python pipeline.py --next[/cyan] again later.")

def main(args):
    state = PipelineState(
        model_name=args.model.model,
        dataset=args.data.dataset,
        config=args,
        cache_dir=args.result_dir,
    )

    if args.status:
        show_status(state)

    elif args.next:
        next_step(state, args)

if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")

    parser = argparse.ArgumentParser(description="IUQ Pipeline with Batch Inference")
    
    # Pipeline commands (at most one per run; enforced by argparse)
    cmd_group = parser.add_argument_group('Pipeline Commands')
    cmd_group.add_argument("--result-dir", type=str, default="./results_batch",
        help="Directory to save the results")
    ex_grp = cmd_group.add_mutually_exclusive_group()
    ex_grp.add_argument("--next", action="store_true",
        help="Advance the pipeline by one step (submit batch or process completed results)")
    ex_grp.add_argument("--status", action="store_true",
        help="Show current pipeline status")
    ex_grp.add_argument("--check-status", type=str, metavar="BATCH_ID",
        help="Check status of a specific batch job by ID")
    
    args, _ = parser.parse_known_args()

    args = OmegaConf.merge(config, vars(args))

    if args.model.model in [
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Qwen/Qwen3-235B-A22B-fp8-tput",
        "Qwen/Qwen3.5-397B-A17B",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "google/gemma-2-27b-it",
        "google/gemma-3n-E4B-it",
        "google/gemma-4-31B-it",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "MiniMaxAI/MiniMax-M2.7",
        "moonshotai/Kimi-K2.5",
    ]:
        args.model.provider = "togetherai"
    elif "gpt" in args.model.model:
        args.model.provider = "openai"
    else:
        raise ValueError(f"Unknown model: {args.model.model}")

    main(args)