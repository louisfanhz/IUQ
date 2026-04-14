# IUQ: Interrogative Uncertainty Quantification for Long-Form Large Language Model Generation [ACL 2026]

This repo is the implementation of **IUQ: Interrogative Uncertainty Quantification for Long-Form Large Language Model Generation**.

IUQ quantifies uncertainty in long-form Large-Language-Model generation by interrogating the model with targeted independent questions. The final claim-level uncertainty score is a combination of cross-sample consistency and the model's faithfulness to the generative context.

![IUQ illustration](assets/IUQ_gemini_generated.png)

---

## Setup

```bash
pip install openai together omegaconf pydantic rich tqdm numpy
```

Set your API keys in `credentials.py`:

```python
openai_api_key   = "sk-..."
together_api_key = "tgp_..."
```

Both keys may be needed at once: even when evaluating an open-source model via TogetherAI, `batch_main.py` calls an OpenAI judge model for correctness evaluation.

Configure `config.yaml` before running either pipeline.

---

## Pipeline Overview

Both scripts implement the same conceptual stages:

1. **Generation** — 1 greedy + N stochastic completions of the input prompt
2. **Claim Extraction** — atomic claim list from each stochastic generation (structured output)
3. **Question Generation** — one targeted interrogation question per claim (no surrounding context)
4. **Supportness** — fraction of diverse generations that support each claim
5. **Respond** — M independent sampled answers per interrogation question
6. **Faithfulness** — contradiction score between each answer and the accumulated claim context; exponential-decay propagation yields per-claim impact weights

Final score: `U(cᵢ) = supportness(cᵢ) × impact(cᵢ)`

---

## `main.py` — Synchronous Pipeline

Designed for **a single prompt**. Each API call is blocking and sequential; all six phases run in one process. No dataset is required. Correctness evaluation (which requires the reference knowledge database) is skipped.

Each phase is checkpointed to disk, so an interrupted run resumes from where it left off.

```bash
python main.py \
  --prompt "Tell me about the life of Marie Curie" \
  --model "meta-llama/Llama-3.3-70B-Instruct-Turbo" \
  --result-dir ./results_synchronous
```

**Output** under `--result-dir`:

```
results_synchronous/
├── <topic>_manifest.json       # phase completion flags
├── <topic>_generations.json    # Phase 1 output
├── <topic>_analysis.json       # accumulated TopicResult (updated after each phase)
└── <topic>_results.json        # final output with UQ scores
```

Example manifest after a partial run:

```json
{
  "phases": {
    "generate": "completed",
    "claim_extraction": "completed",
    "question_generation": "completed",
    "supportness": "completed",
    "respond": "completed",
    "faithfulness": "pending"
  }
}
```

---

## `batch_main.py` — Asynchronous Batch Pipeline

Designed for **full dataset evaluation** (FActScore or LongFact). Instead of blocking on each call, it packages all requests for a phase into a JSONL file, submits them to the provider's batch API, and exits. You run `--next` again later to poll for completion and process the results.

This saves ~50% cost via provider batch discounts and keeps no long-running process alive.

The pipeline has 5 phases. The `interrogation` phase submits **three batches simultaneously**: question generation, supportness evaluation, and correctness evaluation (the correctness batch always uses the OpenAI judge regardless of the main model's provider).

**Workflow:**

```bash
python batch_main.py --status          # inspect current phase statuses
python batch_main.py --next            # submit next pending phase, or process a completed batch
# … wait for the batch to finish, then run --next again
python batch_main.py --run-to-completion   # alternatively: poll until all phases are done
```

Each `--next` call either submits a new batch (if the phase is `pending`) or downloads and processes results (if the batch is `completed`). The command is idempotent and safe to re-run.

**Output** under `--result-dir` (organized by provider org / model name):

```
results_batch/
└── meta-llama/
    ├── Llama-3.3-70B-Instruct-Turbo_factscore_pipeline.json      # manifest with batch IDs and token counts
    ├── Llama-3.3-70B-Instruct-Turbo_factscore_generations.*      # shelve cache + .json export
    └── Llama-3.3-70B-Instruct-Turbo_factscore_analysis_results.* # shelve cache + .json export
```

Example manifest after a completed run:

```json
{
  "phases": {
    "generate":              { "status": "completed", "batch_id": ["3cf03d14-..."], "total_tokens": 9155 },
    "claim_extraction":      { "status": "completed", "batch_id": ["db6844a9-..."], "total_tokens": 16427 },
    "interrogation":         { "status": "completed", "batch_id": ["1334c569-...", "848ee62c-...", "batch_69dc3d92..."],
                               "provider": ["togetherai", "togetherai", "openai"] },
    "respond":               { "status": "completed", "batch_id": "01e6db5f-...", "total_tokens": 167330 },
    "faithfulness_evaluation":{ "status": "completed", "batch_id": "65f110b6-...", "total_tokens": 443216 }
  }
}
```

Batches exceeding 10,000 requests are automatically split and submitted as numbered parts.

---

## Token Usage Estimates

Based on a reference run: **5 topics, 5 diverse generations, 1 question/claim, 3 answers/question**, using `Llama-3.3-70B-Instruct-Turbo` on FActScore (~92 claims/topic on average).

| Phase | Total tokens (5 topics) | Per topic | Notes |
|-------|------------------------|-----------|-------|
| Generation | 9,155 | ~1,800 | Small — just the prompt + completions |
| Claim Extraction | 16,427 | ~3,300 | Structured output; scales with generation length |
| Question Generation | 58,125 | ~11,600 | Scales with claim count |
| Supportness | 849,556 | ~170,000 | **Dominant.** Each claim is checked against all N generations → O(claims × N²) |
| Respond | 167,330 | ~33,500 | Scales with claims × answers per question |
| Faithfulness | 443,216 | ~88,600 | Each answer re-read with accumulated claim context |
| **Total** | **~1,544K** | **~309K** | Correctness eval excluded (token count not tracked in manifest) |

The **supportness phase accounts for ~55% of total token cost** because every claim in every generation is checked against all N diverse generations independently. Reducing `num_gen_samples` has a roughly quadratic effect on this phase.

The **faithfulness phase** is the second largest cost driver: it grows with `num_gen_samples × claims_per_gen × num_ans_per_question`, and each request carries an expanding claim-context window.

---

## Comparison

| | `main.py` | `batch_main.py` |
|--|-----------|-----------------|
| Input | Single CLI prompt | Dataset (FActScore / LongFact) |
| Execution | Blocking, sequential | Non-blocking, batch API |
| Cost | Standard pricing | ~50% discount |
| Correctness eval | No | Yes (OpenAI judge + reference retrieval) |
| Phases | 6 (separate QG + supportness) | 5 (`interrogation` bundles QG + supportness + correctness) |
| State storage | JSON files | Shelve DB + JSON exports |

---

## Repository Structure

```
├── main.py                    # Synchronous single-prompt pipeline
├── batch_main.py              # Asynchronous batch pipeline
├── config.yaml                # Shared configuration
├── credentials.py             # API keys
├── schemas.py                 # Pydantic models: Claim, GenerationSample, TopicResult
├── prompts.py                 # LLM prompt templates
├── compute_uncertainty.py     # Standalone UQ score computation
├── plot.py                    # Visualization utilities
├── utils/api.py               # Synchronous chat_completion (OpenAI / TogetherAI)
├── batch_utils/
│   ├── api.py                 # Batch API clients + BatchRequestCollector
│   ├── generation_phase.py
│   ├── interrogation_phase.py
│   ├── respond_phase.py       # RespondPhase + FaithfulnessEvaluationPhase
│   └── utils.py               # CacheFileManager (shelve-backed)
└── dataset/
    ├── parse_factscore.py
    └── parse_longfact.py
```
