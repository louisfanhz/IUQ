import numpy as np
from typing import List, Dict, Optional, Union, Any, Callable
from enum import Enum
from pydantic import BaseModel, Field


class AtomicClaims(BaseModel):
    atomic_claims: list[str] = Field(description="A list of atomic claims extracted from the text.")


class CorrectnessLabel(str, Enum):
    correct = 'correct'
    incorrect = 'incorrect'
    not_enough_information = 'not_enough_information'


class CorrectnessResult(BaseModel):
    correctness: CorrectnessLabel = Field(description="The correctness of the claim.")


class ClaimAnalysis(BaseModel):
    question: str
    answers: Union[List[Dict[str, Any]], None] = Field(default=None)

    def is_populated(self):
        return self.answers is not None

    def gather_answer_scores(self, metric: str, callback: Callable[List[float], float]):
        return callback([ans[metric] for ans in self.answers])

class Claim(BaseModel):
    content: str
    correctness: Union[str, None] = Field(default=None)
    supportness_score: Union[float, None] = Field(default=None)
    claim_analysis: List[ClaimAnalysis] = Field(default=None)

    def is_populated(self):
        if len(self.claim_analysis) == 0:
            return False
        return all([ca.is_populated() for ca in self.claim_analysis])

    def gather_claim_analysis_scores(self, metric: str, callback: Callable[List[float], float], reduction: str):
        try:
            ans_scores = [ca.gather_answer_scores(metric, callback) for ca in self.claim_analysis]
            if reduction == "mean":
                claim_score = np.mean(ans_scores).item()
            elif reduction == "max":
                claim_score = np.max(ans_scores).item()
            else:
                raise ValueError(f"Invalid reduction method: {reduction}")
        except TypeError as e:
            print(f"Error gathering claim analysis scores for metric {metric}: {e}")
            print(f"claim: {self.content}")
            return None

        return claim_score

class GenerationSample(BaseModel):
    gen_idx: int
    all_claims: List[str]
    all_questions: List[str]
    claims: List[Claim]

    def is_populated(self):
        if len(self.claims) == 0:
            return False
        return all([claim.is_populated() for claim in self.claims])

    def gather_claim_scores(self, ca_reduction: str, ans_callbacks: Dict[str, Callable[List[float], float]]):
        claim_scores = {metric: [] for metric in ans_callbacks.keys()}

        for metric, callback in ans_callbacks.items():
            for claim in self.claims:
                claim_scores[metric].append(claim.gather_claim_analysis_scores(metric, callback, ca_reduction))

        return claim_scores

    def gather_claim_level_faithfulness(self):
        faithfulness = []
        for claim in self.claims:
            if not claim.is_populated():
                print(f"WARNING: claim analysis for [{claim.content}] is empty, assuming no contradiction")
                faithfulness.append(0.0)
                continue
            faithfulness.append(claim.gather_claim_analysis_scores("contradiction", lambda x: x, "mean"))
        return (1 - np.array(faithfulness)).tolist()

    def gather_impacts(self, with_error_propagation: bool = True):
        impacts = []
        for claim in self.claims:
            if not claim.is_populated():
                print(f"WARNING: claim analysis for [{claim.content}] is empty, assuming no contradiction")
                impacts.append(0.0)
                continue
            impacts.append(claim.gather_claim_analysis_scores("contradiction", lambda x: x, "mean"))

        impacts = np.array(impacts)

        if not with_error_propagation:
            return 1 - impacts

        ### exp decay error propagation
        weight_func = np.exp(-np.arange(len(impacts)))
        weights = np.convolve(impacts, weight_func)[:len(impacts)]
        impacts = 1 / np.exp(weights)

        return impacts

    def gather_correctness(self):
        return [claim.correctness for claim in self.claims]

    def gather_supportness_score(self):
        return [claim.supportness_score for claim in self.claims]

    def gather_claim_contents(self):
        return [claim.content for claim in self.claims]

class TopicResult(BaseModel):
    gen_analysis: List[GenerationSample] = Field(default=[])

    def is_populated(self):
        if len(self.gen_analysis) == 0:
            return False
        return all([gen.is_populated() for gen in self.gen_analysis])
