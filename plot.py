import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt

from rich import print as rprint

from schemas import ClaimAnalysis, Claim, TopicResult

class Plotting:
    def __init__(self, save_name: str, results: Dict[str, Any]):
        self.results = results
        self._auroc_results = {}

        metric_names = ["IUQ", "claim_supportness_score"]

        metric_scores = {name: [] for name in metric_names}
        metric_scores["IUQ"] = []
        metric_scores["claim_supportness_score"] = []
        
        labels = []
        claim_contents = []
        for topic in self.results.keys():
            topic_res = TopicResult(**self.results[topic])
            for ga in topic_res.gen_analysis:
                supportness_scores = ga.gather_supportness_score()
                metric_scores["claim_supportness_score"].extend(supportness_scores)

                labels.extend(ga.gather_correctness())

                impacts = ga.gather_impacts()
                metric_scores["IUQ"].extend(supportness_scores * impacts)
                claim_contents.extend(ga.gather_claim_contents())

        ### exclude correctness = not enough information ###
        labels = np.array(labels)
        metric_scores = {name: np.array(scores) for name, scores in metric_scores.items()}
        correct_labels = labels == "correct"
        invalid_labels = labels == "irrelevant"
        # correct_labels = correct_labels[~invalid_labels]
        # for name in metric_names:
        #     metric_scores[name] = metric_scores[name][~invalid_labels]

        self.plot_auroc(save_name, metric_scores, correct_labels, metric_names)

    @property
    def auroc_results(self):
        return self._auroc_results

    def plot_auroc(self, save_name: str, metric_scores: Dict[str, np.ndarray], labels: np.ndarray, metric_names: List[str]):
        plt.figure(figsize=(10, 10))
        for metric_name in metric_names:
            auroc = metrics.roc_auc_score(labels, metric_scores[metric_name])
            auprc = metrics.average_precision_score(labels, metric_scores[metric_name])
            auprc_n = metrics.average_precision_score(1 - labels, -metric_scores[metric_name])

            fpr, tpr, _ = metrics.roc_curve(labels, metric_scores[metric_name])
            plt.plot(fpr, tpr, label=f"{metric_name}, auc={round(auroc, 3)}, auprc={round(auprc, 3)}, auprc_n={round(auprc_n, 3)}")

            # print(f"{metric_name}: auroc={auroc}, auprc={auprc}, auprc_n={auprc_n}")
            print(f"{metric_name}: auroc={auroc}, auprc={auprc}, auprc_n={auprc_n}")

            self._auroc_results[metric_name] = {
                "auroc": auroc,
                "auprc": auprc,
                "auprc_n": auprc_n
            }

        plt.legend()
        plt.savefig(f"./results_batch/{save_name}_roc_plot.png")

def plot_claim_level_auroc():
    result_files = sorted(Path(".").glob("results_batch/**/*_analysis_results.json"))
    save_names = [f.name.replace("_analysis_results.json", "") for f in result_files]
    analysis_results_paths = [str(f) for f in result_files]

    auc_results = {}

    for ar in analysis_results_paths:
        with open(ar, "r") as f:
            results = json.load(f)
        
        rprint(f"Eval result of {ar.split('/')[-1]}:\nnumber of topics loaded: {len(results.keys())}")
        name = save_names[analysis_results_paths.index(ar)]
        plotting = Plotting(name, results)
        auc_results[name] = plotting.auroc_results


if __name__ == "__main__":
    plot_claim_level_auroc()