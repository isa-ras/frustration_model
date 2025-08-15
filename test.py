#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers[torch]",
#   "datasets",
#   "evaluate",
#   "numpy",
#   "scikit-learn",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu118"
# url = "https://download.pytorch.org/whl/cu118"
# [tool.uv]
# index-strategy = "unsafe-best-match"
# ///

import torch
from transformers import pipeline
from evaluate import load as load_metric
from datasets import load_dataset as load_hf_dataset#, Dataset


accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(pred, true):
    return {
        **accuracy_metric.compute(predictions=pred, references=true),
        "f1_weighted": f1_metric.compute(
            predictions=pred,
            references=true,
            average="weighted",
        )["f1"],
    }


def main(
    model: str = "isa-ras/frustration-model",
):
    assert torch.cuda.is_available(), "requires CUDA"

    ds = load_hf_dataset("isa-ras/frustration_dataset")

    pipe = pipeline("text-classification", model="isa-ras/frustration-model")

    test_pred = list(map(lambda e: pipe.model.config.label2id[e["label"]], pipe([r["text"] for r in ds["test"]])))
    
    print(compute_metrics(ds["test"]["label"], test_pred))
    print(pipe("Привет, мир!"))


if __name__ == "__main__":
    main()
