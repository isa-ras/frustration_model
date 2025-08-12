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

from os import environ
from pathlib import Path

import torch
import numpy as np
from evaluate import load as load_metric
from datasets import load_dataset as load_hf_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


def main(
    model: str = "ai-forever/ruBert-large",
    params: dict[str, int|str] = {  # 0.865063564159058
        "learning_rate": 1.0949039973836132e-05,
        "weight_decay": 8.964314813726457e-08,
        "adam_beta1": 0.8816591494898484,
        "adam_beta2": 0.985945634085646,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "warmup_ratio": 0.047598867356552096,
        "lr_scheduler_type": "linear",
    },
):
    assert torch.cuda.is_available(), "requires CUDA"

    ds = load_hf_dataset("isa-ras/frustration_dataset")

    tokenizer = AutoTokenizer.from_pretrained(
        model,
    )
    tokenizer.add_tokens(["<Person>"])

    ds["train"] = ds["train"].map(
        lambda e: tokenizer(e["text"], truncation=True, max_length=512),
        batched=True,
        num_proc=8,
    )
    ds["train"] = ds["train"].remove_columns(["text", "source"])

    ds["test"] = ds["test"].map(
        lambda e: tokenizer(e["text"], truncation=True, max_length=512),
        batched=True,
        num_proc=8,
    )

    ds["validation"] = ds["validation"].map(
        lambda e: tokenizer(e["text"], truncation=True, max_length=512),
        batched=True,
        num_proc=8,
    )


    collator = DataCollatorWithPadding(
        tokenizer,
        padding="longest",
        max_length=512,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels=ds["train"].features["label"].num_classes,
        id2label=ds["train"].features["label"]._int2str,
        label2id=ds["train"].features["label"]._str2int,
    )
    model.resize_token_embeddings(len(tokenizer))

    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            **accuracy_metric.compute(predictions=predictions, references=labels),
            **f1_metric.compute(
                predictions=predictions,
                references=labels,
                average="weighted",
            ),
        }

    training_args = TrainingArguments(
        # output_dir=None,
        output_dir="frustration_model",
        # overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="no",
        per_device_eval_batch_size=4,
        #
        hub_token=environ["HF_HUB_TOKEN"],
        push_to_hub=True,
        hub_model_id="isa-ras/frustration-model",
        hub_strategy="end",
        #
        # **({"per_device_train_batch_size": 4} | (params or {})),
        **params,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=collator,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )

    trainer.train()
    print(trainer.evaluate(ds["test"], metric_key_prefix="test"))
    trainer.save_model()

if __name__ == "__main__":
    main(
        # params={  # 0.7130195375323874
        #     "learning_rate": 8.992036617560402e-06,
        #     "weight_decay": 3.929422872559948e-05,
        #     "adam_beta1": 0.8544801533652253,
        #     "adam_beta2": 0.9400285139652853,
        #     "num_train_epochs": 3,
        #     "per_device_train_batch_size": 2,
        #     "warmup_ratio": 0.22488060818846878,
        #     "lr_scheduler_type": "cosine",
        # },
    )
