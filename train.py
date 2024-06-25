import uuid
from functools import partial

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def is_subword(text, tokenized, tokenizer, index):
    word = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][index])
    start_ind, end_ind = tokenized["offset_mapping"][index]
    word_ref = text[start_ind:end_ind]
    is_subword = len(word) != len(word_ref)
    return is_subword


def tokenize(example, labels2int, tokenizer, iob=True, ignore_subwords=True):

    text, labels = example["source_text"], example["privacy_mask"]

    i = 0
    token_labels = []

    tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    start_token_to_label = {
        tokenized.char_to_token(label["start"]): (label["start"], label["end"], label["label"]) for label in labels
    }
    while i < len(tokenized["input_ids"]):
        if tokenized["special_tokens_mask"][i] == 1:
            token_labels.append(-100)
            i += 1
        elif i not in start_token_to_label:
            if ignore_subwords and is_subword(text, tokenized, tokenizer, i):
                token_labels.append(-100)
            else:
                token_labels.append(labels2int["O"])
            i += 1
        else:
            start, end, label = start_token_to_label[i]
            start_token = tokenized.char_to_token(start)
            assert start_token == i
            j = start_token
            while j < (len(tokenized["input_ids"]) - 1) and tokenized.token_to_chars(j).start < end:
                if j == start_token:
                    if iob:
                        token_labels.append(labels2int["B-" + label])
                    else:
                        token_labels.append(labels2int[label])
                elif ignore_subwords and is_subword(text, tokenized, tokenizer, j):
                    token_labels.append(-100)
                else:
                    if iob:
                        token_labels.append(labels2int["I-" + label])
                    else:
                        token_labels.append(labels2int[label])

                j += 1
            i = j
    tokenized["labels"] = token_labels
    return tokenized


def compute_metrics(eval_pred, label_list, seqeval_metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    results_flat = {f"{k}_f1": v["f1"] for k, v in results.items() if isinstance(v, dict)}
    results_flat.update(
        {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    )
    return results_flat


def train():
    labels = [
        "BOD",
        "BUILDING",
        "CARDISSUER",
        "CITY",
        "COUNTRY",
        "DATE",
        "DRIVERLICENSE",
        "EMAIL",
        "GEOCOORD",
        "GIVENNAME1",
        "GIVENNAME2",
        "IDCARD",
        "IP",
        "LASTNAME1",
        "LASTNAME2",
        "LASTNAME3",
        "PASS",
        "PASSPORT",
        "POSTCODE",
        "SECADDRESS",
        "SEX",
        "SOCIALNUMBER",
        "STATE",
        "STREET",
        "TEL",
        "TIME",
        "TITLE",
        "USERNAME",
    ]

    labels = [f"I-{label}" for label in labels] + [f"B-{label}" for label in labels] + ["O"]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {v: k for k, v in label2id.items()}
    pretrained_name = "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModelForTokenClassification.from_pretrained(pretrained_name, num_labels=len(labels), id2label=id2label)

    ds = load_dataset("ai4privacy/pii-masking-300k")
    ds = ds.filter(lambda x: x["language"] == "English", num_proc=4)
    ds = ds.map(
        partial(tokenize, labels2int=label2id, tokenizer=tokenizer, iob=True, ignore_subwords=True),
        batched=False,
        remove_columns=[
            "source_text",
            "target_text",
            "privacy_mask",
            "span_labels",
            "mbert_text_tokens",
            "mbert_bio_labels",
            "id",
            "language",
            "set",
        ],
        num_proc=8,
    ).remove_columns(["offset_mapping"])
    training_arguments = TrainingArguments(
        output_dir="output",
        max_steps=30000,
        eval_steps=1000,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        overwrite_output_dir=True,
        warmup_ratio=0.2,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=1,
        save_steps=1000,
        lr_scheduler_type="cosine",
        report_to="wandb",
        push_to_hub=True,
        warmup_steps=3000,
        metric_for_best_model="f1",
        greater_is_better=True,
        hub_model_id=f"{pretrained_name}-pii-en",
        hub_strategy="every_save",
        hub_private_repo=True,
        run_name=str(uuid.uuid4()),
        torch_compile=False,
    )
    trainer = Trainer(
        model,
        training_arguments,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, label_list=labels, seqeval_metric=evaluate.load("seqeval")),
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()
