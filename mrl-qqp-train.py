import pandas as pd
import torch
import random

from datasets import load_dataset, Dataset
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.evaluation import SequentialEvaluator, InformationRetrievalEvaluator, \
    BinaryClassificationEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df_train = pd.read_csv("data/qqp_train.tsv", sep="\t", header=None,
                           names=["text1", "text2", "label_text", "label", "unknown"])
    df_test = pd.read_csv("data/qqp_test.tsv", sep="\t", header=None,
                          names=["text1", "text2", "label_text", "label", "unknown"])

    df_train = df_train[(df_train["label"] == "0") | (df_train["label"] == "1")][["text1", "text2", "label"]]
    df_test = df_test[(df_test["label"] == "0") | (df_test["label"] == "1")][["text1", "text2", "label"]]

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train['label'] = df_train['label'].astype('int')
    df_test['label'] = df_test['label'].astype('int')

    # Convert DataFrames to Dataset
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    # Define your model
    model_name = "sentence-transformers/LaBSE"
    # model_name = "intfloat/multilingual-e5-large"
    # model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    BATCH_SIZE = 16  # The larger you select this, the better the results (usually). But it requires more GPU memory
    max_seq_length = 512
    num_epochs = 10

    if "e5" in model_name:
        matryoshka_dims = [768, 512, 256, 128, 64]
    else:
        matryoshka_dims = [512, 256, 128, 64, 32]

    # Save path of the model
    model_save_path = (
            "mrl-models/mrl_qqp_" + model_name.replace("/", "-").replace("sentence-transformers", "st3")
            + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    evaluators = []
    for dim in matryoshka_dims:
        evaluators.append(
            BinaryClassificationEvaluator(
                sentences1=test_dataset["text1"],
                sentences2=test_dataset["text2"],
                labels=test_dataset["label"],
                # name=f"sts-dev-{dim}",
                truncate_dim=dim,
            )
        )
    dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

    inner_train_loss = losses.MultipleNegativesRankingLoss(model)
    train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="qqp-chckpt",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        # fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        # show_progress_bar=True,
        logging_steps=1000,
        report_to="none",
        run_name="matryoshka-qqp",
    )

    # Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    print("Evaluation Results: \n", dev_evaluator(model, output_path=model_save_path))
