from datetime import datetime
import torch

from torch.optim import AdamW
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, SequentialEvaluator
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)

from datasets import load_dataset


if __name__ == "__main__":
    # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # model_name = "sentence-transformers/LaBSE"
    model_name = "intfloat/multilingual-e5-large"
    # model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # The larger you select this, the better the results (usually). But it requires more GPU memory
    BATCH_SIZE = 64
    max_seq_length = 512
    num_epochs = 10

    if "e5" in model_name:
        matryoshka_dims = [768, 512, 256, 128, 64]
    else:
        matryoshka_dims = [512, 256, 128, 64, 32]

    # Save path of the model
    model_save_path = (
            "sts-saved-models/mrl_sts_" + model_name.replace("sentence-transformers", "st3").replace("/", "-")
            + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    train_dataset = load_dataset("emrecan/stsb-mt-turkish", split="train")
    val_dataset = load_dataset("emrecan/stsb-mt-turkish", split="validation")
    test_dataset = load_dataset("emrecan/stsb-mt-turkish", split="test")

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluators = []
    for dim in matryoshka_dims:
        evaluators.append(
            EmbeddingSimilarityEvaluator(
                sentences1=val_dataset["sentence1"],
                sentences2=val_dataset["sentence2"],
                scores=val_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name=f"sts-dev-{dim}",
                truncate_dim=dim,
            )
        )
    dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

    inner_train_loss = losses.CoSENTLoss(model=model)
    train_loss = losses.MatryoshkaLoss(model, loss=inner_train_loss, matryoshka_dims=matryoshka_dims)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="sts-chckpt",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        logging_steps=500,
        report_to="none",
        run_name="matryoshka-sts",  # Will be used in W&B if `wandb` is installed
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    evaluators = []
    for dim in matryoshka_dims:
        evaluators.append(
            EmbeddingSimilarityEvaluator(
                sentences1=test_dataset["sentence1"],
                sentences2=test_dataset["sentence2"],
                scores=test_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name=f"sts-test-{dim}",
                truncate_dim=dim,
            )
        )
    test_evaluator = SequentialEvaluator(evaluators)
    test_results = test_evaluator(model, output_path=model_save_path)
    print("Test Result: ", test_results)
