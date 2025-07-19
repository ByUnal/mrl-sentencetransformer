from datetime import datetime
from datasets import load_dataset

import torch
from torch.optim import AdamW
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, SequentialEvaluator, TripletEvaluator
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    models
)
from sentence_transformers.training_args import BatchSamplers


if __name__ == "__main__":
    # Define your model
    # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_name = "sentence-transformers/LaBSE"
    # model_name = "intfloat/multilingual-e5-large"
    # model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    BATCH_SIZE = 8  # The larger you select this, the better the results (usually). But it requires more GPU memory
    max_seq_length = 512
    num_epochs = 10

    if "e5" in model_name:
        matryoshka_dims = [768, 512, 256, 128, 64]
    else:
        matryoshka_dims = [512, 256, 128, 64, 32]

    # Save path of the model
    model_save_path = (
            "mrl-models/mrl_nli_" + model_name.replace("sentence-transformers", "st3").replace("/", "-")
            + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    model.max_seq_length = max_seq_length
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = load_dataset("emrecan/all-nli-tr", 'triplet', split="train")
    val_dataset = load_dataset("emrecan/all-nli-tr", 'triplet', split="dev")
    test_dataset = load_dataset("emrecan/all-nli-tr", 'triplet', split="test")

    print("Train: ", train_dataset)
    print("Test: ", test_dataset)
    print("Val: ", val_dataset)

    # Evaluation
    evaluators = []
    for dim in matryoshka_dims:
        evaluators.append(
            TripletEvaluator(
                anchors=val_dataset["anchor"],
                positives=val_dataset["positive"],
                negatives=val_dataset["negative"],
                truncate_dim=dim,
            )
        )
    dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

    inner_train_loss = losses.MultipleNegativesRankingLoss(model)
    train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="nli-chckpt",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        logging_steps=2000,
        report_to="none",
        run_name="matryoshka-nli",  # Will be used in W&B if `wandb` is installed
    )

    # Generate trainer object and initialize the training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # Save model
    model.save(model_save_path)

    # # Load model if necessary
    # model = SentenceTransformer(model_save_path)
    # model.max_seq_length = max_seq_length

    triplet_evaluators = []
    for dim in matryoshka_dims:
        evaluators.append(
            TripletEvaluator(
                anchors=test_dataset["anchor"],
                positives=test_dataset["positive"],
                negatives=test_dataset["negative"],
                name=f"sts-triplet-test-{dim}",
                truncate_dim=dim,
            )
        )
    triplet_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])
    print("Triplet Evaluation Results: \n", triplet_evaluator(model, output_path=model_save_path))

    embedding_test_dataset = load_dataset("emrecan/all-nli-tr", 'pair-score', split="test")
    evaluators = []
    for dim in matryoshka_dims:
        evaluators.append(
            EmbeddingSimilarityEvaluator(
                sentences1=embedding_test_dataset["sentence1"],
                sentences2=embedding_test_dataset["sentence2"],
                scores=embedding_test_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name=f"sts--embedding-test-{dim}",
                truncate_dim=dim,
            )
        )
    similarity_evaluator = SequentialEvaluator(evaluators)
    print("Embedding Similarity Evaluation Results: \n", similarity_evaluator(model, output_path=model_save_path))

