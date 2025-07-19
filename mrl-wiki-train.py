import pandas as pd
import random
import torch

from datasets import Dataset
from datetime import datetime

from sklearn.model_selection import train_test_split
from sentence_transformers.evaluation import BinaryClassificationEvaluator, SequentialEvaluator, \
    InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models, losses)

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = pd.read_csv("data/wiki_train_data.csv")
    df.rename(columns={"context": "corpus", "query": "queries"}, inplace=True)

    # Check how many labels are there in the dataset
    unique_labels = df["doc_ids"].unique().tolist()

    # Map each label into its id representation and vice versa
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    df['_id'] = df["doc_ids"].apply(lambda x: labels_to_ids[x]).tolist()

    df_train, df_test = train_test_split(df, train_size=0.7, random_state=42)

    print("Train: ", df_train.shape)
    print("Test: ", df_test.shape)

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    # df_val.reset_index(drop=True, inplace=True)

    # Define your model
    # model_name = "sentence-transformers/LaBSE"
    model_name = "intfloat/multilingual-e5-large"
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
            "mrl-models/mrl_wiki_" + model_name.replace("/", "-").replace("sentence-transformers", "st3")
            + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    model.max_seq_length = max_seq_length
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = Dataset.from_pandas(df_train[["queries", "corpus", "label"]])
    test_dataset = Dataset.from_pandas(df_test[["queries", "corpus", "label"]])

    # Binary Classification Evaluator
    binary_evaluators = []
    for dim in matryoshka_dims:
        binary_evaluators.append(
            BinaryClassificationEvaluator(
                sentences1=test_dataset["queries"],
                sentences2=test_dataset["corpus"],
                labels=test_dataset["label"],
                # name=f"sts-dev-{dim}",
                truncate_dim=dim,
            )
        )
    bin_evaluator = SequentialEvaluator(binary_evaluators)

    # Information Retrieval Evaluator
    df_test._id = df_test._id.astype(str)
    corpus_wiki = Dataset.from_pandas(df_test)
    queries_wiki = Dataset.from_pandas(df_test)
    relevant_docs_data_wiki = pd.DataFrame(zip(df_test["_id"], df_test["_id"], [1] * df_test.shape[0]),
                                           columns=["query-id", "corpus-id", "score"])

    # Shrink the corpus size heavily to only the relevant documents + 10,000 random documents
    required_corpus_ids = list(map(str, relevant_docs_data_wiki["corpus-id"]))
    required_corpus_ids += random.sample(corpus_wiki["_id"], k=100)
    corpus = corpus_wiki.filter(lambda x: x["_id"] in required_corpus_ids)

    # Convert the datasets to dictionaries
    corpus = dict(zip(corpus["_id"], corpus["corpus"]))  # Our corpus (cid => document)
    queries = dict(zip(queries_wiki["_id"], queries_wiki["queries"]))  # Our queries (qid => question)
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for qid, corpus_ids in zip(relevant_docs_data_wiki["query-id"], relevant_docs_data_wiki["corpus-id"]):
        qid = str(qid)
        corpus_ids = str(corpus_ids)
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(corpus_ids)

    retrieval_evaluators = []
    for dim in matryoshka_dims:
        retrieval_evaluators.append(
            InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="ir",
                truncate_dim=dim,
            )
        )
    dev_evaluator = SequentialEvaluator(retrieval_evaluators, main_score_function=lambda scores: scores[0])

    inner_train_loss = losses.MultipleNegativesRankingLoss(model)
    train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="wiki-chckpt",
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
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        # show_progress_bar=True,
        logging_steps=250,
        report_to="none",
        run_name="matryoshka-wiki",
    )

    # 6. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    print("Binary Evaluation Results: \n", bin_evaluator(model, output_path=model_save_path))
    print("Information Retrieval Evaluation Results: \n", dev_evaluator(model, output_path=model_save_path))
