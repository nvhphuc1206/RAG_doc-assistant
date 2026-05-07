"""Evaluation script using RAGAS framework."""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


def evaluate_rag(questions, ground_truths, vector_store):
    """Run RAGAS evaluation on the RAG pipeline.

    Args:
        questions: List of test questions.
        ground_truths: List of expected answers.
        vector_store: ChromaDB vector store.

    Returns:
        RAGAS evaluation results.
    """
    from src.generation.rag_chain import ask

    answers = []
    contexts = []

    for q in questions:
        result = ask(vector_store, q)
        answers.append(result["answer"])
        contexts.append([s["content_preview"] for s in result["sources"]])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return results
