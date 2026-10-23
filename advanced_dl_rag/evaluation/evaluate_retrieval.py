import sys
from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_retri import (
    load_parents,
    load_retrieval_components,
    retrieve_candidate_parents,
    rerank_parents,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QRELS_PATH = PROJECT_ROOT / "evaluation" / "qrels.json"

K_VALUES = [1, 3, 5]


def precision_at_k(ranked_ids, relevant, k):
    top_k = ranked_ids[:k]

    if not top_k:
        return 0.0

    hits = sum(parent_id in relevant for parent_id in top_k)
    return hits / k


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0

    hits = sum(
        parent_id in relevant
        for parent_id in ranked_ids[:k]
    )

    return hits / len(relevant)


def reciprocal_rank(ranked_ids, relevant):
    for rank, parent_id in enumerate(ranked_ids, start=1):
        if parent_id in relevant:
            return 1.0 / rank

    return 0.0


def dcg_at_k(ranked_ids, relevance_grades, k):
    score = 0.0

    for rank, parent_id in enumerate(ranked_ids[:k], start=1):
        grade = relevance_grades.get(parent_id, 0)

       
        score += (2**grade - 1) / __import__("math").log2(rank + 1)

    return score


def ndcg_at_k(ranked_ids, relevance_grades, k):
    actual_dcg = dcg_at_k(ranked_ids, relevance_grades, k)

    ideal_ids = sorted(
        relevance_grades,
        key=lambda parent_id: relevance_grades[parent_id],
        reverse=True,
    )

    ideal_dcg = dcg_at_k(ideal_ids, relevance_grades, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def main():
    with QRELS_PATH.open("r", encoding="utf-8") as file:
        qrels = json.load(file)

    parents = load_parents()

    vector_store, bm25_retriever, reranker = (
        load_retrieval_components()
    )

    all_results = []

    for qid, item in qrels.items():
        query = item["query"]

        relevance_grades = item["relevant_parents"]

       
        relevant = {
            parent_id
            for parent_id, grade in relevance_grades.items()
            if grade > 0
        }

        missing_ids = relevant - parents.keys()

        if missing_ids:
            print(
                f"\nWarning: {len(missing_ids)} qrels IDs "
                f"are not present in the parent index for {qid}."
            )

        if not relevant:
            print(f"Skipping {qid}: no relevant passages labeled.")
            continue

        print(f"\nEvaluating {qid}: {query}")

        candidates = retrieve_candidate_parents(
            query,
            vector_store,
            bm25_retriever,
            parents,
        )

        ranked_parents = rerank_parents(
            query,
            candidates,
            reranker,
        )

        ranked_ids = [
            parent["parent_id"]
            for parent in ranked_parents
        ]



        result = {"query_id": qid}

        for k in K_VALUES:
            result[f"Precision@{k}"] = precision_at_k(
                ranked_ids, relevant, k
            )
            result[f"Recall@{k}"] = recall_at_k(
                ranked_ids, relevant, k
            )
            result[f"nDCG@{k}"] = ndcg_at_k(
                ranked_ids, relevance_grades, k
            )

        result["MRR"] = reciprocal_rank(ranked_ids, relevant)

        all_results.append(result)

        print(f"Retrieved parent IDs: {ranked_ids}")

        for metric, value in result.items():
            if metric != "query_id":
                print(f"{metric}: {value:.4f}")

    if not all_results:
        print("\nNo queries were evaluated.")
        return

    metric_names = [
        key for key in all_results[0]
        if key != "query_id"
    ]

    print("\n" + "=" * 55)
    print("MEAN EVALUATION RESULTS")
    print("=" * 55)

    for metric in metric_names:
        mean_value = sum(
            result[metric] for result in all_results
        ) / len(all_results)

        print(f"{metric}: {mean_value:.4f}")

    output_path = PROJECT_ROOT / "evaluation" / "results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
