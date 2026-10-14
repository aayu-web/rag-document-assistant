
import asyncio
import csv
import os
from importlib import import_module

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)



load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )


rag = import_module("08_reranker")

retrieve_and_rerank = rag.retrieve_and_rerank
get_page_number = rag.get_page_number
prompt = rag.prompt
llm = rag.llm



from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

# Groq exposes an OpenAI-compatible API endpoint.
judge_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

judge_llm = llm_factory(
    model="openai/gpt-oss-120b",
    provider="openai",
    client=judge_client,
    max_tokens=4096,
)


judge_embeddings = RagasHuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

faithfulness_metric = Faithfulness(llm=judge_llm)

answer_relevancy_metric = AnswerRelevancy(
    llm=judge_llm,
    embeddings=judge_embeddings,
)

context_precision_metric = ContextPrecision(llm=judge_llm)
context_recall_metric = ContextRecall(llm=judge_llm)

print("RAGAS evaluation judge initialized successfully.")




TEST_CASES = [
    {
        "question": "When did deep learning begin?",
        "reference": (
            "Early neural-network research associated with "
            "cybernetics dates to the 1940s-1960s."
        ),
        "out_of_scope": False,
    },
    {
        "question": (
            "How are machine learning and deep learning related?"
        ),
        "reference": (
            "Deep learning is a type of representation learning, "
            "which is a subfield of machine learning."
        ),
        "out_of_scope": False,
    },
    {
        "question": (
            "What are the three waves in deep learning history?"
        ),
        "reference": (
            "The three waves are cybernetics in the 1940s-1960s, "
            "connectionism in the 1980s-1990s, and the more recent "
            "resurgence of deep learning."
        ),
        "out_of_scope": False,
    },
    {
        "question": (
            "What is the relationship between AI and machine learning?"
        ),
        "reference": (
            "Machine learning is a subfield of AI. AI also includes "
            "approaches that do not rely on machine learning."
        ),
        "out_of_scope": False,
    },
    {
        "question": "What is the capital of Japan?",
        "reference": "",
        "out_of_scope": True,
    },
]

def generate_answer(question, ranked_docs):
    if not ranked_docs:
        return (
            "The retrieved textbook passages do not provide "
            "enough information to answer this question."
        )

    context_parts = []

    for rank, (score, doc) in enumerate(ranked_docs, start=1):
        page = get_page_number(doc)

        context_parts.append(
            f"[Source {rank} | PDF page {page}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    messages = prompt.invoke({
        "question": question,
        "context": context,
    })

    response = llm.invoke(messages)

    return str(response.content).strip()




async def evaluate_case(case):
    question = case["question"]
    reference = case["reference"]
    out_of_scope = case["out_of_scope"]

    print("\n" + "=" * 75)
    print("QUESTION:", question)
    print("=" * 75)

    
    ranked_docs = retrieve_and_rerank(question)

   
    answer = generate_answer(question, ranked_docs)

    
    contexts = [
        doc.page_content
        for score, doc in ranked_docs
    ]

    
    source_pages = list(dict.fromkeys(
        get_page_number(doc)
        for score, doc in ranked_docs
    ))

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED CHUNKS:", len(contexts))
    print("SOURCE PAGES:", ", ".join(source_pages) or "None")

    result = {
        "question": question,
        "answer": answer,
        "source_pages": ", ".join(source_pages),
        "retrieved_chunks": len(contexts),
        "out_of_scope": out_of_scope,
        "faithfulness": "",
        "answer_relevancy": "",
        "context_precision": "",
        "context_recall": "",
    }

  
    metrics_to_run = [
        (
            "faithfulness",
            faithfulness_metric,
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            },
        ),
        (
            "answer_relevancy",
            answer_relevancy_metric,
            {
                "user_input": question,
                "response": answer,
            },
        ),
    ]

   
    if not out_of_scope and reference.strip():
        metrics_to_run.extend([
            (
                "context_precision",
                context_precision_metric,
                {
                    "user_input": question,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                },
            ),
            (
                "context_recall",
                context_recall_metric,
                {
                    "user_input": question,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                },
            ),
        ])

    for metric_name, metric, kwargs in metrics_to_run:
        try:
            metric_result = await metric.ascore(**kwargs)
            score = float(metric_result.value)

            result[metric_name] = round(score, 4)

            print(f"\n{metric_name}: {score:.4f}")

            reason = getattr(metric_result, "reason", None)
            if reason:
                print("Explanation:", reason)

        except Exception as error:
            result[metric_name] = f"ERROR: {error}"
            print(f"\n{metric_name} failed: {error}")

    return result




async def main():
    results = []

    print("=" * 75)
    print("RAGAS EVALUATION")
    print("=" * 75)

    for index, case in enumerate(TEST_CASES, start=1):
        print(f"\nTEST {index}/{len(TEST_CASES)}")

        try:
            result = await evaluate_case(case)
            results.append(result)

        except Exception as error:
            print("\nTEST FAILED:", error)

            results.append({
                "question": case["question"],
                "answer": f"ERROR: {error}",
                "source_pages": "",
                "retrieved_chunks": 0,
                "out_of_scope": case["out_of_scope"],
                "faithfulness": "",
                "answer_relevancy": "",
                "context_precision": "",
                "context_recall": "",
            })

    if not results:
        print("No evaluation results were produced.")
        return

    output_file = "ragas_evaluation_results.csv"

    fieldnames = [
        "question",
        "answer",
        "source_pages",
        "retrieved_chunks",
        "out_of_scope",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 75)
    print("RAGAS EVALUATION COMPLETE")
    print("=" * 75)
    print("Total test cases:", len(results))
    print("Saved results:", output_file)

    print("\nMEAN METRIC SCORES")
    print("-" * 40)

    for metric_name in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        scores = [
            row[metric_name]
            for row in results
            if isinstance(row[metric_name], (int, float))
        ]

        if scores:
            mean_score = sum(scores) / len(scores)

            print(
                f"{metric_name}: {mean_score:.4f} "
                f"(evaluated on {len(scores)} cases)"
            )
        else:
            print(f"{metric_name}: No valid scores")

    print(
        "\nNote: These are model-based evaluation estimates. "
        "Review the individual answers and scores before "
        "drawing conclusions."
    )


if __name__ == "__main__":
    asyncio.run(main())
