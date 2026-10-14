
import csv
from importlib import import_module


rag = import_module("08_reranker")

retrieve_and_rerank = rag.retrieve_and_rerank
get_page_number = rag.get_page_number
prompt = rag.prompt
llm = rag.llm





TEST_CASES = [
    {
        "question": "When did deep learning begin?",
        "facts": [
            ["1940s", "1940"],
            ["cybernetics"],
        ],
        "out_of_scope": False,
    },
    {
        "question": "How are machine learning and deep learning related?",
        "facts": [
            [
                "deep learning is a subset of representation learning",
                "deep learning belongs to representation learning",
                "deep learning is a kind of representation learning",
            ],
            [
                "representation learning is a subset of machine learning",
                "representation learning is a kind of machine learning",
                "representation learning sits inside machine learning",
            ],
        ],
        "out_of_scope": False,
    },
    {
        "question": "What are the three waves in deep learning history?",
        "facts": [
            ["cybernetics"],
            ["connectionism"],
            ["current resurgence", "third wave", "third wave began"],
        ],
        "out_of_scope": False,
    },
    {
        "question": "What is the relationship between AI and machine learning?",
        "facts": [
            [
                "machine learning is one of those ways",
                "machine learning is a sub area of ai",
                "machine learning sits inside the ai circle",
            ],
            [
                "ai includes machine learning along with other approaches",
                "ai includes many different ways",
                "ai includes machine learning",
            ],
        ],
        "out_of_scope": False,
    },
    {
        "question": "What is the capital of Japan?",
        "facts": [],
        "out_of_scope": True,
    },
]





def generate_answer(question, ranked_docs):
    if not ranked_docs:
        return (
            "The textbook passages retrieved do not provide "
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




def normalize(text):
    return " ".join(
        text.lower()
        .replace("-", " ")
        .replace("*", "")
        .split()
    )



def calculate_fact_coverage(answer, expected_facts):
    """
  
    This remains a lexical baseline, not semantic verification.
    """
    if not expected_facts:
        return ""

    normalized_answer = normalize(answer)

    matched = 0

    for alternatives in expected_facts:
        if isinstance(alternatives, str):
            alternatives = [alternatives]

        if any(
            normalize(phrase) in normalized_answer
            for phrase in alternatives
        ):
            matched += 1

    return round(matched / len(expected_facts), 3)



def check_abstention(answer):
    answer = answer.lower()

    phrases = [
        "not enough information",
        "does not contain",
        "doesn't contain",
        "do not contain",
        "don't contain",
        "cannot determine",
        "can't determine",
        "unable to answer",
        "not provided",
        "not mentioned",
        "outside the scope",
        "not covered in the textbook",
        "context does not provide",
        "context doesn't provide",
        "cannot answer from the context",
        "cannot answer that question from the given context",
    ]

    return any(phrase in answer for phrase in phrases)




def evaluate_case(case):
    question = case["question"]

    
    ranked_docs = retrieve_and_rerank(question)

    
    answer = generate_answer(question, ranked_docs)

    pages = []
    scores = []
    excerpts = []

    for score, doc in ranked_docs:
        page = get_page_number(doc)
        source = doc.metadata.get("source", "Unknown source")

        source_label = f"{source} | PDF page {page}"

        if source_label not in pages:
            pages.append(source_label)

        scores.append(float(score))

        excerpt = " ".join(doc.page_content.split())[:200]
        excerpts.append(f"{source_label}: {excerpt}")

  
    fact_coverage = calculate_fact_coverage(
        answer,
        case["facts"],
    )

   
    if case["out_of_scope"]:
        abstained = check_abstention(answer)
        abstention_result = "PASS" if abstained else "REVIEW"
    else:
        abstention_result = "N/A"

    return {
        "question": question,
        "answer": answer,
        "source_pages": " ; ".join(pages),
        "retrieved_chunks": len(ranked_docs),
        "top_reranker_score": (
            round(max(scores), 4) if scores else ""
        ),
        "mean_reranker_score": (
            round(sum(scores) / len(scores), 4)
            if scores else ""
        ),
        "expected_fact_coverage": fact_coverage,
        "out_of_scope": case["out_of_scope"],
        "abstention_result": abstention_result,
        "retrieved_excerpts": " || ".join(excerpts),
    }




def main():
    results = []

    print("=" * 75)
    print("RAG CHATBOT EVALUATION")
    print("=" * 75)

    for index, case in enumerate(TEST_CASES, start=1):
        print("\n" + "=" * 75)
        print(f"TEST {index}/{len(TEST_CASES)}")
        print("QUESTION:", case["question"])
        print("=" * 75)

        try:
            result = evaluate_case(case)
            results.append(result)

            print("\nANSWER:")
            print(result["answer"])

            print("\nSOURCE PAGES:")
            print(result["source_pages"] or "No sources retrieved")

            print("\nRETRIEVED CHUNKS:")
            print(result["retrieved_chunks"])

            print("\nTOP RERANKER SCORE:")
            print(result["top_reranker_score"])

            print("\nEXPECTED-FACT COVERAGE:")
            print(result["expected_fact_coverage"])

            print("\nOUT-OF-SCOPE CHECK:")
            print(result["abstention_result"])

        except Exception as error:
            print(f"\nERROR: {error}")

            results.append({
                "question": case["question"],
                "answer": f"ERROR: {error}",
                "source_pages": "",
                "retrieved_chunks": 0,
                "top_reranker_score": "",
                "mean_reranker_score": "",
                "expected_fact_coverage": "",
                "out_of_scope": case["out_of_scope"],
                "abstention_result": "ERROR",
                "retrieved_excerpts": "",
            })

    
    if results:
        output_file = "rag_evaluation_results.csv"

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

        print("\n" + "=" * 75)
        print("EVALUATION COMPLETE")
        print("=" * 75)
        print("Total test cases:", len(TEST_CASES))
        print("Results saved to:", output_file)

        successful = [
            result for result in results
            if not result["answer"].startswith("ERROR:")
        ]

        print("Successful evaluations:", len(successful))

        in_scope = [
            result for result in successful
            if not result["out_of_scope"]
            and result["expected_fact_coverage"] != ""
        ]

        if in_scope:
            average_coverage = sum(
                result["expected_fact_coverage"]
                for result in in_scope
            ) / len(in_scope)

            print(
                "Mean lexical fact-coverage baseline:",
                round(average_coverage, 3),
            )

        print(
            "\nNote: Reranker scores are not probabilities. "
            "Lexical fact coverage is only a baseline and "
            "does not measure semantic correctness or faithfulness."
        )


if __name__ == "__main__":
    main()
