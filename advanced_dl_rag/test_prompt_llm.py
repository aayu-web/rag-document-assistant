
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.prompts import QUESTION_REWRITE_PROMPT, ANSWER_PROMPT


load_dotenv(
    r"C:\Users\jhaaa\OneDrive\Desktop\Rag-project\.env"
)

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


rewrite_chain = QUESTION_REWRITE_PROMPT | llm
answer_chain = ANSWER_PROMPT | llm


def main():
    
    chat_history = (
        "User: What are rule-based AI systems?\n"
        "Assistant: They use predefined rules to make decisions."
    )

    question = "Why are they unreliable?"

    rewritten = rewrite_chain.invoke({
        "chat_history": chat_history,
        "question": question
    })

    print("\nREWRITTEN QUESTION")
    print("=" * 50)
    print(rewritten.content)

    
    context = """
    [Source: dl.pdf, page 10]

    Supervised learning uses labeled training data.
    Unsupervised learning identifies patterns in unlabeled data.
    """

    original_question = (
        "What is the difference between supervised "
        "and unsupervised learning?"
    )

    answer = answer_chain.invoke({
        "context": context,
        "question": original_question
    })

    print("\nGENERATED ANSWER")
    print("=" * 50)
    print(answer.content)


if __name__ == "__main__":
    main()
