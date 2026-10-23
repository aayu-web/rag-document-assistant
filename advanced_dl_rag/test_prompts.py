
from src.prompts import QUESTION_REWRITE_PROMPT, ANSWER_PROMPT


rewrite_messages = QUESTION_REWRITE_PROMPT.format_messages(
    chat_history=(
        "User: What are rule-based AI systems?\n"
        "Assistant: They use predefined rules to make decisions."
    ),
    question="Why are they unreliable?"
)

print("QUESTION PROMPT")
print("=" * 50)

for message in rewrite_messages:
    print(f"\n{message.type.upper()}:\n{message.content}")



answer_messages = ANSWER_PROMPT.format_messages(
    context=(
        "[Source: dl.pdf, page 10]\n"
        "Supervised learning uses labeled training data. "
        "Unsupervised learning identifies patterns in unlabeled data."
    ),
    question=(
        "What is the difference between supervised "
        "and unsupervised learning?"
    )
)

print("\n\nANSWER PROMPT")
print("=" * 50)

for message in answer_messages:
    print(f"\n{message.type.upper()}:\n{message.content}")
