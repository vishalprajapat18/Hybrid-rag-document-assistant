from typing import List
from app.models import Message

def build_prompt(question: str, results, history: List[Message]):
    context = ""
    for chunk in results:
        context += f"[{chunk['filename']}, page {chunk['page']}]\n{chunk['text']}\n\n"

    conversation = ""
    for msg in history:
        conversation += f"{msg.role}: {msg.content}\n"

    prompt = f"""
You are a helpful research assistant.

Answer ONLY using the context below.
Mention the page number(s) you used, e.g. "(page 12)".
If the answer is not present, say:
"I could not find this in the document."

Previous conversation:
{conversation}

Context:
{context}

Question:
{question}
"""

    return prompt
