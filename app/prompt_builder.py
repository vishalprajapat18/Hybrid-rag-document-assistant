from typing import List
from app.models import Message

def build_prompt(question: str, results,history:List[Message]):
    context = ""
#   for result in results:
#       context += f"""
#Page {result.payload['page']}:
#{result.payload['text']}
    for chunk in results:
      context += f"{chunk}\n\n"
    conversation =""
    for msg in history:
        conversation+= f"{msg.role}:{msg.content}\n"

    prompt = f"""
You are a helpful research assistant.

Answer ONLY using the context below.
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