from retriever import retrieve
from judge import judge_agent
import ollama


def verification_agent(query):

    # Step 1: retrieve documents
    chunks = retrieve(query, k=3)
    context = "\n\n".join(chunks)

    # Step 2: get answer from judge
    final_answer = judge_agent(query)

    verification_prompt = f"""
You are a verification AI.

Question:
{query}

Documents:
{context}

Original Answer:
{final_answer}

Tasks:
1. Check if the answer is fully supported by the documents.
2. Identify unsupported claims.
3. Rewrite the answer so that it only contains information supported by the documents.
4. Remove any hallucinated or unsupported conclusions.

Return the corrected final answer.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": verification_prompt}]
    )

    corrected_answer = response["message"]["content"]

    print("\nORIGINAL ANSWER:\n")
    print(final_answer)

    print("\nVERIFIED & CORRECTED ANSWER:\n")
    print(corrected_answer)


query = "Are we compliant with encryption policy?"

verification_agent(query)