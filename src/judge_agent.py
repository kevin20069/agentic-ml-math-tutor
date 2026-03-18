from reasoning_agent import reasoning_agent
import ollama

def judge_agent(query):

    # Step 1: get reasoning output
    reasoning_output = reasoning_agent(query)

    # Step 2: create debate prompts

    pro_prompt = f"""
You are a compliance expert defending the system.

Question:
{query}

AI reasoning:
{reasoning_output}

Explain why the system can still be considered compliant.
"""
    con_prompt = f"""
You are a security auditor criticizing the system.

Question:
{query}

AI reasoning:
{reasoning_output}

Explain why the system is NOT compliant.
"""

    pro_response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": pro_prompt}]
    )

    con_response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": con_prompt}]
    )

    pro_argument = pro_response["message"]["content"]
    con_argument = con_response["message"]["content"]

    # Step 3: Judge decision

    judge_prompt = f"""
You are a senior AI judge.

Question:
{query}

Reasoning Output:
{reasoning_output}

Pro Argument:
{pro_argument}

Con Argument:
{con_argument}

Your task:
1. Evaluate both arguments
2. Decide which is stronger
3. Provide a final conclusion
"""

    judge_response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    final_decision = judge_response["message"]["content"]

    print("\nREASONING OUTPUT:\n")
    print(reasoning_output)

    print("\nPRO ARGUMENT:\n")
    print(pro_argument)

    print("\nCON ARGUMENT:\n")
    print(con_argument)

    print("\nFINAL JUDGEMENT:\n")
    print(final_decision)
    return final_decision


query = "Are we compliant with encryption policy?"

judge_agent(query)

