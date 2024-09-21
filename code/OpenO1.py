import openai  # OpenAI API를 사용한다고 가정
import anthropic  # Anthropic(Claude) API를 사용한다고 가정
import phi  # 가상의 Phi-3.5 API를 가정

# 1. API 설정
openai.api_key = "your-openai-api-key"
claude = anthropic.Anthropic(api_key="your-anthropic-api-key")
phi_client = phi.PhiClient(api_key="your-phi-api-key")

# 2. 프롬프트 템플릿 정의
PROMPT_TEMPLATE = """
Answer the following question as best you can.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: Use self-asking approach to breakdown the Thought into simple sub-questions.
Observation: Answer the sub-questions one by one, very carefully, and verify step by step.
Self-Reflection: Criticize the Observation by asking questions and answering them one by one, about why the answer can be wrong, very carefully, and verify step by step.
... (this Thought/Action/Observation/Self-Reflection must repeat at least 2 times for consistency check)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {question}
"""

# 3. 각 모델에 대한 함수 정의
def query_gpt4(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1000
    )
    return response.choices[0].text.strip()

def query_claude(prompt):
    response = claude.completions.create(
        prompt=prompt,
        max_tokens_to_sample=1000
    )
    return response.completion

def query_phi(prompt):
    response = phi_client.complete(prompt, max_tokens=1000)
    return response.text

# 4. 메인 함수 정의
def solve_problem(question, model="gpt4"):
    full_prompt = PROMPT_TEMPLATE.format(question=question)
    
    if model == "gpt4":
        return query_gpt4(full_prompt)
    elif model == "claude":
        return query_claude(full_prompt)
    elif model == "phi":
        return query_phi(full_prompt)
    else:
        raise ValueError("Unsupported model")

# 5. 실행 및 결과 비교
question = "How many R's in the word Strawberry?"

print("GPT-4 Response:")
print(solve_problem(question, "gpt4"))

print("\nClaude Response:")
print(solve_problem(question, "claude"))

print("\nPhi-3.5 Response:")
print(solve_problem(question, "phi"))

# 6. 결과 분석 (간단한 예시)
def analyze_results(gpt4_response, claude_response, phi_response):
    models = ["GPT-4", "Claude", "Phi-3.5"]
    responses = [gpt4_response, claude_response, phi_response]
    
    for model, response in zip(models, responses):
        final_answer = response.split("Final Answer:")[-1].strip()
        print(f"{model} final answer: {final_answer}")
        
        steps = response.count("Thought:")
        print(f"{model} used {steps} thinking steps")
        
        if "Self-Reflection:" in response:
            print(f"{model} performed self-reflection")
        else:
            print(f"{model} did not perform self-reflection")

# 결과 분석 실행
gpt4_result = solve_problem(question, "gpt4")
claude_result = solve_problem(question, "claude")
phi_result = solve_problem(question, "phi")

analyze_results(gpt4_result, claude_result, phi_result)


'''
이 코드는 다음과 같은 단계로 구성되어 있습니다:

1. 필요한 API 라이브러리를 임포트하고 API 키를 설정합니다.
2. ReAct, Self-Ask, Reflection 방법론을 적용한 프롬프트 템플릿을 정의합니다.
3. 각 모델(GPT-4, Claude, Phi-3.5)에 대한 쿼리 함수를 정의합니다.
4. 문제를 해결하는 메인 함수를 정의합니다. 이 함수는 주어진 질문과 모델에 따라 적절한 쿼리 함수를 호출합니다.
5. 동일한 질문에 대해 세 모델의 응답을 얻고 출력합니다.
6. 결과를 분석하는 함수를 정의하고 실행합니다. 이 함수는 각 모델의 최종 답변, 사용된 사고 단계의 수, 자기 성찰 수행 여부 등을 비교합니다.

이 코드를 실행하면 각 모델의 응답을 비교하고 분석할 수 있습니다. 실제 환경에서는 각 API의 정확한 사용법과 에러 처리, 속도 제한 등을 고려해야 합니다. 또한, 결과 분석 부분을 더욱 정교하게 만들어 각 모델의 성능을 더 자세히 비교할 수 있습니다.
'''