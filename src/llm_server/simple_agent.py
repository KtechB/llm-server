from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAIChat


def ask_question(question: str) -> str:
    llm = OpenAIChat(model_name="gpt-3.5-turbo")
    template = """Question: {question}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(question)
    return answer
