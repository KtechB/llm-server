from langchain import ConversationChain, PromptTemplate, LLMChain
from langchain.llms import OpenAIChat
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

from langchain.chains.conversation.memory import ConversationBufferWindowMemory


def ask_question(question: str) -> str:
    llm = OpenAIChat(model_name="gpt-3.5-turbo")
    template = """Question: {question}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(question)
    return answer


def create_conversational_chain():
    llm = OpenAIChat(model_name="gpt-3.5-turbo")

    system_template = "あなたは関西弁を巧みに使いこなす親切で気のいい狐です。人間と会話をしています。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # chatプロンプトテンプレートの準備
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            human_message_prompt,
        ]
    )

    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    return chain
