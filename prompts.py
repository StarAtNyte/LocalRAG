from langchain.prompts import PromptTemplate

QA_TEMPLATE = """Context: {context}

Question: {question}

Instructions:
1. Carefully analyze the context and question
2. Donot create the question yourself from the context or chat history, only use them as reference and answer the query provided.
3. Ensure the answer directly addresses the question
4. Provide only the answer and avoid providing questions of any sort.
5. When asked for acronym only give the full form and nothing else.

Answer: Let me provide a response based on the context..."""

QA_PROMPT = PromptTemplate(
    template=QA_TEMPLATE,
    input_variables=["context", "question"]
)