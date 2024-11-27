## System Overview

This system leverages the following components:

* ChatOllama: for generating responses.
* Hugging Face Embeddings: Model for text embedding generation.
* FAISS: Library for fast similarity search.
* Conversational Retrieval Chain: Chain responsible for retrieving relevant documents and combining them for response generation.
* ConversationBufferMemory: Memory component for storing conversation history.
