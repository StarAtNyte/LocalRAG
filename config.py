class Config:
    MODEL_NAME = "llama3.2:latest"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    BATCH_SIZE = 5  
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    VECTOR_STORE_PATH = "vector_store"
    CLEAR_CUDA_CACHE = True  
    CUDA_CACHE_INTERVAL = 10  
    
    TEMPERATURE = 0.4
    NUM_CTX = 1500  
    NUM_THREAD = 4
    REPEAT_PENALTY = 1.2
    TOP_K = 5
    TOP_P = 0.9
    
    CHUNK_SIZE = 256  
    CHUNK_OVERLAP = 50  
    NUM_RETRIEVED_DOCS = 4  
    RERANK_TOP_K = 3  
    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3
    
    MAX_MEMORY_PERCENT = 90  
    BATCH_MEMORY_BUFFER = 1024  
    
    TRAIN_PATH = "data/train.csv"
    TEST_PATH = "data/test.csv"
    SUBMISSION_PATH = "data/submission.csv"