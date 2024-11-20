import os
from utils import *
from config import *
from prompts import *
from rag_system import *
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        print("Initializing RAG system...")
        rag_system = LocalRAGSystem()
        
        print("\nLoading and preparing training data...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        train_df = train_df.fillna({'Query': '', 'Response': ''})
        documents = rag_system.prepare_training_data(train_df)
        
        if not os.path.exists(Config.VECTOR_STORE_PATH):
            print("Creating new vector store...")
            rag_system.setup_hybrid_retrieval(documents)
        else:
            print("Loading existing vector store...")
            rag_system.setup_hybrid_retrieval(documents)  
        
        print("Setting up QA chain...")
        rag_system.setup_qa_chain()
        
        try:
            print("\nProcessing all queries...")
            rag_system.process_test_file()
            
            print("\nChecking for missing IDs...")
            test_df = pd.read_csv(Config.TEST_PATH)
            submission_df = pd.read_csv(Config.SUBMISSION_PATH)

            process_missing_ids(rag_system, test_df, submission_df)

            print("\nChecking for empty responses...")
            rag_system.process_empty_responses()
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving progress...")
            save_progress(Config.SUBMISSION_PATH, [], backup=True)
            
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            save_progress(Config.SUBMISSION_PATH, [], backup=True)
            raise
            
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()




    