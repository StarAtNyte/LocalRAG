import os
from typing import List, Dict, Any, Tuple
import pandas as pd
import torch
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import numpy as np
from tqdm import tqdm
import platform
import math
import signal
import psutil
import subprocess
import time
from config import Config
from prompts import QA_PROMPT
from utils import clean_response, save_progress
import signal
import gc
from contextlib import contextmanager
from typing import Optional, Dict, List
import re
from langchain.callbacks import get_openai_callback
from utils import *

                
class LocalRAGSystem:
    def __init__(self):
        self.config = Config()
        self.setup_system()
        self.setup_models()
        self.memory = self.setup_memory()
        self.batch_queue = []
         
    def setup_system(self):
        """Initialize system and server."""
        self.system = platform.system()
        self.ollama_process = None
        self.start_ollama_server()

    def setup_models(self):
        """Initialize all necessary models."""
        try:
            self.setup_llm()
            self.setup_embeddings()
            self.setup_cross_encoder()
        except Exception as e:
            print(f"Error setting up models: {str(e)}")
            raise

    def setup_llm(self):
        """Initialize the LLM with optimized settings."""
        try:
            self.llm = ChatOllama(
                model=self.config.MODEL_NAME,
                format="json",
                temperature=self.config.TEMPERATURE,
                num_ctx=self.config.NUM_CTX,
                num_thread=self.config.NUM_THREAD,
                repeat_penalty=self.config.REPEAT_PENALTY,
                top_k=self.config.TOP_K,
                top_p=self.config.TOP_P,
                timeout=60
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

    def setup_embeddings(self):
        """Initialize embedding model."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': device},
                encode_kwargs={
                    'device': device,
                    'batch_size': 32,
                    'normalize_embeddings': True
                }
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

    def setup_cross_encoder(self):
        """Initialize cross-encoder for reranking."""
        try:
            self.cross_encoder = CrossEncoder(self.config.CROSS_ENCODER_MODEL)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize cross-encoder: {str(e)}")

    def check_gpu_memory(self):
        """Monitor GPU memory usage and clear if necessary."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            memory_percent = (memory_allocated / memory_reserved) * 100 if memory_reserved > 0 else 0
            
            if memory_percent > self.config.MAX_MEMORY_PERCENT:
                self.clear_gpu_memory()
    
    def clear_gpu_memory(self):
        """Clear GPU memory and cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("Cleared GPU memory cache")
    
    @contextmanager
    def memory_management(self):
        """Context manager for memory management."""
        try:
            yield
        finally:
            self.check_gpu_memory()
            
    def setup_memory(self):
        """Initialize conversation memory."""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )
    
    def clear_chat_history(self):
        """Clear the chat history periodically."""
        print("Clearing chat history...")
        self.memory.clear()

    def start_ollama_server(self):
        """Start Ollama server if not running."""
        if not self.find_ollama_process():
            self.run_ollama_process()
        else:
            print("Ollama server is already running.")

    def run_ollama_process(self):
        """Run Ollama process depending on the platform."""
        try:
            if self.system == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                self.ollama_process = subprocess.Popen(
                    ['ollama.exe', 'serve'],
                    startupinfo=startupinfo,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                self.ollama_process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            time.sleep(10)
            print("Ollama server started successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to start Ollama server: {str(e)}")
        
    def restart_ollama_server(self):
        """Restart the Ollama server."""
        print("\nRestarting Ollama server...")
        
        ollama_process = self.find_ollama_process()
        if ollama_process:
            try:
                if self.system == 'Windows':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(ollama_process.pid)], 
                                check=True, 
                                stderr=subprocess.DEVNULL)
                else:
                    os.kill(ollama_process.pid, signal.SIGTERM)
                time.sleep(5)  
            except Exception as e:
                print(f"Error terminating Ollama process: {str(e)}")
        
        self.start_ollama_server()
        time.sleep(10)  
        
        self.setup_llm()
        print("Ollama server restarted successfully!")

    def find_ollama_process(self):
        """Find running Ollama process."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return None

    def terminate_ollama_process(self):
        """Terminate the existing Ollama process."""
        ollama_process = self.find_ollama_process()
        if ollama_process:
            try:
                self.kill_process(ollama_process)
                time.sleep(5) 
            except Exception as e:
                print(f"Error terminating Ollama process: {str(e)}")

    def kill_process(self, process):
        """Kill the provided process based on the system platform."""
        try:
            if self.system == 'Windows':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], check=True, stderr=subprocess.DEVNULL)
            else:
                os.kill(process.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process {process.pid}: {str(e)}")


    def prepare_training_data(self, train_df: pd.DataFrame) -> List[Any]:
        """Prepare training data with enhanced chunking."""
        try:
            train_df = train_df.fillna({'Query': '', 'Response': ''})
            train_df['combined'] = train_df.apply(
                lambda row: f"Question: {row['Query']}\nAnswer: {row['Response']}\n\n",
                axis=1
            )
            
            loader = DataFrameLoader(train_df, page_content_column="combined")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            split_documents = text_splitter.split_documents(documents)
            print(f"Prepared {len(split_documents)} document chunks")
            return split_documents
        except Exception as e:
            raise RuntimeError(f"Failed to prepare training data: {str(e)}")

    def setup_hybrid_retrieval(self, documents: List[Any]):
        """Set up hybrid retrieval system with proper error handling."""
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            dense_retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": self.config.NUM_RETRIEVED_DOCS,
                    "score_threshold": 0.75
                }
            )

            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = self.config.NUM_RETRIEVED_DOCS

            self.retriever = EnsembleRetriever(
                retrievers=[dense_retriever, bm25_retriever],
                weights=[self.config.DENSE_WEIGHT, self.config.SPARSE_WEIGHT]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to setup hybrid retrieval: {str(e)}")

    def rerank_documents(self, query: str, documents: List[Any]) -> List[Any]:
        """Rerank retrieved documents using cross-encoder with error handling."""
        try:
            if not documents:
                return documents

            valid_documents = [
                doc for doc in documents 
                if hasattr(doc, 'page_content') and isinstance(doc.page_content, str)
            ]
            
            if not valid_documents:
                return []

            pairs = [[query, doc.page_content] for doc in valid_documents]
            scores = self.cross_encoder.predict(pairs)
            
            scores = [float(score) if isinstance(score, (int, float, np.number)) else 0.0 for score in scores]
            
            doc_score_pairs = list(zip(valid_documents, scores))
            reranked_docs = [doc for doc, score in sorted(doc_score_pairs, key=lambda x: float(x[1]), reverse=True)]
            
            return reranked_docs[:self.config.RERANK_TOP_K]
        except Exception as e:
            print(f"Reranking error: {str(e)}")
            return documents[:self.config.RERANK_TOP_K]

    def process_single_query(self, query_data: Dict) -> Dict:
        """Process a single query with periodic chat history clearance."""
        
        if isinstance(query_data['Query'], float) and math.isnan(query_data['Query']) or not query_data['Query'].strip():
            return {
                    'trustii_id': query_data['trustii_id'],
                    'Query': query_data['Query'],
                    'Response': ''  
            }
        
        try:
            retrieved_docs = self.retriever.get_relevant_documents(query_data['Query'])
            retrieved_docs = retrieved_docs[:self.config.NUM_RETRIEVED_DOCS]
            
            reranked_docs = self.rerank_documents(query_data['Query'], retrieved_docs)
            context = "\n\n".join([doc.page_content for doc in reranked_docs[:5]])  
            
            if len(self.memory.chat_memory.messages) > 20:
                self.memory.clear()
            
            result = self.qa_chain.invoke(
                {
                    "question": query_data['Query'],
                    "context": context[:8000],  
                    "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])
                }
            )
            
            response = {
                'trustii_id': query_data['trustii_id'],
                'Query': query_data['Query'],
                'Response': clean_response(result['answer'])
            }
            
            self.memory.chat_memory.add_user_message(query_data['Query'])
            self.memory.chat_memory.add_ai_message(response['Response'])
            
            return response
            
        except Exception as e:
            print(f"Error processing query {query_data['trustii_id']}: {str(e)}")
            return {
                'trustii_id': query_data['trustii_id'],
                'Query': query_data['Query'],
                'Response': f"Error: {str(e)}"
            }

    def setup_qa_chain(self):
        """Set up QA chain with custom prompt."""
        try:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                combine_docs_chain_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to setup QA chain: {str(e)}")

    def get_processed_queries(self, submission_path: str) -> set:
        """Get set of trustii_ids that have already been processed successfully."""
        try:
            if not os.path.exists(submission_path):
                return set()
                
            submission_df = pd.read_csv(submission_path)
            
            processed_ids = submission_df[
                (submission_df['Response'].notna()) & 
                (submission_df['Response'] != '') &
                (~submission_df['Response'].str.startswith('Error:', na=False))
            ]['trustii_id'].astype(int).tolist()
            
            return set(processed_ids)
        except Exception as e:
            print(f"Error reading submission file: {str(e)}")
            return set()
        
    def process_test_file(self):
        """Process test file, skipping already processed queries and invalid queries."""
        test_df = pd.read_csv(self.config.TEST_PATH)
        
        processed_ids = self.get_processed_queries(self.config.SUBMISSION_PATH)
        print(f"Found {len(processed_ids)} already processed queries")
        
        unprocessed_df = test_df[~test_df['trustii_id'].astype(int).isin(processed_ids)]
        print(f"Remaining queries to process: {len(unprocessed_df)}")
        
        if len(unprocessed_df) == 0:
            print("All queries have been processed!")
            return
        
        results = []
        if os.path.exists(self.config.SUBMISSION_PATH):
            existing_df = pd.read_csv(self.config.SUBMISSION_PATH)
            results = existing_df.to_dict('records')
        
        query_count = 0
        
        for _, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing queries"):
            
            if isinstance(row['Query'], float) and math.isnan(row['Query']) or not row['Query'].strip():
                print(f"Skipping invalid or empty query for trustii_id {row['trustii_id']}")
                continue  

            if query_count > 0 and query_count % 50 == 0:
                self.restart_ollama_server()
                
            query_data = {
                'trustii_id': row['trustii_id'],
                'Query': row['Query']
            }
            
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    result = self.process_single_query(query_data)
                    
                    if (
                        result['Response'] and 
                        not pd.isna(result['Response']) and 
                        not result['Response'].startswith('Error:')
                    ):
                        success = True
                        if results:
                            existing_idx = next(
                                (i for i, r in enumerate(results) 
                                if r['trustii_id'] == result['trustii_id']), 
                                None
                            )
                            if existing_idx is not None:
                                results[existing_idx] = result
                            else:
                                results.append(result)
                        else:
                            results.append(result)
                    else:
                        retry_count += 1
                        print(f"\nRetry {retry_count} for trustii_id {query_data['trustii_id']}")
                        time.sleep(2)
                except Exception as e:
                    print(f"\nError processing trustii_id {query_data['trustii_id']}: {str(e)}")
                    retry_count += 1
                    time.sleep(2)
                    
                    if "connection" in str(e).lower() or "timeout" in str(e).lower():
                        print("Possible server issue detected, restarting Ollama...")
                        self.restart_ollama_server()
            
            if len(results) % 10 == 0:
                save_sorted_progress(self.config.SUBMISSION_PATH, results)
            
            query_count += 1
        
        self.final_save_with_missing_check(test_df=test_df, results=results)
    
    def final_save_with_missing_check(self, test_df: pd.DataFrame, results: List[Dict]):
        """Perform final save with check for missing IDs and one last attempt to process them."""
        print("\nPerforming final check for missing IDs...")
        
        results_df = pd.DataFrame(results)
        all_test_ids = set(test_df['trustii_id'].astype(int))
        processed_ids = set(results_df['trustii_id'].astype(int))
        missing_ids = all_test_ids - processed_ids
        
        if missing_ids:
            print(f"Found {len(missing_ids)} missing IDs. Attempting to process them...")
            
            for missing_id in tqdm(missing_ids, desc="Processing missing IDs"):
                query_row = test_df[test_df['trustii_id'] == missing_id].iloc[0]
                query_data = {
                    'trustii_id': missing_id,
                    'Query': query_row['Query']
                }
                
                try:
                    result = self.process_single_query(query_data)
                    
                    if (
                        not result['Response'] or 
                        pd.isna(result['Response']) or 
                        result['Response'].startswith('Error:')
                    ):
                        result = {
                            'trustii_id': missing_id,
                            'Query': query_row['Query'],
                            'Response': ''
                        }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"\nError processing missing ID {missing_id}: {str(e)}")
                    results.append({
                        'trustii_id': missing_id,
                        'Query': query_row['Query'],
                        'Response': ''
                    })
        
        save_sorted_progress(self.config.SUBMISSION_PATH, results)
        
        final_df = pd.read_csv(self.config.SUBMISSION_PATH)
        final_ids = set(final_df['trustii_id'].astype(int))
        if final_ids == all_test_ids:
            print("Final save successful! All test IDs are present in the submission file.")
        else:
            remaining_missing = all_test_ids - final_ids
            print(f"Warning: There are still {len(remaining_missing)} missing IDs in the final save.")
            print("Missing IDs:", sorted(remaining_missing))

    def process_empty_responses(self, max_retries: int = 3):
        """Process empty responses, skipping already processed ones."""
        try:
            submission_df = pd.read_csv(self.config.SUBMISSION_PATH)
            
            processed_ids = self.get_processed_queries(self.config.SUBMISSION_PATH)
            
            empty_responses = submission_df[
                (~submission_df['trustii_id'].astype(int).isin(processed_ids)) &
                (
                    (submission_df['Response'].isna()) | 
                    (submission_df['Response'] == '') |
                    (submission_df['Response'].str.startswith('Error:'))
                )
            ]
            
            if empty_responses.empty:
                print("No empty responses found that need processing.")
                return
            
            print(f"Found {len(empty_responses)} queries with empty or error responses to retry.")
            results = []
            query_count = 0
            
            for _, row in tqdm(empty_responses.iterrows(), total=len(empty_responses), desc="Retrying empty responses"):
                if query_count > 0 and query_count % 50 == 0:
                    self.restart_ollama_server()
                    
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        query_data = {
                            'trustii_id': row['trustii_id'],
                            'Query': row['Query']
                        }
                        
                        self.memory.clear()
                        
                        result = self.process_single_query(query_data)
                        
                        if (
                            result['Response'] and 
                            not pd.isna(result['Response']) and 
                            not result['Response'].startswith('Error:')
                        ):
                            success = True
                            results.append(result)
                        else:
                            retry_count += 1
                            print(f"\nRetry {retry_count} for trustii_id {row['trustii_id']}")
                            time.sleep(2)
                            
                    except Exception as e:
                        print(f"\nError processing trustii_id {row['trustii_id']}: {str(e)}")
                        retry_count += 1
                        time.sleep(2)
                        
                        if "connection" in str(e).lower() or "timeout" in str(e).lower():
                            print("Possible server issue detected, restarting Ollama...")
                            self.restart_ollama_server()
                
                if not success:
                    print(f"\nFailed to get valid response for trustii_id {row['trustii_id']} after {max_retries} attempts")
                
                query_count += 1
            
            if results:
                current_df = pd.read_csv(self.config.SUBMISSION_PATH)
                new_results_df = pd.DataFrame(results)
                
                for _, row in new_results_df.iterrows():
                    current_df.loc[
                        current_df['trustii_id'] == row['trustii_id'], 
                        'Response'
                    ] = row['Response']
                
                save_progress(self.config.SUBMISSION_PATH, current_df.to_dict('records'))
                print(f"\nSuccessfully updated {len(results)} responses in submission file")
        except Exception as e:
            print(f"Error processing empty responses: {str(e)}")
            if "No such file or directory" in str(e):
                empty_df = pd.DataFrame(columns=['trustii_id', 'Query', 'Response'])
                save_progress(self.config.SUBMISSION_PATH, empty_df.to_dict('records'))