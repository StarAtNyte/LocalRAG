#utils.py
import re
import pandas as pd
from typing import Set, Union
import os
from tqdm import tqdm
from config import Config
from typing import Set, Union
from typing import Optional, Dict, List
import time

def clean_response(text: Union[str, float, None]) -> str:
   
    if text is None:
        return ""
    if isinstance(text, (int, float)):
        return str(text)
    if not isinstance(text, str):
        return ""
    
    sentences = text.split('.')
    cleaned_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        patterns_to_remove = [
            # Question starters
            r'^(What|How|Why|When|Where|Which|Who|Describe|Define|Explain|List|Identify|Compare|Discuss|Analyze|Evaluate|Tell me about)\s*',
            # Question marks and related patterns
            r'\?+\s*:?\s*',
            r'question:\s*',
            # Common question-answer separators
            r':\s*\[.*?\]\s*',
            r':\s*\{\s*.*?\}\s*',
            # Metadata and formatting
            r'\b(text|question|answer|response|solution|output):\s*',
            r'(,\s*)?type\s*:\s*(answer|response|solution)\b',
            r'^\s*[\"\']|[\"\']$',
            r'^\s*[{[\(]|[}\]\)]\s*$',
            r'(I think|In my opinion|Based on|According to|Let me|I would say|I believe)\s*',
            r'(Here\'s|Here is|This is)\s*(the|a|my|an)?\s*(answer|response|explanation):\s*',
            r'```\w*\s*|```$',
            r'^\s*#\s*|^\s*\*\s*',
        ]

        cleaned_sentence = sentence.strip()
        for pattern in patterns_to_remove:
            cleaned_sentence = re.sub(pattern, '', cleaned_sentence, flags=re.IGNORECASE)
            
        cleaned_sentence = re.sub(r'.*?\?:?\s*', '', cleaned_sentence)
        cleaned_sentence = re.sub(r':\s*\[.*?\]', '', cleaned_sentence)
        cleaned_sentence = re.sub(r'\[.*?\]', '', cleaned_sentence)
        cleaned_sentence = re.sub(r'\(.*?\)', '', cleaned_sentence)
        cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)
        cleaned_sentence = cleaned_sentence.strip()
        
        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)
    
    text = '. '.join(cleaned_sentences)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\s*,\s*', ', ', text)  # Fix comma spacing
    text = re.sub(r'\.{2,}', '.', text)  # Fix multiple periods
    text = text.replace('"', '').replace('"', '').replace('"', '')  # Remove quotes
    text = text.replace('…', '').replace('−', '-')  # Fix special characters
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = text.strip()
    if text and not text[-1] in '.!?':
        text += '.'
        
    text = re.sub(r':\s*$', '.', text)
    
    return text.strip()

def batch_clean_responses(responses: pd.Series) -> pd.Series:
    
    return responses.fillna('').astype(str).apply(clean_response)


def find_missing_ids(test_path: str, submission_path: str) -> Set[int]:
    """Find trustii_ids that are in test.csv but missing from submission.csv"""
    test_df = pd.read_csv(test_path)
    submission_df = pd.read_csv(submission_path)
    
    test_ids = set(test_df['trustii_id'].dropna().astype(int))
    submission_ids = set(submission_df['trustii_id'].dropna().astype(int))
    
    missing_ids = test_ids - submission_ids
    print(f"Found {len(missing_ids)} missing IDs: {sorted(missing_ids)}")
    return missing_ids

def save_progress(output_path: str, results: list, backup: bool = True):
    """Save results to CSV with optional backup handling."""
    if backup and os.path.exists(output_path):
        backup_path = output_path.replace('.csv', '_backup.csv')
        if os.path.exists(backup_path):
            print(f"Backup file already exists: {backup_path}. Removing old backup.")
            os.remove(backup_path)
        os.rename(output_path, backup_path)
    
    df = pd.DataFrame(results)
    df['Response'] = df['Response'].apply(clean_response)
    df.to_csv(output_path, index=False)
    print(f"\nProgress saved: {len(df)} entries")

def process_missing_ids(rag_system, test_df: pd.DataFrame, submission_df: pd.DataFrame) -> None:
    """Process any missing IDs from the submission file."""
    test_ids = set(test_df['trustii_id'].dropna().astype(int))
    submission_ids = set(submission_df['trustii_id'].dropna().astype(int))
    missing_ids = test_ids - submission_ids
    
    if missing_ids:
        print(f"\nFound {len(missing_ids)} missing IDs: {sorted(missing_ids)}")
        print("Processing missing rows...")
        
        missing_rows = test_df[test_df['trustii_id'].isin(missing_ids)]
        missing_results = []
        
        for _, row in tqdm(missing_rows.iterrows(), total=len(missing_rows), desc="Processing missing rows"):
            query_data = {
                'trustii_id': row['trustii_id'],
                'Query': row['Query']
            }
            result = rag_system.process_single_query(query_data)
            missing_results.append(result)
        
        if missing_results:
            missing_df = pd.DataFrame(missing_results)
            combined_df = pd.concat([submission_df, missing_df], ignore_index=True)
            combined_df = combined_df.sort_values('trustii_id')
            combined_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print("\nSuccessfully added missing rows to submission.csv")
        
        final_df = pd.read_csv(Config.SUBMISSION_PATH)
        final_ids = set(final_df['trustii_id'].dropna().astype(int))
        still_missing = test_ids - final_ids
        
        if still_missing:
            print(f"\nWarning: Still missing IDs: {sorted(still_missing)}")
        else:
            print("\nAll IDs successfully processed!")
    else:
        print("\nNo missing IDs found - all rows processed successfully!")


def save_sorted_progress(filepath, results):
    """Save results to CSV file, sorted by trustii_id."""
    df = pd.DataFrame(results)
    df['trustii_id'] = df['trustii_id'].astype(int)
    df = df.sort_values('trustii_id')
    df.to_csv(filepath, index=False)
