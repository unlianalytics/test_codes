# main3.py - Enhanced RAG Implementation for T-Mobile RF AI Agent
# This version focuses on improved RAG performance and better knowledge base utilization

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import PyPDF2
import chromadb
import numpy as np
from typing import List, Dict, Set, Optional
import re
import glob
import pandas as pd
from datetime import datetime, timedelta
import warnings
import hashlib
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import logging
import threading
import oracledb
import csv

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="T-Mobile RF AI Agent (Enhanced RAG)", description="Enhanced AI agent with improved RAG capabilities for RF Engineering")

# Mount static files
import os
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Initialize OpenAI client (OpenAI > 1.0 syntax)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"))

# OpenAI embedding model configuration
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's embedding model

# Initialize ChromaDB for vector storage
chroma_client = chromadb.Client()
collection_name = "tmobile_rf_knowledge_enhanced"
try:
    knowledge_collection = chroma_client.get_collection(collection_name)
    logger.info(f"Using existing collection: {collection_name}")
except:
    knowledge_collection = chroma_client.create_collection(collection_name)
    logger.info(f"Created new collection: {collection_name}")

# ===== ENHANCED CONFIGURATION SECTION =====
# File Discovery Configuration
PDF_DIRECTORY = r"C:\Users\magno\Downloads\pdf_files"  # Directory containing PDFs
CSV_DIRECTORY = r"C:\Users\magno\Downloads\csv_metrics"  # Directory containing CSV metrics
CSV_FALLBACK_DIRECTORY = "csv_files"  # Fallback directory for local CSV files
CACHE_DIRECTORY = r"C:\Users\magno\Downloads\agent_cache"  # Cache directory for processed data
AUTO_LOAD_PDFS = True  # Set to True for automatic PDF discovery
AUTO_LOAD_CSVS = True  # Set to True for automatic CSV discovery

# Enhanced Performance Configuration
MAX_WORKERS = 4          # Parallel workers (increase for more cores)
BATCH_SIZE = 50          # Reduced batch size for better embedding quality
CHUNK_SIZE = 1000        # Increased chunk size for better context
CHUNK_OVERLAP = 200      # Overlap between chunks for better continuity
ENABLE_CACHING = True    # Enable file processing cache
ENABLE_INCREMENTAL = True # Enable incremental processing

# RAG Enhancement Configuration
MIN_RELEVANCE_SCORE = 0.6  # Minimum relevance score for documents
MAX_CONTEXT_LENGTH = 4000  # Maximum context length for LLM
ENABLE_HYBRID_SEARCH = True  # Enable hybrid search (semantic + keyword)
ENABLE_RERANKING = True   # Enable result reranking

# Add this to the configuration section
FORCE_RELOAD_ON_STARTUP = True  # Set to True to force reload all files on startup
CHECK_EMPTY_KB_ON_STARTUP = True  # Set to True to check if KB is empty and reload if needed
USE_FALLBACK_CSV = True  # Use fallback CSV directory if main directory is empty

# ===== END CONFIGURATION =====

# Generate embeddings function
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI's embedding model"""
    try:
        if not texts:
            return []
        
        # Generate embeddings using OpenAI
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        
        # Extract embeddings from response
        embeddings = [data.embedding for data in response.data]
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []

# Enhanced file tracking for incremental processing
class EnhancedFileTracker:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tracker_file = self.cache_dir / "enhanced_file_tracker.pkl"
        self.processed_files = self.load_tracker()
    
    def load_tracker(self) -> Dict[str, Dict]:
        """Load processed files tracker"""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_tracker(self):
        """Save processed files tracker"""
        with open(self.tracker_file, 'wb') as f:
            pickle.dump(self.processed_files, f)
    
    def get_file_hash(self, file_path: str) -> str:
        """Get file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def is_file_modified(self, file_path: str) -> bool:
        """Check if file has been modified since last processing"""
        file_hash = self.get_file_hash(file_path)
        last_hash = self.processed_files.get(file_path, {}).get('hash', '')
        return file_hash != last_hash
    
    def mark_file_processed(self, file_path: str, chunks_count: int):
        """Mark file as processed"""
        self.processed_files[file_path] = {
            'hash': self.get_file_hash(file_path),
            'processed_at': datetime.now().isoformat(),
            'chunks_count': chunks_count
        }
        self.save_tracker()
    
    def get_unprocessed_files(self, file_paths: List[str]) -> List[str]:
        """Get list of files that need processing"""
        return [path for path in file_paths if self.is_file_modified(path)]

# Initialize enhanced file tracker
file_tracker = EnhancedFileTracker(CACHE_DIRECTORY)

# Enhanced PDF processing function with better chunking
def process_pdf_enhanced(pdf_path: str, source_name: str = "unknown") -> List[Dict]:
    """Extract text from PDF with enhanced chunking strategy"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_chunks = []
            
            logger.info(f"Processing PDF: {source_name} ({len(pdf_reader.pages)} pages)")
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Use enhanced chunking with overlap
                    chunks = split_text_into_chunks_enhanced(text, CHUNK_SIZE, CHUNK_OVERLAP)
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            text_chunks.append({
                                'text': chunk.strip(),
                                'page': page_num + 1,
                                'chunk_id': f"{source_name}_page_{page_num + 1}_chunk_{i + 1}",
                                'source': source_name,
                                'file_path': pdf_path,
                                'file_type': 'pdf',
                                'chunk_index': i,
                                'total_chunks': len(chunks)
                            })
            
            logger.info(f"Extracted {len(text_chunks)} chunks from {source_name}")
            return text_chunks
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return []

# Enhanced CSV processing function
def process_csv_enhanced(csv_path: str, source_name: str = "unknown") -> List[Dict]:
    """Process CSV files with enhanced analysis"""
    try:
        # Read CSV file with optimized settings
        df = pd.read_csv(csv_path, nrows=10000)  # Limit rows for large files
        
        # Get basic information about the dataset
        rows, cols = df.shape
        columns = list(df.columns)
        
        # Create enhanced summary information
        summary_info = f"Dataset: {source_name}\n"
        summary_info += f"Dimensions: {rows} rows × {cols} columns\n"
        summary_info += f"Columns: {', '.join(columns)}\n"
        
        # Get data types
        dtypes_info = "Data Types:\n"
        for col, dtype in df.dtypes.items():
            dtypes_info += f"  - {col}: {dtype}\n"
        
        # Get enhanced statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_info = "Numeric Column Statistics:\n"
            for col in numeric_cols[:10]:  # Increased to 10 columns
                stats = df[col].describe()
                stats_info += f"  - {col}:\n"
                stats_info += f"    Mean: {stats['mean']:.4f}\n"
                stats_info += f"    Std: {stats['std']:.4f}\n"
                stats_info += f"    Min: {stats['min']:.4f}\n"
                stats_info += f"    Max: {stats['max']:.4f}\n"
                stats_info += f"    Median: {stats['50%']:.4f}\n"
        else:
            stats_info = "No numeric columns found for statistical analysis.\n"
        
        # Get sample data
        sample_data = "Sample Data (first 5 rows):\n"
        sample_data += df.head(5).to_string(index=False)
        
        # Create chunks from the information
        chunks = []
        
        # Summary chunk
        chunks.append({
            'text': summary_info,
            'chunk_id': f"{source_name}_summary",
            'source': source_name,
            'file_path': csv_path,
            'file_type': 'csv',
            'data_type': 'summary'
        })
        
        # Data types chunk
        chunks.append({
            'text': dtypes_info,
            'chunk_id': f"{source_name}_dtypes",
            'source': source_name,
            'file_path': csv_path,
            'file_type': 'csv',
            'data_type': 'schema'
        })
        
        # Statistics chunk
        chunks.append({
            'text': stats_info,
            'chunk_id': f"{source_name}_stats",
            'source': source_name,
            'file_path': csv_path,
            'file_type': 'csv',
            'data_type': 'statistics'
        })
        
        # Sample data chunk
        chunks.append({
            'text': sample_data,
            'chunk_id': f"{source_name}_sample",
            'source': source_name,
            'file_path': csv_path,
            'file_type': 'csv',
            'data_type': 'sample'
        })
        
        # Enhanced network KPI analysis chunk
        kpi_analysis = analyze_network_kpis_enhanced(df, source_name)
        chunks.append({
            'text': kpi_analysis,
            'chunk_id': f"{source_name}_kpi_analysis",
            'source': source_name,
            'file_path': csv_path,
            'file_type': 'csv',
            'data_type': 'kpi_analysis'
        })
        
        logger.info(f"Processed CSV: {source_name} ({len(chunks)} chunks)")
        return chunks
    except Exception as e:
        logger.error(f"Error processing CSV {csv_path}: {e}")
        return []

# Enhanced network KPI analysis function
def analyze_network_kpis_enhanced(df: pd.DataFrame, source_name: str) -> str:
    """Enhanced analysis of network KPI metrics from CSV data"""
    try:
        analysis = f"Enhanced Network KPI Analysis for {source_name}:\n\n"
        
        # Look for common RF engineering metrics with expanded keywords
        rf_metrics = {
            'signal_strength': ['rsrp', 'rsrq', 'rssi', 'signal_strength', 'power', 'dbm', 'signal_power'],
            'quality': ['sinr', 'snr', 'quality', 'ber', 'fer', 'signal_quality', 'link_quality'],
            'throughput': ['throughput', 'data_rate', 'speed', 'mbps', 'kbps', 'bandwidth', 'capacity'],
            'coverage': ['coverage', 'distance', 'range', 'area', 'coverage_area', 'cell_coverage'],
            'interference': ['interference', 'noise', 'interference_ratio', 'noise_floor', 'interference_power'],
            'latency': ['latency', 'delay', 'ping', 'rtt', 'response_time', 'propagation_delay'],
            'mobility': ['handover', 'mobility', 'roaming', 'cell_change', 'location_update'],
            'capacity': ['capacity', 'load', 'utilization', 'congestion', 'traffic_load']
        }
        
        found_metrics = {}
        columns_lower = [col.lower() for col in df.columns]
        
        for category, keywords in rf_metrics.items():
            found_metrics[category] = []
            for keyword in keywords:
                for col in df.columns:
                    if keyword in col.lower():
                        found_metrics[category].append(col)
        
        # Analyze found metrics with enhanced insights
        for category, metrics in found_metrics.items():
            if metrics:
                analysis += f"{category.replace('_', ' ').title()} Metrics Found:\n"
                for metric in metrics:
                    if metric in df.columns:
                        col_data = df[metric].dropna()
                        if len(col_data) > 0:
                            if pd.api.types.is_numeric_dtype(col_data):
                                mean_val = col_data.mean()
                                std_val = col_data.std()
                                min_val = col_data.min()
                                max_val = col_data.max()
                                median_val = col_data.median()
                                
                                analysis += f"  - {metric}:\n"
                                analysis += f"    Mean: {mean_val:.4f}\n"
                                analysis += f"    Std: {std_val:.4f}\n"
                                analysis += f"    Min: {min_val:.4f}\n"
                                analysis += f"    Max: {max_val:.4f}\n"
                                analysis += f"    Median: {median_val:.4f}\n"
                                
                                # Add interpretation for common metrics
                                if 'rsrp' in metric.lower():
                                    if mean_val > -80:
                                        analysis += f"    Interpretation: Excellent signal strength\n"
                                    elif mean_val > -100:
                                        analysis += f"    Interpretation: Good signal strength\n"
                                    else:
                                        analysis += f"    Interpretation: Poor signal strength\n"
                                elif 'sinr' in metric.lower():
                                    if mean_val > 20:
                                        analysis += f"    Interpretation: Excellent signal quality\n"
                                    elif mean_val > 10:
                                        analysis += f"    Interpretation: Good signal quality\n"
                                    else:
                                        analysis += f"    Interpretation: Poor signal quality\n"
                            else:
                                unique_count = col_data.nunique()
                                analysis += f"  - {metric}: {unique_count} unique values\n"
                analysis += "\n"
        
        if not any(found_metrics.values()):
            analysis += "No specific RF engineering metrics found. This may be a general dataset.\n"
        
        return analysis
    except Exception as e:
        return f"Error analyzing KPIs: {str(e)}"

# Enhanced text chunking function with overlap
def split_text_into_chunks_enhanced(text: str, max_length: int, overlap: int) -> List[str]:
    """Split text into enhanced chunks using sentence boundaries with overlap"""
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_length, save current chunk
        if len(current_chunk) + len(sentence) + 1 > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous chunk
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + ". " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Enhanced batch processing for knowledge base
def add_to_knowledge_base_enhanced(chunks: List[Dict]):
    """Add chunks to knowledge base with enhanced processing"""
    try:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            
            # Prepare batch data
            texts = [chunk['text'] for chunk in batch]
            metadatas = [{
                'source': chunk.get('source', 'unknown'),
                'file_type': chunk.get('file_type', 'unknown'),
                'page': chunk.get('page', ''),
                'chunk_id': chunk.get('chunk_id', ''),
                'file_path': chunk.get('file_path', ''),
                'data_type': chunk.get('data_type', ''),
                'chunk_index': chunk.get('chunk_index', ''),
                'total_chunks': chunk.get('total_chunks', '')
            } for chunk in batch]
            ids = [chunk.get('chunk_id', f"chunk_{i}_{j}") for j, chunk in enumerate(batch)]
            
            # Generate embeddings
            embeddings = generate_embeddings(texts)
            
            if embeddings and len(embeddings) == len(texts):
                # Add to ChromaDB
                knowledge_collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added batch {i//BATCH_SIZE + 1} with {len(batch)} chunks")
            else:
                logger.error(f"Failed to generate embeddings for batch {i//BATCH_SIZE + 1} - expected {len(texts)}, got {len(embeddings) if embeddings else 0}")
        
        return True
    except Exception as e:
        logger.error(f"Error adding batch to knowledge base: {e}")
        return False

# Enhanced search function with multiple retrieval methods
def search_knowledge_base_enhanced(query: str, top_k: int = 8) -> List[Dict]:
    """Enhanced search with multiple retrieval methods and reranking"""
    try:
        # Generate query embedding
        embeddings = generate_embeddings([query])
        
        if not embeddings or len(embeddings) == 0:
            logger.error("Failed to generate query embedding")
            return []
        
        query_embedding = embeddings[0]
        
        # Search in ChromaDB with more results for reranking
        results = knowledge_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more results for reranking
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                relevance_score = 1 - distance
                
                # Apply minimum relevance threshold
                if relevance_score >= MIN_RELEVANCE_SCORE:
                    formatted_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'relevance_score': relevance_score,
                        'distance': distance
                    })
        
        # Rerank results if enabled
        if ENABLE_RERANKING and formatted_results:
            formatted_results = rerank_results(query, formatted_results)
        
        # Return top_k results
        return formatted_results[:top_k]
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return []

# Result reranking function
def rerank_results(query: str, results: List[Dict]) -> List[Dict]:
    """Rerank results based on multiple factors"""
    try:
        query_lower = query.lower()
        
        for result in results:
            text_lower = result['text'].lower()
            
            # Calculate keyword match score
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            keyword_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            
            # Calculate length penalty (prefer medium-length chunks)
            length_penalty = 1.0
            text_length = len(result['text'])
            if text_length < 100:
                length_penalty = 0.8  # Penalize very short chunks
            elif text_length > 2000:
                length_penalty = 0.9  # Slightly penalize very long chunks
            
            # Combine scores
            final_score = (result['relevance_score'] * 0.7 + keyword_overlap * 0.3) * length_penalty
            result['final_score'] = final_score
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
        
    except Exception as e:
        logger.error(f"Error reranking results: {e}")
        return results

# File discovery functions
def get_pdf_files_from_directory(directory: str) -> Dict[str, str]:
    """Get all PDF files from a directory"""
    pdf_files = {}
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return {}
        
        pdf_pattern = os.path.join(directory, "*.pdf")
        pdf_paths = glob.glob(pdf_pattern)
        
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            source_name = os.path.splitext(filename)[0]
            pdf_files[source_name] = pdf_path
            
        return pdf_files
    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {e}")
        return {}

def get_csv_files_from_directory(directory: str) -> Dict[str, str]:
    """Get all CSV files from a directory"""
    csv_files = {}
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return {}
        
        csv_pattern = os.path.join(directory, "*.csv")
        csv_paths = glob.glob(csv_pattern)
        
        for csv_path in csv_paths:
            filename = os.path.basename(csv_path)
            source_name = os.path.splitext(filename)[0]
            csv_files[source_name] = csv_path
            
        return csv_files
    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {e}")
        return {}

# Parallel processing function
def process_file_parallel(file_path: str, file_type: str, source_name: str) -> List[Dict]:
    """Process a single file in parallel"""
    try:
        if file_type == 'pdf':
            return process_pdf_enhanced(file_path, source_name)
        elif file_type == 'csv':
            return process_csv_enhanced(file_path, source_name)
        else:
            return []
    except Exception as e:
        logger.error(f"Error processing {file_type} file {file_path}: {e}")
        return []

# Enhanced knowledge base initialization
def initialize_knowledge_base_enhanced():
    """Initialize knowledge base with enhanced processing"""
    logger.info("🚀 Initializing Enhanced T-Mobile RF AI Agent Knowledge Base...")
    
    # Check if we should force reload
    if FORCE_RELOAD_ON_STARTUP:
        logger.info("🔄 Force reload enabled - processing all files")
        # Clear the file tracker to force reprocessing
        file_tracker.processed_files = {}
        file_tracker.save_tracker()
    
    # Check if knowledge base is empty and reload if needed
    if CHECK_EMPTY_KB_ON_STARTUP:
        current_count = knowledge_collection.count()
        if current_count == 0:
            logger.info("📚 Knowledge base is empty - forcing reload of all files")
            # Clear the file tracker to force reprocessing
            file_tracker.processed_files = {}
            file_tracker.save_tracker()
        else:
            logger.info(f"📚 Knowledge base contains {current_count} chunks")
    
    all_chunks = []
    processed_files = []
    
    start_time = time.time()
    
    # Process PDF files
    if AUTO_LOAD_PDFS:
        logger.info(f"📚 Processing PDF files from: {PDF_DIRECTORY}")
        pdf_files = get_pdf_files_from_directory(PDF_DIRECTORY)
        
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            # Get unprocessed files
            if ENABLE_INCREMENTAL:
                unprocessed_pdfs = file_tracker.get_unprocessed_files(list(pdf_files.values()))
                logger.info(f"Processing {len(unprocessed_pdfs)} new/modified PDF files")
            else:
                unprocessed_pdfs = list(pdf_files.values())
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file = {}
                for source_name, pdf_path in pdf_files.items():
                    if pdf_path in unprocessed_pdfs:
                        future = executor.submit(process_file_parallel, pdf_path, 'pdf', source_name)
                        future_to_file[future] = (source_name, pdf_path)
                
                for future in as_completed(future_to_file):
                    source_name, pdf_path = future_to_file[future]
                    try:
                        chunks = future.result()
                        if chunks:
                            all_chunks.extend(chunks)
                            processed_files.append(f"PDF: {source_name}")
                            file_tracker.mark_file_processed(pdf_path, len(chunks))
                            logger.info(f"✅ Added {len(chunks)} chunks from {source_name}")
                        else:
                            logger.warning(f"⚠️ No chunks extracted from {source_name}")
                    except Exception as e:
                        logger.error(f"❌ Error processing {source_name}: {e}")
        else:
            logger.warning(f"No PDF files found in {PDF_DIRECTORY}")
    
    # Process CSV files with fallback logic
    if AUTO_LOAD_CSVS:
        logger.info(f"📊 Processing CSV files from: {CSV_DIRECTORY}")
        csv_files = get_csv_files_from_directory(CSV_DIRECTORY)
        
        # If no CSV files found in main directory and fallback is enabled, try fallback directory
        if not csv_files and USE_FALLBACK_CSV:
            logger.info(f"No CSV files found in main directory, trying fallback: {CSV_FALLBACK_DIRECTORY}")
            csv_files = get_csv_files_from_directory(CSV_FALLBACK_DIRECTORY)
            if csv_files:
                logger.info(f"✅ Found {len(csv_files)} CSV files in fallback directory")
        
        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files")
            
            # Get unprocessed files
            if ENABLE_INCREMENTAL:
                unprocessed_csvs = file_tracker.get_unprocessed_files(list(csv_files.values()))
                logger.info(f"Processing {len(unprocessed_csvs)} new/modified CSV files")
            else:
                unprocessed_csvs = list(csv_files.values())
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file = {}
                for source_name, csv_path in csv_files.items():
                    if csv_path in unprocessed_csvs:
                        future = executor.submit(process_file_parallel, csv_path, 'csv', source_name)
                        future_to_file[future] = (source_name, csv_path)
                
                for future in as_completed(future_to_file):
                    source_name, csv_path = future_to_file[future]
                    try:
                        chunks = future.result()
                        if chunks:
                            all_chunks.extend(chunks)
                            processed_files.append(f"CSV: {source_name}")
                            file_tracker.mark_file_processed(csv_path, len(chunks))
                            logger.info(f"✅ Added {len(chunks)} chunks from {source_name}")
                        else:
                            logger.warning(f"⚠️ No chunks extracted from {source_name}")
                    except Exception as e:
                        logger.error(f"❌ Error processing {source_name}: {e}")
        else:
            logger.warning(f"No CSV files found in either {CSV_DIRECTORY} or {CSV_FALLBACK_DIRECTORY}")
    
    # Add chunks to knowledge base
    if all_chunks:
        logger.info(f"Adding {len(all_chunks)} chunks to knowledge base...")
        success = add_to_knowledge_base_enhanced(all_chunks)
        if success:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"✅ Successfully added {len(all_chunks)} total chunks to knowledge base")
            logger.info(f"Processed files: {', '.join(processed_files)}")
            logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
        else:
            logger.error("❌ Failed to add chunks to knowledge base")
    else:
        logger.info("ℹ️ No new files to process (all files are up to date)")

# Background processing task
async def process_files_background():
    """Background task for processing files"""
    while True:
        try:
            initialize_knowledge_base_enhanced()
            await asyncio.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logger.error(f"Error in background processing: {e}")
            await asyncio.sleep(60)

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/rf-ai-agent", response_class=HTMLResponse)
async def rf_ai_agent(request: Request):
    return templates.TemplateResponse("rf_ai_agent.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    """Enhanced chat with improved RAG implementation"""
    try:
        start_time = time.time()
        
        # Search knowledge base for relevant information
        relevant_docs = search_knowledge_base_enhanced(message, top_k=8)
        
        # Create enhanced context from relevant documents
        context = ""
        if relevant_docs:
            context_parts = []
            for doc in relevant_docs:
                source_info = f"[Source: {doc['metadata'].get('source', 'unknown')}"
                if doc['metadata'].get('file_type') == 'csv':
                    source_info += f", Type: {doc['metadata'].get('data_type', 'data')}"
                elif doc['metadata'].get('file_type') == 'pdf':
                    source_info += f", Page: {doc['metadata'].get('page', 'unknown')}"
                source_info += f", Relevance: {doc.get('relevance_score', 0):.3f}]"
                context_parts.append(f"{doc['text']}\n{source_info}")
            
            context = "\n\n".join(context_parts)
            
            # Limit context length to prevent token overflow
            if len(context) > MAX_CONTEXT_LENGTH:
                context = context[:MAX_CONTEXT_LENGTH] + "... [truncated]"
            
            context = f"\n\nRelevant information from knowledge base:\n{context}"
        
        # Enhanced system prompt with better RAG instructions and formatting
        system_prompt = f"""You are a helpful AI assistant specialized in RF engineering and network performance analysis. 
        
        CRITICAL INSTRUCTIONS FOR RAG:
        1. ALWAYS use the provided knowledge base information when available
        2. Cite specific sources when referencing information from the knowledge base
        3. If the knowledge base contains relevant information, prioritize it over general knowledge
        4. Be specific and detailed when using knowledge base information
        5. If you don't find relevant information in the knowledge base, say so clearly
        
        RESPONSE FORMATTING INSTRUCTIONS:
        - Format your responses with proper HTML markup for better readability
        - Use <h3> tags for section headers (e.g., <h3>Key Points</h3>)
        - Use <ul> and <li> tags for bullet points and lists
        - Use <strong> tags for important terms and concepts
        - Use <em> tags for emphasis
        - Use <p> tags to separate paragraphs
        - Use <code> tags for technical terms, parameters, or code snippets
        - Use <blockquote> tags when citing specific information from sources
        - Structure complex responses with clear sections and subsections
        - Make your responses visually appealing and easy to scan
        
        You have access to additional knowledge from multiple PDF documents and CSV files containing KPI metrics and network performance data.
        When using this knowledge, make sure to integrate it naturally into your responses and cite the sources.
        
        For network performance questions, focus on KPI metrics, signal quality, coverage analysis, and engineering insights.
        Be ready to discuss RF engineering concepts, network optimization, and performance evaluation.
        
        Additional knowledge context:{context}"""
        
        # Send message to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,  # Increased for more detailed responses
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.3  # Lower temperature for more focused responses
        )
        
        response_time = time.time() - start_time
        
        # Log the interaction for analysis
        logger.info(f"Query: {message[:100]}... | Docs found: {len(relevant_docs)} | Response time: {response_time:.2f}s")
        
        return {
            "response": response.choices[0].message.content,
            "rag_metrics": {
                "documents_retrieved": len(relevant_docs),
                "average_relevance": round(sum(doc.get('relevance_score', 0) for doc in relevant_docs) / len(relevant_docs), 3) if relevant_docs else 0,
                "response_time_seconds": round(response_time, 2)
            }
        }
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return {"response": f"Sorry, I'm having some technical difficulties: {str(e)}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "OpenAI GPT-4o (Enhanced RAG)", "timestamp": datetime.now().isoformat()}

@app.get("/knowledge-status")
async def knowledge_status():
    """Enhanced knowledge base status"""
    try:
        count = knowledge_collection.count()
        
        # Get detailed statistics
        if count > 0:
            results = knowledge_collection.get()
            sources = set()
            file_types = set()
            data_types = set()
            
            for metadata in results['metadatas']:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
                if metadata and 'file_type' in metadata:
                    file_types.add(metadata['file_type'])
                if metadata and 'data_type' in metadata:
                    data_types.add(metadata['data_type'])
            
            sources_list = list(sources)
            file_types_list = list(file_types)
            data_types_list = list(data_types)
        else:
            sources_list = []
            file_types_list = []
            data_types_list = []
        
        return {
            "status": "success",
            "knowledge_chunks": count,
            "sources": sources_list,
            "file_types": file_types_list,
            "data_types": data_types_list,
            "message": f"Enhanced knowledge base contains {count} chunks from {len(sources_list)} sources",
            "rag_config": {
                "embedding_model": EMBEDDING_MODEL,
                "min_relevance_score": MIN_RELEVANCE_SCORE,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "enable_reranking": ENABLE_RERANKING
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/file-sources")
async def get_file_sources():
    """Get enhanced file sources information"""
    try:
        sources_info = {
            "pdf_directory": PDF_DIRECTORY if AUTO_LOAD_PDFS else None,
            "csv_directory": CSV_DIRECTORY if AUTO_LOAD_CSVS else None,
            "cache_directory": CACHE_DIRECTORY,
            "auto_load_pdfs": AUTO_LOAD_PDFS,
            "auto_load_csvs": AUTO_LOAD_CSVS,
            "enable_caching": ENABLE_CACHING,
            "enable_incremental": ENABLE_INCREMENTAL,
            "max_workers": MAX_WORKERS,
            "batch_size": BATCH_SIZE,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
        
        if AUTO_LOAD_PDFS:
            pdf_files = get_pdf_files_from_directory(PDF_DIRECTORY)
            sources_info["pdf_files"] = pdf_files
            sources_info["pdf_count"] = len(pdf_files)
        
        if AUTO_LOAD_CSVS:
            csv_files = get_csv_files_from_directory(CSV_DIRECTORY)
            sources_info["csv_files"] = csv_files
            sources_info["csv_count"] = len(csv_files)
        
        return sources_info
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/refresh-knowledge-base")
async def refresh_knowledge_base(background_tasks: BackgroundTasks):
    """Manually refresh the knowledge base"""
    try:
        background_tasks.add_task(initialize_knowledge_base_enhanced)
        return {"status": "success", "message": "Enhanced knowledge base refresh started in background"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# RAG Evaluation Endpoints
@app.get("/rag-evaluation")
async def rag_evaluation_dashboard():
    """Get comprehensive RAG evaluation metrics"""
    try:
        kb_stats = await knowledge_status()
        
        embedding_stats = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": 1536,
            "total_embeddings": knowledge_collection.count(),
            "embedding_quality": "High (OpenAI ada-002)"
        }
        
        return {
            "status": "success",
            "knowledge_base": kb_stats,
            "embedding_performance": embedding_stats,
            "evaluation_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/test-rag-query")
async def test_rag_query(query: str = Form(...), top_k: int = Form(8)):
    """Test RAG query and return detailed analysis"""
    try:
        start_time = time.time()
        relevant_docs = search_knowledge_base_enhanced(query, top_k)
        search_time = time.time() - start_time
        
        analysis = {
            "query": query,
            "search_time_seconds": round(search_time, 3),
            "documents_found": len(relevant_docs),
            "average_relevance_score": 0,
            "source_distribution": {},
            "top_sources": []
        }
        
        if relevant_docs:
            relevance_scores = [doc.get('relevance_score', 0) for doc in relevant_docs]
            analysis["average_relevance_score"] = round(sum(relevance_scores) / len(relevance_scores), 3)
            
            sources = {}
            for doc in relevant_docs:
                source = doc['metadata'].get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            analysis["source_distribution"] = sources
        
        return {"status": "success", "analysis": analysis, "relevant_documents": relevant_docs}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Maintenance page routes
@app.get("/network-analyzer", response_class=HTMLResponse)
async def network_analyzer(request: Request):
    return templates.TemplateResponse("network_analyzer.html", {"request": request})

@app.get("/coverage-planner", response_class=HTMLResponse)
async def coverage_planner(request: Request):
    return templates.TemplateResponse("coverage_planner.html", {"request": request})

@app.get("/kpi-dashboard", response_class=HTMLResponse)
async def kpi_dashboard(request: Request):
    return templates.TemplateResponse("kpi_dashboard.html", {"request": request})

@app.get("/troubleshooting-guide", response_class=HTMLResponse)
async def troubleshooting_guide(request: Request):
    return templates.TemplateResponse("troubleshooting_guide.html", {"request": request})

@app.get("/documentation-hub", response_class=HTMLResponse)
async def documentation_hub(request: Request):
    return templates.TemplateResponse("documentation_hub.html", {"request": request})

@app.get("/parameter-search", response_class=HTMLResponse)
async def parameter_search(request: Request):
    """Parameter Search page"""
    return templates.TemplateResponse("parameter_search.html", {"request": request})

@app.get("/parameter-audit", response_class=HTMLResponse)
async def parameter_audit(request: Request):
    """Parameter Audit page"""
    return templates.TemplateResponse("parameter_audit.html", {"request": request})

@app.get("/knowledge-base", response_class=HTMLResponse)
async def knowledge_base(request: Request):
    """Knowledge Base page"""
    return templates.TemplateResponse("knowledge_base.html", {"request": request})

@app.get("/technical-specs", response_class=HTMLResponse)
async def technical_specs(request: Request):
    """Technical Specifications page"""
    return templates.TemplateResponse("technical_specs.html", {"request": request})

@app.get("/best-practices", response_class=HTMLResponse)
async def best_practices(request: Request):
    """Best Practices page"""
    return templates.TemplateResponse("best_practices.html", {"request": request})

@app.get("/training-resources", response_class=HTMLResponse)
async def training_resources(request: Request):
    """Training Resources page"""
    return templates.TemplateResponse("training_resources.html", {"request": request})

@app.get("/alarm-checker", response_class=HTMLResponse)
def alarm_checker(request: Request):
    return templates.TemplateResponse("alarm_checker.html", {"request": request})

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize enhanced knowledge base on startup"""
    initialize_knowledge_base_enhanced()

# LTE CSV path and cache
lte_csv_path = 'csv_files/Parameters_LTE.csv'
lte_csv_cache = None
lte_csv_lock = threading.Lock()

def get_lte_csv():
    global lte_csv_cache
    with lte_csv_lock:
        if lte_csv_cache is None:
            try:
                # Try different encodings and clean column names
                lte_csv_cache = pd.read_csv(lte_csv_path, dtype=str, encoding='latin1')
                
                # Clean column names - remove extra spaces and normalize
                lte_csv_cache.columns = lte_csv_cache.columns.str.strip()
                
                # Handle common column name variations
                column_mapping = {
                    'abbreviated name': 'Abbreviated name',
                    'abbreviated_name': 'Abbreviated name',
                    'managed object': 'Managed object',
                    'managed_object': 'Managed object',
                    'managedobject': 'Managed object'
                }
                
                # Apply column mapping
                for old_name, new_name in column_mapping.items():
                    if old_name in lte_csv_cache.columns:
                        lte_csv_cache = lte_csv_cache.rename(columns={old_name: new_name})
                
                # Fill NaN values
                lte_csv_cache = lte_csv_cache.fillna("")
                
                print(f"[DEBUG] LTE CSV loaded successfully. Columns: {list(lte_csv_cache.columns)}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load LTE CSV: {e}")
                # Return empty DataFrame with expected columns
                lte_csv_cache = pd.DataFrame(columns=['Abbreviated name', 'Managed object'])
        
        return lte_csv_cache

# NR CSV path and cache
nr_csv_path = 'csv_files/Parameters_NR.csv'
nr_csv_cache = None
nr_csv_lock = threading.Lock()

def get_nr_csv():
    global nr_csv_cache
    with nr_csv_lock:
        if nr_csv_cache is None:
            try:
                # Try different encodings and clean column names
                nr_csv_cache = pd.read_csv(nr_csv_path, dtype=str, encoding='latin1')
                
                # Clean column names - remove extra spaces and normalize
                nr_csv_cache.columns = nr_csv_cache.columns.str.strip()
                
                # Handle common column name variations
                column_mapping = {
                    'abbreviated name': 'Abbreviated name',
                    'abbreviated_name': 'Abbreviated name',
                    'managed object': 'Managed object',
                    'managed_object': 'Managed object',
                    'managedobject': 'Managed object'
                }
                
                # Apply column mapping
                for old_name, new_name in column_mapping.items():
                    if old_name in nr_csv_cache.columns:
                        nr_csv_cache = nr_csv_cache.rename(columns={old_name: new_name})
                
                # Fill NaN values
                nr_csv_cache = nr_csv_cache.fillna("")
                
                print(f"[DEBUG] NR CSV loaded successfully. Columns: {list(nr_csv_cache.columns)}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load NR CSV: {e}")
                # Return empty DataFrame with expected columns
                nr_csv_cache = pd.DataFrame(columns=['Abbreviated name', 'Managed object'])
        
        return nr_csv_cache

# Autocomplete endpoints for abbreviated names
@app.get('/api/ltepar-abbreviated-autocomplete')
def api_ltepar_abbreviated_autocomplete(query: str = ""):
    """Get abbreviated names for LTE parameters matching the query"""
    df = get_lte_csv()
    if query:
        mask = df['Abbreviated name'].str.lower().str.contains(query.strip().lower(), na=False)
        names = sorted(df.loc[mask, 'Abbreviated name'].dropna().unique())
    else:
        names = sorted(df['Abbreviated name'].dropna().unique())
    return JSONResponse({'abbreviated_names': names})

@app.get('/api/nrpar-abbreviated-autocomplete')
def api_nrpar_abbreviated_autocomplete(query: str = ""):
    """Get abbreviated names for NR parameters matching the query"""
    df = get_nr_csv()
    if query:
        mask = df['Abbreviated name'].str.lower().str.contains(query.strip().lower(), na=False)
        names = sorted(df.loc[mask, 'Abbreviated name'].dropna().unique())
    else:
        names = sorted(df['Abbreviated name'].dropna().unique())
    return JSONResponse({'abbreviated_names': names})

# Update the LTE endpoints with better error handling
@app.get('/api/ltepar-search-managed-objects')
def api_ltepar_search_managed_objects(abbreviated_name: str = ""):
    try:
        df = get_lte_csv()
        print(f"[DEBUG] LTE CSV columns: {list(df.columns)}")
        print(f"[DEBUG] Looking for abbreviated_name: '{abbreviated_name}'")
        
        if 'Abbreviated name' not in df.columns:
            return JSONResponse({"error": f"Column 'Abbreviated name' not found. Available columns: {list(df.columns)}"})
        
        if 'Managed object' not in df.columns:
            return JSONResponse({"error": f"Column 'Managed object' not found. Available columns: {list(df.columns)}"})
        
        if abbreviated_name:
            filtered = df[df['Abbreviated name'].str.strip().str.lower() == abbreviated_name.strip().lower()]
            print(f"[DEBUG] Found {len(filtered)} matching rows")
        else:
            filtered = df
        
        managed_objects = sorted(filtered['Managed object'].dropna().unique())
        print(f"[DEBUG] Returning {len(managed_objects)} managed objects")
        
        return JSONResponse({ 'managed_objects': managed_objects })
    except Exception as e:
        print(f"[ERROR] LTE search error: {str(e)}")
        return JSONResponse({"error": f"LTE search error: {str(e)}"})

@app.get('/api/ltepar-parameter-details')
def api_ltepar_parameter_details(abbreviated_name: str = "", managed_object: str = ""):
    try:
        df = get_lte_csv()
        print(f"[DEBUG] LTE parameter details - abbreviated_name: '{abbreviated_name}', managed_object: '{managed_object}'")
        
        if 'Abbreviated name' not in df.columns:
            return JSONResponse({"error": f"Column 'Abbreviated name' not found. Available columns: {list(df.columns)}"})
        
        if 'Managed object' not in df.columns:
            return JSONResponse({"error": f"Column 'Managed object' not found. Available columns: {list(df.columns)}"})
        
        filtered = df
        if abbreviated_name:
            filtered = filtered[filtered['Abbreviated name'].str.strip().str.lower() == abbreviated_name.strip().lower()]
        if managed_object:
            filtered = filtered[filtered['Managed object'].str.strip() == managed_object.strip()]
        
        print(f"[DEBUG] Found {len(filtered)} matching rows")
        details = filtered.fillna("").to_dict(orient='records')
        
        return JSONResponse({ 'details': details })
    except Exception as e:
        print(f"[ERROR] LTE parameter details error: {str(e)}")
        return JSONResponse({"error": f"LTE parameter details error: {str(e)}"})

@app.get('/api/nrpar-search-managed-objects')
def api_nrpar_search_managed_objects(abbreviated_name: str = ""):
    df = get_nr_csv()
    if abbreviated_name:
        filtered = df[df['Abbreviated name'].str.strip().str.lower() == abbreviated_name.strip().lower()]
    else:
        filtered = df
    managed_objects = sorted(filtered['Managed object'].dropna().unique())
    return JSONResponse({ 'managed_objects': managed_objects })

@app.get('/api/nrpar-parameter-details')
def api_nrpar_parameter_details(abbreviated_name: str = "", managed_object: str = ""):
    df = get_nr_csv()
    filtered = df
    if abbreviated_name:
        filtered = filtered[filtered['Abbreviated name'].str.strip().str.lower() == abbreviated_name.strip().lower()]
    if managed_object:
        filtered = filtered[filtered['Managed object'].str.strip() == managed_object.strip()]
    details = filtered.fillna("").to_dict(orient='records')
    return JSONResponse({ 'details': details })

@app.get("/test-chat")
async def test_chat():
    """Simple test endpoint to verify chat functionality"""
    try:
        kb_count = knowledge_collection.count()
        return {
            "status": "success",
            "message": "Chat endpoint is working!",
            "timestamp": datetime.now().isoformat(),
            "knowledge_chunks": kb_count,
            "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "openai_key_length": len(os.getenv("OPENAI_API_KEY", ""))
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/simple-chat")
async def simple_chat(message: str = Form(...)):
    """Simple chat without RAG for testing"""
    try:
        # Simple response without knowledge base
        system_prompt = """You are a helpful AI assistant specialized in RF engineering and network performance analysis. 
        You can help with general RF engineering questions and network optimization concepts."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.3
        )
        
        return {
            "response": response.choices[0].message.content,
            "rag_metrics": {
                "documents_retrieved": 0,
                "average_relevance": 0,
                "response_time_seconds": 0
            }
        }
    
    except Exception as e:
        logger.error(f"Error in simple chat: {e}")
        return {"response": f"Sorry, I'm having some technical difficulties: {str(e)}"}

# Add this diagnostic endpoint to check CSV structure
@app.get('/api/debug-csv-structure')
def debug_csv_structure():
    """Debug endpoint to check CSV column names and structure"""
    try:
        lte_df = get_lte_csv()
        nr_df = get_nr_csv()
        
        return {
            "lte_csv": {
                "columns": list(lte_df.columns),
                "shape": lte_df.shape,
                "sample_abbreviated_names": lte_df['Abbreviated name'].dropna().head(5).tolist() if 'Abbreviated name' in lte_df.columns else "Column not found",
                "sample_managed_objects": lte_df['Managed object'].dropna().head(5).tolist() if 'Managed object' in lte_df.columns else "Column not found",
                "total_abbreviated_names": len(lte_df['Abbreviated name'].dropna().unique()) if 'Abbreviated name' in lte_df.columns else 0,
                "total_managed_objects": len(lte_df['Managed object'].dropna().unique()) if 'Managed object' in lte_df.columns else 0
            },
            "nr_csv": {
                "columns": list(nr_df.columns),
                "shape": nr_df.shape,
                "sample_abbreviated_names": nr_df['Abbreviated name'].dropna().head(5).tolist() if 'Abbreviated name' in nr_df.columns else "Column not found",
                "sample_managed_objects": nr_df['Managed object'].dropna().head(5).tolist() if 'Managed object' in nr_df.columns else "Column not found",
                "total_abbreviated_names": len(nr_df['Abbreviated name'].dropna().unique()) if 'Abbreviated name' in nr_df.columns else 0,
                "total_managed_objects": len(nr_df['Managed object'].dropna().unique()) if 'Managed object' in nr_df.columns else 0
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get('/api/test-lte-search')
def test_lte_search(abbreviated_name: str = ""):
    """Test endpoint to debug LTE search issues"""
    try:
        df = get_lte_csv()
        
        # Check if CSV loaded properly
        if df.empty:
            return JSONResponse({"error": "LTE CSV is empty or failed to load"})
        
        # Check column names
        columns = list(df.columns)
        if 'Abbreviated name' not in columns:
            return JSONResponse({"error": f"'Abbreviated name' column not found. Available columns: {columns}"})
        
        if 'Managed object' not in columns:
            return JSONResponse({"error": f"'Managed object' column not found. Available columns: {columns}"})
        
        # Check if there's data
        total_rows = len(df)
        unique_abbr = df['Abbreviated name'].dropna().nunique()
        unique_mo = df['Managed object'].dropna().nunique()
        
        # Test search if abbreviated_name provided
        if abbreviated_name:
            filtered = df[df['Abbreviated name'].str.strip().str.lower() == abbreviated_name.strip().lower()]
            found_rows = len(filtered)
            found_mo = filtered['Managed object'].dropna().unique().tolist()
        else:
            found_rows = 0
            found_mo = []
        
        return JSONResponse({
            "status": "success",
            "csv_info": {
                "total_rows": total_rows,
                "unique_abbreviated_names": unique_abbr,
                "unique_managed_objects": unique_mo,
                "columns": columns
            },
            "search_test": {
                "search_term": abbreviated_name,
                "found_rows": found_rows,
                "found_managed_objects": found_mo
            }
        })
        
    except Exception as e:
        return JSONResponse({"error": f"Test failed: {str(e)}"})

# Add this endpoint to clear LTE CSV cache
@app.get('/api/clear-lte-cache')
def clear_lte_cache():
    """Clear LTE CSV cache to force reload"""
    global lte_csv_cache
    lte_csv_cache = None
    return JSONResponse({"message": "LTE CSV cache cleared successfully"})

# Add this simple debug endpoint
@app.get('/api/debug-lte-csv')
def debug_lte_csv():
    """Simple debug endpoint to check LTE CSV"""
    try:
        df = get_lte_csv()
        return {
            "success": True,
            "rows": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(3).to_dict('records') if not df.empty else []
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/houston-rf-team", response_class=HTMLResponse)
async def houston_rf_team(request: Request):
    return templates.TemplateResponse("houston_rf_team.html", {"request": request})

@app.get("/test-images", response_class=HTMLResponse)
async def test_images(request: Request):
    with open('test_images.html', 'r') as f:
        return HTMLResponse(content=f.read())

@app.get("/debug-team-page", response_class=HTMLResponse)
async def debug_team_page(request: Request):
    return templates.TemplateResponse("houston_rf_team.html", {"request": request})

@app.get("/debug-images", response_class=HTMLResponse)
async def debug_images(request: Request):
    return templates.TemplateResponse("debug_images.html", {"request": request})

@app.get("/simple-team-test", response_class=HTMLResponse)
async def simple_team_test(request: Request):
    return templates.TemplateResponse("simple_team_test.html", {"request": request})

@app.get("/parameter-query", response_class=HTMLResponse)
async def parameter_query(request: Request):
    return templates.TemplateResponse("parameter_query.html", {"request": request})

@app.get('/api/houston-site-names')
def get_houston_site_names():
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv_files', 'Houston_Sites_OSS_IP.csv')
        df = pd.read_csv(csv_path, dtype=str, encoding='utf-8').fillna("")
        site_names = sorted(df['Site Name'].unique())
        return JSONResponse(site_names)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get('/api/houston-site-details')
def get_houston_site_details(site_name: str = ""):
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv_files', 'Houston_Sites_OSS_IP.csv')
        df = pd.read_csv(csv_path, dtype=str, encoding='utf-8').fillna("")
        filtered = df[df['Site Name'].str.strip() == site_name.strip()]
        result = filtered[['MRBTSID', 'local Ip Address', 'OSS Name', 'oss IP address']].to_dict(orient='records')
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def get_oracle_connection(oss_ip_address):
    """Create Oracle database connection"""
    try:
        connection_string = f"pmagno7/TMobile015@{oss_ip_address}:1521/orcl"
        connection = oracledb.connect(connection_string)
        return connection
    except Exception as e:
        print(f"Oracle connection error: {e}")
        return None

@app.get('/api/fetch-alarms')
def fetch_alarms(site_name: str = "", time_range: str = "24h", oss_ip_address: str = ""):
    try:
        print(site_name,time_range, oss_ip_address)
        if not site_name or not oss_ip_address:
            return JSONResponse({"error": "Site name and OSS IP address are required"}, status_code=400)
        
        # Calculate start time based on time range
        now = datetime.now()
        if time_range == "24h":
            start_time = now - timedelta(hours=24)
        elif time_range == "7d":
            start_time = now - timedelta(days=7)
        elif time_range == "14d":
            start_time = now - timedelta(days=14)
        else:
            start_time = now - timedelta(hours=24)  # default to 24 hours
        
        # Format start time for Oracle
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create site filter with % pattern
        site_filter = f"%{site_name}%"
        
        query = """
        SELECT
            SUBSTR(co.co_name, 2, 8) AS SiteID,
            co.co_name AS Site_Cell,
            co.co_dn,
            DECODE(
                a.SEVERITY,
                '1', 'CRITICAL',
                '2', 'MAJOR',
                '3', 'MINOR',
                'UNKNOWN'
            ) AS ALARM_SEVERITY,
            TO_CHAR(a.alarm_time, 'MM-DD-YYYY HH24:MI:SS') AS ALARM_TIME,
            a.alarm_number,
            a.alarm_status,
            a.text,
            a.supplementary_info
        FROM
            fx_alarm a
            JOIN ctp_common_objects co ON a.ne_gid = co.co_gid
        WHERE
            a.alarm_number NOT IN ('9249')
            AND a.alarm_time >= :start_time
            AND (:site_filter IS NULL OR co.co_name LIKE :site_filter)
        """
        
        # Connect to Oracle and execute query
        connection = get_oracle_connection(oss_ip_address)
        if not connection:
            return JSONResponse({"error": "Failed to connect to Oracle database"}, status_code=500)
        
        cursor = connection.cursor()
        cursor.execute(query, {
            'start_time': start_time_str,
            'site_filter': site_filter
        })
        
        # Fetch results
        columns = [col[0] for col in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        cursor.close()
        connection.close()
        
        return JSONResponse(results)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Parameter Query API Endpoints - Independent from Alarm Checker
@app.get('/api/parameter-site-names')
def get_parameter_site_names():
    """Get site names for Parameter Query (uses same CSV as Alarm Checker)"""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_files", "Houston_Sites_OSS_IP.csv")
        if not os.path.exists(csv_path):
            return {"success": False, "error": "Site CSV file not found", "sites": []}
        
        sites = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('Site Name') and row.get('oss IP address'):
                    sites.append({
                        'site_name': row['Site Name'].strip(),
                        'oss_ip_address': row['oss IP address'].strip()
                    })
        
        logger.info(f"Parameter Query: Loaded {len(sites)} sites from CSV")
        return {"success": True, "sites": sites}
    except Exception as e:
        logger.error(f"Parameter Query: Error loading site names: {e}")
        return {"success": False, "error": str(e), "sites": []}

@app.get('/api/parameter-managed-objects')
def get_parameter_managed_objects():
    """Get managed objects from the CSV mapping file"""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_files", "mo_query.csv")
        if not os.path.exists(csv_path):
            return {"success": False, "error": "Managed Object CSV file not found", "managed_objects": []}
        
        managed_objects = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('managed_object') and row.get('sql_query'):
                    managed_objects.append({
                        'managed_object': row['managed_object'].strip(),
                        'query_description': row.get('query_description', '').strip(),
                        'sql_query': row['sql_query'].strip()
                    })
        
        logger.info(f"Parameter Query: Loaded {len(managed_objects)} managed objects from CSV")
        return {"success": True, "managed_objects": managed_objects}
    except Exception as e:
        logger.error(f"Parameter Query: Error loading managed objects: {e}")
        return {"success": False, "error": str(e), "managed_objects": []}

@app.post('/api/fetch-parameter-data')
async def fetch_parameter_data(request: Request):
    """Fetch parameter data from Oracle based on managed object selection"""
    try:
        body = await request.json()
        site_name = body.get('site_name', '')
        managed_object = body.get('managed_object', '')
        oss_ip_address = body.get('oss_ip_address', '')
        
        if not site_name or not managed_object or not oss_ip_address:
            return {"success": False, "error": "Missing required parameters"}
        
        # Get the SQL query for the selected managed object
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_files", "mo_query.csv")
        if not os.path.exists(csv_path):
            return {"success": False, "error": "Managed Object CSV file not found"}
        
        sql_query = None
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('managed_object', '').strip() == managed_object:
                    sql_query = row.get('sql_query', '').strip()
                    break
        
        if not sql_query:
            return {"success": False, "error": f"No SQL query found for managed object: {managed_object}"}
        
        # Execute the query with the site name parameter
        try:
            connection = get_oracle_connection(oss_ip_address)
            if not connection:
                return {"success": False, "error": "Failed to connect to Oracle database"}
            
            cursor = connection.cursor()
            
            # Replace :site_name placeholder with actual site name
            sql_query = sql_query.replace(':site_name', f"'{site_name}'")
            
            logger.info(f"Parameter Query: Executing query for {managed_object} at {site_name}")
            cursor.execute(sql_query)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                data.append(dict(zip(columns, row)))
            
            cursor.close()
            connection.close()
            
            logger.info(f"Parameter Query: Retrieved {len(data)} records for {managed_object}")
            return {"success": True, "data": data}
            
        except Exception as db_error:
            logger.error(f"Parameter Query: Database error: {db_error}")
            return {"success": False, "error": f"Database error: {str(db_error)}"}
            
    except Exception as e:
        logger.error(f"Parameter Query: Error fetching parameter data: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 