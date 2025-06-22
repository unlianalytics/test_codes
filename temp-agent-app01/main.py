from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import anthropic
import os
from dotenv import load_dotenv
import json
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Set
import re
import glob
import pandas as pd
from datetime import datetime
import warnings
import hashlib
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

app = FastAPI(title="T-Mobile RF AI Agent", description="Optimized AI agent with T-Mobile branding and RAG capabilities for RF Engineering")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
)

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB for vector storage
chroma_client = chromadb.Client()
collection_name = "tmobile_rf_knowledge"
try:
    knowledge_collection = chroma_client.get_collection(collection_name)
except:
    knowledge_collection = chroma_client.create_collection(collection_name)

# ===== CONFIGURATION SECTION =====
# File Discovery Configuration
PDF_DIRECTORY = r"C:\Users\magno\Downloads\pdf_files"  # Directory containing PDFs
CSV_DIRECTORY = r"C:\Users\magno\Downloads\csv_metrics"  # Directory containing CSV metrics
CACHE_DIRECTORY = r"C:\Users\magno\Downloads\agent_cache"  # Cache directory for processed data
AUTO_LOAD_PDFS = True  # Set to True for automatic PDF discovery
AUTO_LOAD_CSVS = True  # Set to True for automatic CSV discovery

# Performance Configuration
MAX_WORKERS = 4          # Parallel workers (increase for more cores)
BATCH_SIZE = 100         # Chunk batch size
CHUNK_SIZE = 800         # Optimized chunk size
ENABLE_CACHING = True    # Enable file processing cache
ENABLE_INCREMENTAL = True # Enable incremental processing

# ===== END CONFIGURATION =====

# File tracking for incremental processing
class FileTracker:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tracker_file = self.cache_dir / "file_tracker.pkl"
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

# Initialize file tracker
file_tracker = FileTracker(CACHE_DIRECTORY)

# Optimized PDF processing function
def process_pdf_optimized(pdf_path: str, source_name: str = "unknown") -> List[Dict]:
    """Extract text from PDF with optimized chunking"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_chunks = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Use optimized chunk size
                    chunks = split_text_into_chunks_optimized(text, CHUNK_SIZE)
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            text_chunks.append({
                                'text': chunk.strip(),
                                'page': page_num + 1,
                                'chunk_id': f"{source_name}_page_{page_num + 1}_chunk_{i + 1}",
                                'source': source_name,
                                'file_path': pdf_path,
                                'file_type': 'pdf'
                            })
            
            return text_chunks
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return []

# Optimized CSV processing function
def process_csv_optimized(csv_path: str, source_name: str = "unknown") -> List[Dict]:
    """Process CSV files with optimized analysis"""
    try:
        # Read CSV file with optimized settings
        df = pd.read_csv(csv_path, nrows=10000)  # Limit rows for large files
        
        # Get basic information about the dataset
        rows, cols = df.shape
        columns = list(df.columns)
        
        # Create summary information
        summary_info = f"Dataset: {source_name}\n"
        summary_info += f"Dimensions: {rows} rows × {cols} columns\n"
        summary_info += f"Columns: {', '.join(columns)}\n"
        
        # Get data types
        dtypes_info = "Data Types:\n"
        for col, dtype in df.dtypes.items():
            dtypes_info += f"  - {col}: {dtype}\n"
        
        # Get basic statistics for numeric columns (optimized)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_info = "Numeric Column Statistics:\n"
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                stats = df[col].describe()
                stats_info += f"  - {col}:\n"
                stats_info += f"    Mean: {stats['mean']:.4f}\n"
                stats_info += f"    Std: {stats['std']:.4f}\n"
                stats_info += f"    Min: {stats['min']:.4f}\n"
                stats_info += f"    Max: {stats['max']:.4f}\n"
        else:
            stats_info = "No numeric columns found for statistical analysis.\n"
        
        # Get sample data (optimized)
        sample_data = "Sample Data (first 3 rows):\n"
        sample_data += df.head(3).to_string(index=False)
        
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
        
        # Add specific KPI analysis if columns suggest network metrics
        kpi_analysis = analyze_network_kpis_optimized(df, source_name)
        if kpi_analysis:
            chunks.append({
                'text': kpi_analysis,
                'chunk_id': f"{source_name}_kpi_analysis",
                'source': source_name,
                'file_path': csv_path,
                'file_type': 'csv',
                'data_type': 'kpi_analysis'
            })
        
        return chunks
        
    except Exception as e:
        print(f"Error processing CSV {csv_path}: {e}")
        return []

def analyze_network_kpis_optimized(df: pd.DataFrame, source_name: str) -> str:
    """Analyze network KPI metrics with optimized processing"""
    try:
        analysis = f"Network KPI Analysis for {source_name}:\n\n"
        
        # Common network KPI column patterns
        network_patterns = {
            'signal_strength': ['rssi', 'signal', 'strength', 'power', 'level'],
            'quality_metrics': ['sinr', 'snr', 'quality', 'ber', 'fer'],
            'throughput': ['throughput', 'data_rate', 'speed', 'capacity'],
            'coverage': ['coverage', 'area', 'distance', 'range'],
            'interference': ['interference', 'noise', 'crosstalk'],
            'availability': ['availability', 'uptime', 'downtime', 'reliability'],
            'latency': ['latency', 'delay', 'response_time', 'ping'],
            'handover': ['handover', 'handoff', 'mobility'],
            'users': ['users', 'subscribers', 'connections', 'sessions']
        }
        
        columns_lower = [col.lower() for col in df.columns]
        
        for metric_type, patterns in network_patterns.items():
            matching_cols = []
            for pattern in patterns:
                for col in df.columns:
                    if pattern in col.lower():
                        matching_cols.append(col)
            
            if matching_cols:
                analysis += f"{metric_type.replace('_', ' ').title()} Metrics:\n"
                for col in matching_cols[:3]:  # Limit to first 3 matching columns
                    if col in df.select_dtypes(include=[np.number]).columns:
                        stats = df[col].describe()
                        analysis += f"  - {col}:\n"
                        analysis += f"    Average: {stats['mean']:.4f}\n"
                        analysis += f"    Range: {stats['min']:.4f} to {stats['max']:.4f}\n"
                        analysis += f"    Std Dev: {stats['std']:.4f}\n"
                    else:
                        unique_vals = df[col].nunique()
                        analysis += f"  - {col}: {unique_vals} unique values\n"
                analysis += "\n"
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing network KPIs: {e}")
        return ""

def split_text_into_chunks_optimized(text: str, max_length: int) -> List[str]:
    """Split text into chunks with optimized algorithm"""
    # Use sentence boundaries for better chunking
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def add_to_knowledge_base_batch(chunks: List[Dict]):
    """Add text chunks to the vector database in batches"""
    try:
        if not chunks:
            return True
            
        # Process in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            
            texts = [chunk['text'] for chunk in batch]
            metadatas = [
                {
                    'chunk_id': chunk['chunk_id'],
                    'source': chunk['source'],
                    'file_path': chunk['file_path'],
                    'file_type': chunk.get('file_type', 'unknown'),
                    'data_type': chunk.get('data_type', 'general'),
                    'page': chunk.get('page', 0)
                } for chunk in batch
            ]
            ids = [chunk['chunk_id'] for chunk in batch]
            
            # Generate embeddings in batch
            embeddings = embedding_model.encode(texts).tolist()
            
            # Add to collection
            knowledge_collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        return True
    except Exception as e:
        print(f"Error adding to knowledge base: {e}")
        return False

def search_knowledge_base_optimized(query: str, top_k: int = 5) -> List[Dict]:
    """Search the knowledge base with optimized retrieval"""
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query]).tolist()
        
        # Search collection with optimized parameters
        results = knowledge_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Return documents with metadata and relevance scores
        if results['documents'] and results['metadatas']:
            documents_with_metadata = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                distance = results['distances'][0][i] if i < len(results['distances'][0]) else 0
                documents_with_metadata.append({
                    'text': doc,
                    'metadata': metadata,
                    'relevance_score': 1 - distance  # Convert distance to relevance score
                })
            return documents_with_metadata
        return []
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return []

def get_pdf_files_from_directory(directory: str) -> Dict[str, str]:
    """Get all PDF files from a directory"""
    pdf_files = {}
    try:
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return {}
        
        pdf_pattern = os.path.join(directory, "*.pdf")
        pdf_paths = glob.glob(pdf_pattern)
        
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            source_name = os.path.splitext(filename)[0]
            pdf_files[source_name] = pdf_path
            
        return pdf_files
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return {}

def get_csv_files_from_directory(directory: str) -> Dict[str, str]:
    """Get all CSV files from a directory"""
    csv_files = {}
    try:
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return {}
        
        csv_pattern = os.path.join(directory, "*.csv")
        csv_paths = glob.glob(csv_pattern)
        
        for csv_path in csv_paths:
            filename = os.path.basename(csv_path)
            source_name = os.path.splitext(filename)[0]
            csv_files[source_name] = csv_path
            
        return csv_files
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return {}

def process_file_parallel(file_path: str, file_type: str, source_name: str) -> List[Dict]:
    """Process a single file (for parallel processing)"""
    try:
        if file_type == 'pdf':
            return process_pdf_optimized(file_path, source_name)
        elif file_type == 'csv':
            return process_csv_optimized(file_path, source_name)
        else:
            return []
    except Exception as e:
        print(f"Error processing {file_type} file {file_path}: {e}")
        return []

# Initialize knowledge base with optimized processing
def initialize_knowledge_base():
    """Initialize the knowledge base with optimized file processing"""
    all_chunks = []
    processed_files = []
    
    start_time = time.time()
    
    # Process PDF files
    if AUTO_LOAD_PDFS:
        print(f"\n Scanning directory for PDF files: {PDF_DIRECTORY}")
        pdf_files = get_pdf_files_from_directory(PDF_DIRECTORY)
        
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF files")
            
            # Get unprocessed files
            if ENABLE_INCREMENTAL:
                unprocessed_pdfs = file_tracker.get_unprocessed_files(list(pdf_files.values()))
                print(f"Processing {len(unprocessed_pdfs)} new/modified PDF files")
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
                            print(f"  ✓ Added {len(chunks)} chunks from {source_name}")
                        else:
                            print(f"  ✗ No chunks extracted from {source_name}")
                    except Exception as e:
                        print(f"  ✗ Error processing {source_name}: {e}")
        else:
            print(f"No PDF files found in {PDF_DIRECTORY}")
    
    # Process CSV files
    if AUTO_LOAD_CSVS:
        print(f"\n Scanning directory for CSV files: {CSV_DIRECTORY}")
        csv_files = get_csv_files_from_directory(CSV_DIRECTORY)
        
        if csv_files:
            print(f"Found {len(csv_files)} CSV files")
            
            # Get unprocessed files
            if ENABLE_INCREMENTAL:
                unprocessed_csvs = file_tracker.get_unprocessed_files(list(csv_files.values()))
                print(f"Processing {len(unprocessed_csvs)} new/modified CSV files")
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
                            print(f"  ✓ Added {len(chunks)} chunks from {source_name}")
                        else:
                            print(f"  ✗ No chunks extracted from {source_name}")
                    except Exception as e:
                        print(f"  ✗ Error processing {source_name}: {e}")
        else:
            print(f"No CSV files found in {CSV_DIRECTORY}")
    
    if all_chunks:
        success = add_to_knowledge_base_batch(all_chunks)
        if success:
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"\n Successfully added {len(all_chunks)} total chunks to knowledge base")
            print(f" Processed files: {', '.join(processed_files)}")
            print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        else:
            print("\n❌ Failed to add chunks to knowledge base")
    else:
        print("\n✅ No new files to process (all files are up to date)")

# Background task for processing files
async def process_files_background():
    """Background task for processing files"""
    while True:
        try:
            initialize_knowledge_base()
            # Wait for 5 minutes before checking for new files
            await asyncio.sleep(300)
        except Exception as e:
            print(f"Error in background processing: {e}")
            await asyncio.sleep(60)

# Initialize knowledge base on startup
initialize_knowledge_base()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main landing page with app selection"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/rf-ai-agent", response_class=HTMLResponse)
async def rf_ai_agent(request: Request):
    """RF AI Agent page"""
    return templates.TemplateResponse("rf_ai_agent.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    """Handle chat messages with Claude using optimized RAG"""
    try:
        # Search knowledge base for relevant information
        relevant_docs = search_knowledge_base_optimized(message)
        
        # Create context from relevant documents
        context = ""
        if relevant_docs:
            context_parts = []
            for doc in relevant_docs:
                source_info = f"[Source: {doc['metadata'].get('source', 'unknown')}"
                if doc['metadata'].get('file_type') == 'csv':
                    source_info += f", Type: {doc['metadata'].get('data_type', 'data')}"
                elif doc['metadata'].get('file_type') == 'pdf':
                    source_info += f", Page: {doc['metadata'].get('page', 'unknown')}"
                source_info += f", Relevance: {doc.get('relevance_score', 0):.2f}]"
                context_parts.append(f"{doc['text']}\n{source_info}")
            
            context = "\n\n".join(context_parts)
            context = f"\n\nRelevant information from knowledge base:\n{context}"
        
        # Create a Texas-themed system prompt with RAG context
        system_prompt = f"""You are a helpful AI assistant with a Texas personality, specialized in RF engineering and network performance analysis. 
        You're friendly, knowledgeable, and occasionally use Texas expressions. 
        Keep responses conversational and helpful while maintaining the Texas spirit.
        
        You have access to additional knowledge from multiple PDF documents and CSV files containing KPI metrics and network performance data.
        When using this knowledge, make sure to integrate it naturally into your Texas-style responses.
        If the information comes from a specific source, you can mention it casually in your response.
        
        For network performance questions, focus on KPI metrics, signal quality, coverage analysis, and engineering insights.
        Be ready to discuss RF engineering concepts, network optimization, and performance evaluation.
        
        Additional knowledge context:{context}"""
        
        # Send message to Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )
        
        return {"response": response.content[0].text}
    
    except Exception as e:
        return {"response": f"Sorry partner, I'm having some technical difficulties: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Optimized Texas AI Agent with RAG for RF Engineering is running!"}

@app.get("/knowledge-status")
async def knowledge_status():
    """Check the status of the knowledge base"""
    try:
        count = knowledge_collection.count()
        
        # Get unique sources and file types
        if count > 0:
            results = knowledge_collection.get()
            sources = set()
            file_types = set()
            for metadata in results['metadatas']:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
                if metadata and 'file_type' in metadata:
                    file_types.add(metadata['file_type'])
            sources_list = list(sources)
            file_types_list = list(file_types)
        else:
            sources_list = []
            file_types_list = []
        
        return {
            "status": "success",
            "knowledge_chunks": count,
            "sources": sources_list,
            "file_types": file_types_list,
            "message": f"Knowledge base contains {count} chunks from {len(sources_list)} sources ({', '.join(file_types_list)})"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking knowledge base: {str(e)}"
        }

@app.get("/file-sources")
async def get_file_sources():
    """Get information about available file sources"""
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
            "batch_size": BATCH_SIZE
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
        return {
            "status": "error",
            "message": f"Error getting file sources: {str(e)}"
        }

@app.post("/refresh-knowledge-base")
async def refresh_knowledge_base(background_tasks: BackgroundTasks):
    """Manually refresh the knowledge base"""
    try:
        background_tasks.add_task(initialize_knowledge_base)
        return {
            "status": "success",
            "message": "Knowledge base refresh started in background"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error refreshing knowledge base: {str(e)}"
        }

@app.get("/network-analyzer", response_class=HTMLResponse)
async def network_analyzer(request: Request):
    """Network Analyzer maintenance page"""
    return templates.TemplateResponse("network_analyzer.html", {"request": request})

@app.get("/coverage-planner", response_class=HTMLResponse)
async def coverage_planner(request: Request):
    """Coverage Planner maintenance page"""
    return templates.TemplateResponse("coverage_planner.html", {"request": request})

@app.get("/kpi-dashboard", response_class=HTMLResponse)
async def kpi_dashboard(request: Request):
    """KPI Dashboard maintenance page"""
    return templates.TemplateResponse("kpi_dashboard.html", {"request": request})

@app.get("/troubleshooting-guide", response_class=HTMLResponse)
async def troubleshooting_guide(request: Request):
    """Troubleshooting Guide maintenance page"""
    return templates.TemplateResponse("troubleshooting_guide.html", {"request": request})

@app.get("/documentation-hub", response_class=HTMLResponse)
async def documentation_hub(request: Request):
    """Documentation Hub maintenance page"""
    return templates.TemplateResponse("documentation_hub.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)