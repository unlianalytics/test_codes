from fastapi import FastAPI, Request, Form
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
from typing import List, Dict
import re
import glob
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

app = FastAPI(title="Texas AI Agent", description="A simple AI agent with Texas flag theme and RAG capabilities for RF Engineering")

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
collection_name = "texas_ai_knowledge"
try:
    knowledge_collection = chroma_client.get_collection(collection_name)
except:
    knowledge_collection = chroma_client.create_collection(collection_name)

# ===== CONFIGURATION SECTION =====
# File Discovery Configuration
PDF_DIRECTORY = r"C:\Users\magno\Downloads\pdf_files"  # Directory containing PDFs
CSV_DIRECTORY = r"C:\Users\magno\Downloads\csv_metrics"  # Directory containing CSV metrics
AUTO_LOAD_PDFS = True  # Set to True for automatic PDF discovery
AUTO_LOAD_CSVS = True  # Set to True for automatic CSV discovery

# ===== END CONFIGURATION =====

# PDF processing function
def process_pdf(pdf_path: str, source_name: str = "unknown") -> List[Dict]:
    """Extract text from PDF and split into chunks"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_chunks = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Split text into smaller chunks (roughly 500 characters each)
                    chunks = split_text_into_chunks(text, 500)
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

# CSV processing function for KPI metrics
def process_csv(csv_path: str, source_name: str = "unknown") -> List[Dict]:
    """Process CSV files and extract KPI metrics information"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
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
        
        # Get basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_info = "Numeric Column Statistics:\n"
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                stats = df[col].describe()
                stats_info += f"  - {col}:\n"
                stats_info += f"    Mean: {stats['mean']:.4f}\n"
                stats_info += f"    Std: {stats['std']:.4f}\n"
                stats_info += f"    Min: {stats['min']:.4f}\n"
                stats_info += f"    Max: {stats['max']:.4f}\n"
        else:
            stats_info = "No numeric columns found for statistical analysis.\n"
        
        # Get sample data (first few rows)
        sample_data = "Sample Data (first 5 rows):\n"
        sample_data += df.head().to_string(index=False)
        
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
        kpi_analysis = analyze_network_kpis(df, source_name)
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

def analyze_network_kpis(df: pd.DataFrame, source_name: str) -> str:
    """Analyze network KPI metrics from the dataset"""
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
                for col in matching_cols:
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
        
        # Check for time-based data
        time_columns = [col for col in df.columns if any(time_word in col.lower() for time_word in ['time', 'date', 'timestamp'])]
        if time_columns:
            analysis += "Time-based Analysis:\n"
            for col in time_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        date_range = f"{df[col].min()} to {df[col].max()}"
                        analysis += f"  - {col}: {date_range}\n"
                except:
                    pass
            analysis += "\n"
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing network KPIs: {e}")
        return ""

def split_text_into_chunks(text: str, max_length: int) -> List[str]:
    """Split text into chunks of approximately max_length characters"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def add_to_knowledge_base(chunks: List[Dict]):
    """Add text chunks to the vector database"""
    try:
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [
            {
                'chunk_id': chunk['chunk_id'],
                'source': chunk['source'],
                'file_path': chunk['file_path'],
                'file_type': chunk.get('file_type', 'unknown'),
                'data_type': chunk.get('data_type', 'general'),
                'page': chunk.get('page', 0)
            } for chunk in chunks
        ]
        ids = [chunk['chunk_id'] for chunk in chunks]
        
        # Generate embeddings
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

def search_knowledge_base(query: str, top_k: int = 5) -> List[Dict]:
    """Search the knowledge base for relevant information"""
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query]).tolist()
        
        # Search collection
        results = knowledge_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Return documents with metadata
        if results['documents'] and results['metadatas']:
            documents_with_metadata = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                documents_with_metadata.append({
                    'text': doc,
                    'metadata': metadata
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

# Initialize knowledge base with PDFs and CSVs
def initialize_knowledge_base():
    """Initialize the knowledge base with PDF and CSV files"""
    all_chunks = []
    processed_files = []
    
    # Process PDF files
    if AUTO_LOAD_PDFS:
        print(f"\nScanning directory for PDF files: {PDF_DIRECTORY}")
        pdf_files = get_pdf_files_from_directory(PDF_DIRECTORY)
        
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF files:")
            for source_name, pdf_path in pdf_files.items():
                print(f"  - {source_name}: {pdf_path}")
            
            for source_name, pdf_path in pdf_files.items():
                print(f"\nProcessing PDF: {source_name}...")
                chunks = process_pdf(pdf_path, source_name)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files.append(f"PDF: {source_name}")
                    print(f"  ✓ Added {len(chunks)} chunks from {source_name}")
                else:
                    print(f"  ✗ No chunks extracted from {source_name}")
        else:
            print(f"No PDF files found in {PDF_DIRECTORY}")
    
    # Process CSV files
    if AUTO_LOAD_CSVS:
        print(f"\nScanning directory for CSV files: {CSV_DIRECTORY}")
        csv_files = get_csv_files_from_directory(CSV_DIRECTORY)
        
        if csv_files:
            print(f"Found {len(csv_files)} CSV files:")
            for source_name, csv_path in csv_files.items():
                print(f"  - {source_name}: {csv_path}")
            
            for source_name, csv_path in csv_files.items():
                print(f"\nProcessing CSV: {source_name}...")
                chunks = process_csv(csv_path, source_name)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files.append(f"CSV: {source_name}")
                    print(f"  ✓ Added {len(chunks)} chunks from {source_name}")
                else:
                    print(f"  ✗ No chunks extracted from {source_name}")
        else:
            print(f"No CSV files found in {CSV_DIRECTORY}")
    
    if all_chunks:
        success = add_to_knowledge_base(all_chunks)
        if success:
            print(f"\nSuccessfully added {len(all_chunks)} total chunks to knowledge base")
            print(f"Processed files: {', '.join(processed_files)}")
        else:
            print("\n❌ Failed to add chunks to knowledge base")
    else:
        print("\n❌ No text chunks extracted from any files")

# Initialize knowledge base on startup
initialize_knowledge_base()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with Texas-themed chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    """Handle chat messages with Claude using RAG"""
    try:
        # Search knowledge base for relevant information
        relevant_docs = search_knowledge_base(message)
        
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
                source_info += "]"
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
    return {"status": "healthy", "message": "Texas AI Agent with RAG for RF Engineering is running!"}

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
            "auto_load_pdfs": AUTO_LOAD_PDFS,
            "auto_load_csvs": AUTO_LOAD_CSVS
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)