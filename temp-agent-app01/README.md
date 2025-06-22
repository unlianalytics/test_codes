# Texas AI Agent 🤠 - Enhanced with RAG

A sophisticated AI chat agent with a Texas flag theme, built with FastAPI, HTML, CSS, and powered by Claude AI. Now enhanced with RAG (Retrieval-Augmented Generation) capabilities for deeper knowledge access.

## Features

- 🎨 **Texas Flag Theme**: Beautiful UI with Texas colors (red, white, blue)
- 🤖 **Claude AI Integration**: Powered by Anthropic's Claude 3.5 Sonnet
- 💬 **Real-time Chat**: Interactive chat interface like ChatGPT
- 📚 **RAG Enhanced**: Retrieval-Augmented Generation with PDF knowledge base
- 📄 **PDF Processing**: Automatic extraction and indexing of PDF documents
- 🔍 **Vector Search**: Semantic search through knowledge base
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **FastAPI Backend**: High-performance Python web framework
- 🎭 **Texas Personality**: AI assistant with Texas charm and expressions

## Screenshots

The application features:
- Animated Texas flag in the header
- RAG Enhanced badge showing knowledge capabilities
- Clean chat interface with Texas-themed colors
- Knowledge base status indicator
- Responsive design for all devices
- Loading animations and smooth interactions

## Prerequisites

- Python 3.8 or higher
- Anthropic API key (free tier available)
- PDF file to enhance knowledge base (optional)

## Installation

1. **Clone or download this project**
   ```bash
   git clone <repository-url>
   cd temp-agent-app01
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   - Copy `env_example.txt` to `.env`
   - Get your Anthropic API key from [Anthropic Console](https://console.anthropic.com/)
   - Replace `your-anthropic-api-key-here` with your actual API key

4. **Add your PDF knowledge base (optional)**
   - Place your PDF file at `C:\Users\magno\Downloads\deep_dive.pdf`
   - Or modify the path in `main.py` to point to your PDF file
   - The system will automatically process and index the PDF on startup

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Open your browser**
   - Navigate to `http://localhost:8000`
   - Start chatting with your enhanced Texas AI agent!

## Project Structure

```
temp-agent-app01/
├── main.py              # FastAPI application with RAG
├── requirements.txt     # Python dependencies
├── env_example.txt      # Environment variables template
├── README.md           # This file
├── templates/
│   └── index.html      # Main HTML template
└── static/
    ├── styles.css      # Texas-themed CSS styles
    └── script.js       # Frontend JavaScript
```

## API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send messages to Claude AI with RAG
- `GET /health` - Health check endpoint
- `GET /knowledge-status` - Check knowledge base status

## RAG Features

### PDF Processing
- Automatic text extraction from PDF files
- Intelligent text chunking for optimal retrieval
- Metadata tracking (page numbers, chunk IDs)

### Vector Search
- Semantic search using sentence transformers
- ChromaDB for efficient vector storage
- Top-k retrieval for relevant context

### Enhanced Responses
- Context-aware responses using retrieved information
- Natural integration of knowledge base content
- Maintains Texas personality while providing detailed answers

## Customization

### Changing the Theme
The Texas theme is defined in CSS variables in `static/styles.css`:
```css
:root {
    --texas-red: #D00C33;
    --texas-blue: #002868;
    --texas-white: #FFFFFF;
    /* ... */
}
```

### Modifying AI Personality
Edit the system prompt in `main.py`:
```python
system_prompt = f"""You are a helpful AI assistant with a Texas personality. 
You're friendly, knowledgeable, and occasionally use Texas expressions. 
Keep responses conversational and helpful while maintaining the Texas spirit.

You have access to additional knowledge that you can use to provide more accurate and detailed responses.
When using this knowledge, make sure to integrate it naturally into your Texas-style responses.

Additional knowledge context:{context}"""
```

### Adding Different PDF Files
Modify the PDF path in `main.py`:
```python
def initialize_knowledge_base():
    pdf_path = r"path/to/your/pdf/file.pdf"
    # ... rest of the function
```

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Make sure your `.env` file exists and contains the correct API key
   - Verify your Anthropic API key is valid and has credits

2. **PDF Processing Error**
   - Ensure the PDF file exists at the specified path
   - Check that the PDF is not password-protected
   - Verify the PDF contains extractable text

3. **Knowledge Base Issues**
   - Check the `/knowledge-status` endpoint for status
   - Restart the application to reinitialize the knowledge base
   - Verify ChromaDB dependencies are installed

4. **Port Already in Use**
   - Change the port in `main.py`:
   ```python
   uvicorn.run(app, host="0.0.0.0", port=8001)  # Change 8000 to 8001
   ```

5. **Dependencies Not Found**
   - Run: `pip install -r requirements.txt`
   - Make sure you're using Python 3.8+

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Features
- Backend: Modify `main.py`
- Frontend: Edit files in `templates/` and `static/`
- Styling: Update `static/styles.css`
- RAG: Modify PDF processing and vector search functions

### Knowledge Base Management
- The knowledge base is automatically initialized on startup
- ChromaDB stores vectors locally in memory
- To clear and rebuild: restart the application

## Performance Notes

- First startup may take longer due to PDF processing
- Vector embeddings are generated using the `all-MiniLM-L6-v2` model
- ChromaDB provides fast similarity search
- Response times depend on knowledge base size and query complexity

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!

---

**Howdy! 🤠** Enjoy your enhanced Texas AI Agent with RAG capabilities! 