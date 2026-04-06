# Multilingual RAG System

A production-quality Retrieval-Augmented Generation (RAG) system built on the Natural Questions dataset. This system features text chunking, metadata enrichment, Sentence-Transformers for embeddings, FAISS for vector search, intelligent caching, query processing with expansion, and multi-provider LLM support (Ollama, Groq, OpenAI, Google Gemini).

## Setup

1. **Environment Variables**:
   Copy `.env.example` to `.env` and configure your settings:
   ```bash
   cp .env.example .env
   ```
   *Open `.env` and set `LLM_PROVIDER` to your preferred backend (`ollama`, `groq`, `openai`, or `google`). Be sure to configure the corresponding API key (`GROQ_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`). If you use Ollama, make sure it is running locally (e.g. `ollama serve`). Fallback mode is active if no LLM is provided.*

2. **Dataset**:
   Ensure you have the datasets files in the project root.

3. **Install Dependencies**:
   This project uses `uv` for dependency management:
   ```bash
   uv pip install -r requirements.txt
   ```

## Available Commands

The application is managed through the `main.py` CLI script, which provides several subcommands.

### 1. Build the FAISS Index
Before querying, you need to build the vector index from the dataset.
```bash
# Build with default sample size (10,000 rows as per .env)
python main.py build

# Build with a specific sample size (useful for quick testing)
python main.py build --sample 500

# Build using the full dataset
python main.py build --sample 0
```
This process extracts text, chunks it, generates embeddings, builds the FAISS index, and saves it to the `index_data/` directory.

### 2. Interactive Query Mode
Run an interactive CLI loop to ask questions directly in your terminal.
```bash
python main.py query

# You can also specify the number of retrieved passages
python main.py query --top-k 5
```
*Special Commands inside the CLI*:
- Type `quit`, `exit`, or `q` to exit.
- Type `clear` to reset the conversation history and cache.

### 3. Start the API Server
Start the FastAPI application to serve the RAG pipeline over HTTP.
```bash
python main.py serve
```
By default, this runs on `http://0.0.0.0:8000`. You can access the automatic API documentation at `http://127.0.0.1:8000/docs`.

### 4. Evaluate Metrics
Run automated evaluation (Precision, Recall, MRR, latency) on a sample of the dataset.
```bash
python main.py evaluate --num-samples 100 --top-k 5
```

### 5. Data Exploration
Run EDA (Exploratory Data Analysis) on the dataset to generate statistics and plots.
```bash
python main.py explore --sample 5000
```
Plots and JSON reports are saved to the `output/` directory.

 
```
