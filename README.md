# AI Tryouts

> A hands-on learning lab for AI concepts — built from scratch, inspired by Karpathy's LLM101n

This repo is where I'm building out core AI/ML concepts from the ground up. No shortcuts, no black boxes — just code, math, and a lot of experimentation. Think of it as my public learning journal for agentic AI and foundational ML techniques.

---

## Projects

| Project | What it does | Core concept | Stack |
|---------|--------------|--------------|-------|
| **HNSW** | Builds a vector search index and performs similarity search on sentence embeddings | Approximate nearest neighbor search using Hierarchical Navigable Small World graphs | FAISS, Sentence Transformers, NumPy |
| **Semantic Search Engine** | Interactive Wikipedia search powered by embeddings and cosine similarity | Semantic search with transformer-based embeddings, visualized calculation steps | Streamlit, Hugging Face Datasets, Sentence Transformers, scikit-learn |

---

## HNSW Vector Search

### What it is
A command-line implementation of vector similarity search using FAISS's HNSW (Hierarchical Navigable Small World) algorithm. Takes 20 distinct sentences, converts them to embeddings using a sentence transformer model, builds a multi-layer graph index, and performs fast approximate nearest neighbor search.

### Why it's interesting
HNSW is one of the most efficient algorithms for high-dimensional vector search — the same tech powering modern vector databases. This project demystifies how it works under the hood: multi-layer graphs, greedy search through layers, and the tradeoffs between speed and accuracy (controlled by M, efConstruction, efSearch parameters). Plus, it includes a detailed conceptual walkthrough of the algorithm printed right in the output.

### How to run it

**Prerequisites:**
- Python 3.x
- pip

**Install & Run:**
```bash
cd HNSW

# Create and activate virtual environment
python3 -m venv venv_hnsw
source venv_hnsw/bin/activate  # On Windows: .\venv_hnsw\Scripts\activate

# Install dependencies
pip install sentence-transformers faiss-cpu numpy

# Run the script
python vector_search_hnsw.py
```

### What you'll see
- Status messages as the model loads (first run downloads the all-MiniLM-L6-v2 model)
- Sample of the generated 384-dimensional embeddings
- HNSW graph construction details (max level, entry point)
- A query sentence and its top 5 most similar sentences with distance scores
- A comprehensive explanation of how HNSW's multi-layer greedy search works
- Index parameters (M=32, efConstruction=200, efSearch=100)

**Try customizing:** Change the sentences, adjust HNSW parameters (M, efConstruction, efSearch), or swap in a different sentence transformer model to see how results change.

---

## Semantic Search Engine with Streamlit

### What it is
A full-stack semantic search engine that lets you search 100 Wikipedia articles using natural language queries. Built with Streamlit for the UI, Hugging Face Datasets for data, and Sentence Transformers for embeddings. It computes cosine similarity between your query and all articles, then shows you the top 5 matches — along with every calculation step visualized on screen.

### Why it's interesting
This isn't just "type and get results" — it shows you exactly how semantic search works internally. You see the query embedding (a 384-dim vector), the cosine similarity formula in LaTeX, all similarity scores, and a breakdown of dot products and magnitudes for each result. It's built to teach as much as it is to search. Plus, it saves embeddings locally so subsequent runs are instant.

### How to run it

**Prerequisites:**
- Python 3.x
- pip

**Install & Run:**
```bash
cd Semantic_Search_Engine_with_Streamlit

# Create and activate virtual environment
python3 -m venv bert-env
source bert-env/bin/activate  # On Windows: bert-env\Scripts\activate

# Install dependencies
pip install sentence-transformers streamlit datasets scikit-learn numpy

# Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### What you'll see

**On first run:**
1. Model loading (all-MiniLM-L6-v2)
2. Wikipedia articles loading from Hugging Face (100 articles, truncated to 1000 chars each)
3. A button to generate and save embeddings (creates `wikipedia_embeddings.npy` and `wikipedia_texts.npy`)

**On subsequent runs:**
- Instant load from saved `.npy` files

**When you search:**
- Query encoding into a vector
- The cosine similarity formula (LaTeX rendered)
- Expandable view of all similarity scores
- Top 5 results with preview text
- For each result: expandable detailed calculation showing dot product, magnitudes, and final similarity score
- Query time in seconds

**Try searching for:** "ancient civilizations", "space exploration", "machine learning", or anything you're curious about.

---

## What I learned

Building these projects taught me way more than just reading papers ever could.

**On embeddings:**
I used to think of embeddings as magic black boxes. Now I get it — they're just dense vector representations that capture semantic meaning. The `all-MiniLM-L6-v2` model turns text into 384-dimensional vectors, and similar meanings end up close together in that space. Cosine similarity is the key: just dot product divided by magnitudes. Simple, elegant.

**On HNSW:**
HNSW blew my mind. It's not brute-force search — it's a hierarchical graph where you zoom in from coarse to fine. Upper layers give you fast approximations, layer 0 gives you precision. The parameters (M, efConstruction, efSearch) let you trade speed for accuracy. Now I understand why vector databases can search billions of vectors in milliseconds.

**On tooling:**
FAISS is ridiculously fast. Streamlit makes it trivial to build interactive demos. Hugging Face has datasets and models for everything. The ecosystem around transformers and vector search is mature and well-documented — you just have to dive in.

---

## Inspiration

This repo is heavily inspired by [Karpathy's LLM101n](https://github.com/karpathy/LLM101n) — the idea that you learn by building, not just reading. Ground-up implementation beats surface-level understanding every time.

---

## Links

Learning in public over at **TechBitsbyGN**:

- **YouTube:** [youtube.com/@TechBitsbyGN](https://www.youtube.com/@TechBitsbyGN)
- **Instagram:** [instagram.com/techbitsbygn](https://www.instagram.com/techbitsbygn)
- **X (Twitter):** [x.com/techbitsbygn](https://x.com/techbitsbygn)

---

**License:** MIT (do whatever you want with this code)
**Questions?** Open an issue or reach out on any of the platforms above.
