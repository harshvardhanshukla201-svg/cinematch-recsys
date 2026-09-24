# CineMatch: Multi-Stage Hybrid Recommendation Engine

An end-to-end, production-grade movie recommendation platform pairing collaborative filtering with deep semantic vector search, dynamic session re-ranking, and Maximal Marginal Relevance (MMR) diversity optimization.

Built with a high-performance **FastAPI** backend, dense vector indexing via **FAISS**, matrix factorization via **SVD**, and a modern **Next.js (App Router)** cinema client — fully containerized using **Docker Compose**.

---

## 🏛️ System Architecture

CineMatch uses an industrial 3-stage recommendation funnel designed to balance relevance, computational efficiency, and serendipity:

```text
[ User Interaction / Search Query ]
                    |
                    v
+----------------------------------------------------------+
| Stage 1: Candidate Retrieval (L1)                        |
| - FAISS Vector Search: Top-K dense semantic retrieval     |
|   over 384-dimensional text embeddings                    |
| - SVD Matrix Factorization: Behavioral latent space        |
+----------------------------┬-------------------------------+
                             | (Top ~300 candidates)
                             v
+----------------------------------------------------------+
| Stage 2: Hybrid Scoring & Ranking (L2)                    |
| - Linear blend: Collaborative rating (65%) +               |
|   Empirical community baseline (35%)                       |
| - Recency calibration for contemporary catalog titles       |
| - Real-time session vector projection & live scoring        |
+----------------------------┬-------------------------------+
                             | (Ranked candidate list)
                             v
+----------------------------------------------------------+
| Stage 3: MMR Diversification Layer (L3)                   |
| - Maximal Marginal Relevance with tunable lambda            |
| - Mitigates filter-bubble homogenization                    |
+----------------------------┬-------------------------------+
                             |
                             v
+----------------------------------------------------------+
| Async Metadata Enrichment (Async httpx + TMDB)             |
| - Poster artwork resolution & plot synopsis ingestion        |
| - Rate-limited concurrency pool (asyncio.Semaphore)          |
| - Disk-persisted caching layer                                |
+----------------------------------------------------------+
```

---

## 🔬 Offline Evaluation & Benchmarks

The system includes an automated evaluation suite (`backend/evaluate.py`) benchmarking ranking accuracy, intra-list diversity, and catalog coverage across test cohorts:

| Architecture | NDCG@10 | Precision@10 | Intra-List Diversity | Catalog Coverage |
|---|:---:|:---:|:---:|:---:|
| **Collaborative (SVD)** | 0.9293 | 0.2920 | 0.5728 | 3.3% (58 / 1,782) |
| **Content-Based (FAISS)** | 0.9591 | 0.0680 | 0.4250 | 19.5% (348 / 1,782) |
| **Hybrid (Ensemble)** | **0.9384** | **0.1900** | **0.6112** | **1.9% (33 / 1,782)** |

### Key Metrics

- **NDCG@10 (0.9384)** — Validates that high-relevance items are placed consistently at top ranking positions.
- **Intra-List Diversity (0.6112)** — The hybrid engine delivers the highest candidate variety, preventing recommendation fatigue.
- **Maximal Marginal Relevance (MMR)**:

$$
\text{MMR}(d) = \arg\max_{d \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(d, q) - (1 - \lambda) \max_{s \in S} \text{Sim}_2(d, s) \right]
$$

  Users can steer the exploration parameter ($\lambda \in [0.2, 1.0]$) in real time from the UI.

---

## ⚡ Tech Stack

- **Backend & ML Inference:** Python 3.10, FastAPI, FAISS (`IndexFlatIP`), scikit-surprise (SVD), LightGBM, NumPy, Pandas.
- **Asynchronous Pipelines:** `httpx`, `asyncio`, TMDB REST API.
- **Frontend:** Next.js 15+ (App Router), TypeScript, Tailwind CSS, Lucide Icons.
- **DevOps & Infrastructure:** Docker, Docker Compose, multi-stage Node/Python Alpine builds.

---

## 🚀 Quickstart with Docker Compose

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- A [TMDB API Key](https://www.themoviedb.org/documentation/api).

### 1. Clone the Repository

```bash
git clone https://github.com/harshvardhanshukla201-svg/cinematch-recsys.git
cd cinematch-recsys
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
TMDB_API_KEY=your_tmdb_api_key_here
```

### 3. Spin Up the Platform

```bash
docker compose up --build -d
```

### 4. Access the Services

- **Cinema Web UI:** `http://localhost:3000`
- **FastAPI Interactive Docs:** `http://localhost:8000/docs`
- **Backend Health Check:** `http://localhost:8000/`

---

## 💻 Local Development Setup (Without Docker)

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 📊 Running Offline Benchmarks

Run the offline evaluation harness across user cohorts:

```bash
cd backend
python evaluate.py
```

This reports NDCG@10, Precision@10, intra-list diversity, and catalog coverage metrics for each architecture (Collaborative, Content-Based, and Hybrid), reproducing the results shown above.
