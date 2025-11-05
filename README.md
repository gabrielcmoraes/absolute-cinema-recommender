# Recommender System — Two-Tower Retrieval + SVD Re-rank (MovieLens)

## This repository contains a production-style movie recommender built around:

- **Two-Tower model (PyTorch)** for retrieval trained with triplet loss using implicit positives/negatives from ratings (user tower + item tower with content features).
- **SVD factors (TruncatedSVD)** for collaborative **re-ranking** with optional content blend and MMR diversity.
- **Serving path**: AWS **SageMaker** endpoint (retrieval) + **AWS Lambda** (re-rank API) + small **Lambda Layer** for data files.

> You can run this repo **without retraining** using the prebuilt artifacts already included.

## Why serverless (Lambda) for the final hop?
Serverless gives us low idle cost, elastic concurrency, and a simple way to compose retrieval (SageMaker) + re-rank (SVD/MMR). The re-ranker is light (pure NumPy + small CSV/NPZ), so keeping it in Lambda reduces endpoint count and simplifies deployments.

What I optimized for
* **P50 latency**: retrieval call + re-rank (vector math only).
* **P95/P99 latency**: avoided library cold starts by shipping only numpy/pandas in the layer data (CSV/NPZ) and not in the function ZIP; the function code is small.
* **Cold start mitigation**: I keep the Lambda ZIP tiny and ship data files in a Layer (python/ path) so cold starts avoid heavy wheels. On init, Lambda loads svd_runtime_data.npz, builds Q_NORM, and caches movie_index. SageMaker containers precompute item vectors once per container start in model_fn, so per-request work is only a user-profile average + cosine scan.

## Production considerations
**Latency & cost**
* Retrieval is offloaded to SageMaker (GPU/CPU as needed), while the Lambda execution time is short (re-rank + formatting).
* SVD factors are pre-normalized; re-rank is a few dot products + a small MMR loop (Top-N only), keeping CPU time minimal.

**Reliability & fallbacks**
* If SageMaker is unavailable, Lambda falls back to pure SVD re-rank over global candidates (or popularity) to still return something.
* Payload is sanitized (titles, genres, scores clamped to [0,1]) to keep the contract consistent.

**Security**
* Lambda’s role has least-privilege: only sagemaker:InvokeEndpoint and basic CloudWatch logging.
* The public handler validates input (IDs, limits). CORS is explicit and narrow.

**Observability**
* Structured logs: endpoint latency, candidate size, rerank time, and which fallback triggered.
* CloudWatch alarms: error rate, throttle count, and high P95 duration.

## Re-rank strategy
* **Hybrid signal**: collaborative (SVD cosine) is the primary ranker.
* **Content awareness**: optional blend with genre Jaccard and year proximity kernel to damp odd out-of-genre picks.
* **Diversity via MMR**: classic Maximal Marginal Relevance with adjustable λ and a genre-weighted item-item similarity, improving list variety without tanking relevance.

## Evaluation
* Offline sweeps are saved under `artifacts/eval/` (e.g., `sweep_quick.final.csv`, `sweep_medium.final.csv`) with the final MMR/content weights we tested. You can add your own scripts to compute Recall@K / NDCG@K given your evaluation seeds.

## What’s inside
* **Retrieval (Two-Tower)**
  * Item tower: movie embedding (ID) + content features (genres + tag genome MLP).
  * User tower: user embedding → preference vector.
  * Training via TripletMarginLoss to separate positives from hard negatives.
* **Re-rank (SVD factors)**
  * TruncatedSVD on the user-item matrix (user-centered ratings).
  * Cosine similarity re-rank of candidates (+ optional blend with content/popularity and MMR for diversity).
* **Serving**
  * **SageMaker**: inference.py with model_fn/transform_fn hosts the Two-Tower and precomputes item vectors.
  * **Lambda**: lambda_function.py performs retrieval (via SageMaker) and final re-ranking + response formatting.
  * **Lambda Layer**: ships light data files to avoid large deployment packages.

---
## Repository layout
```bash
recommender-deploy-clean/
├─ artifacts/
│  ├─ eval/
│  │  ├─ sweep_quick.csv
│  │  ├─ sweep_quick.final.csv
│  │  ├─ sweep_medium.csv
│  │  └─ sweep_medium.final.csv
│  ├─ factors/
│  │  └─ svd_ctr_raw_qk128.npz             # SVD item factors (Q), movie_ids, movie_index
│  └─ twotower/
│     ├─ model.pth                         # Two-Tower weights
│     ├─ movie_encoder.pkl
│     └─ user_encoder.pkl
├─ data/raw/ml-latest/                      # MovieLens data used to build features/artifacts
│  ├─ movie.csv
│  ├─ rating.csv
│  ├─ link.csv
│  ├─ tag.csv
│  ├─ genome_scores.csv
│  └─ genome_tags.csv
├─ deployment/
│  ├─ lambda/
│  │  ├─ layers/recommender-data/python/   # Lambda layer payload (Python search path)
│  │  │  ├─ link.csv
│  │  │  ├─ movie.csv
│  │  │  └─ svd_runtime_data.npz           # (Same content as artifacts/factors*)
│  │  └─ lambda_function.py                # Ranking API (SageMaker + SVD/MMR blend)
│  └─ sagemaker/
│     ├─ code/
│     │  ├─ inference.py                   # model_fn/transform_fn (Two-Tower host)
│     │  ├─ recommender.py                 # Two-Tower architecture
│     │  └─ requirements.txt
│     └─ model.tar.gz                      # Ready-to-deploy model bundle for hosting
├─ src/
│  ├─ model/
│  │  ├─ __init__.py
│  │  └─ recommender.py                    # Two-Tower (training-time definition)
│  ├─ models/collaborative/
│  │  ├─ train_svd_centered.py             # SVD (user-centered ratings)
│  │  └─ train_svd_grid.py                 # SVD variants grid (optional)
│  ├─ scripts/
│  │  └─ train_two_tower.py                # Two-Tower training script (optional)
│  └─ utils/
│     ├─ io.py
│     └─ paths.py
├─ build_layer.py                           # Helper to package Lambda layer
├─ requirements.txt
├─ index.html                               # Simple manual test page (optional)
└─ README.md
```

> **Note:** the Lambda Layer uses the path `layers/recommender-data/python/` on purpose (AWS adds the `python/` folder to `sys.path` automatically).

---
## Quick start (use prebuilt artifacts)

### 1) Deploy SageMaker endpoint (Two-Tower retrieval)
- Upload `deployment/sagemaker/model.tar.gz` to S3.
- Create a **SageMaker model** using that tarball (entry point is `code/inference.py` which implements `model_fn`/`transform_fn`).
- Create an **endpoint** from that model (choose instance size as needed).

The tarball already contains:
code/ (inference.py, recommender.py, requirements.txt)
model/ (model.pth, movie_encoder.pkl, movie_features_df.pkl, content_feature_names.pkl)


### 2) Create the Lambda Layer (data files)
- Zip `deployment/lambda/layers/recommender-data/` **keeping** the `python/` directory at the zip root.
- Publish as a **Lambda Layer** and note its ARN.

Contents:
- `svd_runtime_data.npz` — SVD Q factors + movie index
- `movie.csv` — titles/genres
- `link.csv` — TMDB → MovieLens mapping (used if client sends TMDB IDs)

### 3) Create the Lambda function (re-rank API)
- Use `deployment/lambda/lambda_function.py` as the handler (`lambda_function.lambda_handler`).
- Attach the Layer from step 2.
- Ensure IAM role has `sagemaker:InvokeEndpoint` permission.
- Adjust constants in the file if needed:
  ```python
  REGION = "us-east-2"
  ENDPOINT_NAME = "recomendador-api"
Test
```json
{
  "tmdb_ids": [1585, 4935, 597],
  "top_n": 5
}
```
Response
```json
{
  "recommendations": [
    {"movie_id": 1234, "title": "Some Movie (2006)", "genres": "Drama|Fantasy", "affinity_score": 0.78}
  ]
}
```


## How it works (serving path)
1.  Client sends seed movies to Lambda.
2.  Lambda calls SageMaker (Two-Tower) to retrieve ~200 candidates.
3.  Lambda re-ranks using:
    * SVD cosine similarity,
    * optional content/popularity blend,
    * optional MMR diversity.
4.  Returns Top-N with `affinity_score` normalized to [0, 1].
You can tune (inside `lambda_function.py`):

```python
BLEND_WEIGHTS = {
  "alpha_cf": 1.00,  # collaborative
  "beta_gen": 0.15,  # genre overlap (Jaccard)
  "gamma_year": 0.10,# year proximity kernel
  "delta_pop": 0.05, # popularity (if available)
  "tau_year": 8.0
}

# Diversity via MMR
MMR_PARAMS = {
  "lambda_mmr": 0.80,  # 1.0=only relevance, 0.0=only diversity
  "genre_weight": 0.60
}
```

## Optional — (Re)training locally
You do not need this for deployment. For full reproducibility:
```powershell
$env:PYTHONPATH="$PWD\src"
```

### Train SVD (user-centered)
```powershell
python -m src.models.collaborative.train_svd_centered --k 128
```
It saves `artifacts/factors/svd_centered_qk128.npz`.
If you change the file name, update `deployment/lambda/layers/.../svd_runtime_data.npz` accordingly.


### Train Two-Tower
```powershell
python -m src.scripts.train_two_tower
```
It writes artifacts under `artifacts/twotower/` and a `model/` folder for packaging into a new `model.tar.gz`.

## Notes for maintainers
* Imports: repo is structured as a package (see `src/__init__.py` and `src/model/__init__.py`).
Quick check:
```powershell
$env:PYTHONPATH="$PWD\src"
python -c "import src.model.recommender as r; print(r.TwoTowerModel)"
```
* The **Lambda Layer** path intentionally includes `/python/` so Python can import the shipped CSV/NPZ.

## Data & License
* This project uses **MovieLens** datasets and **Tag Genome (GroupLens)**.
Please follow their licenses and cite appropriately.

## Contact
If you want to know more about the project, message me at my [Linkedin: Gabriel Moraes](https://www.linkedin.com/in/gabriel-de-moraes-b63471157/). The website can be accessed through this [link](http://movie-recommender-gabriel-moraes-portfolio.s3-website.us-east-2.amazonaws.com/)

## TL;DR
* **Architecture**: Two-Tower retrieval (PyTorch, triplet loss) + SVD re-rank + MMR diversity.
* **Cloud**: SageMaker endpoint (model_fn/transform_fn), AWS Lambda + Layer (small, fast, cheap).
* **Engineering**: Clean artifacts, reproducible training (optional), production-grade packaging.
