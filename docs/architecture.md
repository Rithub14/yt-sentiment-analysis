# Architecture Overview

## Pipeline Flow
1. **Data collection**
   - `/analyze` fetches YouTube comments and can persist raw + labeled data.

2. **Data versioning (DVC)**
   - Raw data: `data/raw/comments.json`
   - Processed data: `data/processed/comments_cleaned.csv`
   - Tracked via DVC stages and `dvc.lock`.

3. **Training pipeline (DVC stages)**
   - `preprocess` → labels comments via sentiment pipeline
   - `train` → fine-tunes model and logs to MLflow
   - `drift` → compares distribution vs baseline
   - `evaluate` → metrics + confusion matrix report

4. **Model registry (MLflow)**
   - Baseline model registered as v1
   - Fine-tuned models registered as v2+ and staged
   - Promotion workflow moves Staging → Production

5. **CI/CD**
   - Tests + Docker build
   - Optional DVC repro + drift gate
   - Reports uploaded as CI artifacts

6. **Deployment**
   - K8s deployment + service
   - Image tag override via deploy workflow
   - Deploys only if Production model exists

## Notes
- Environment separation via ConfigMaps + Secrets.
- Metrics endpoint (`/metrics`) exposes latency and request counters.
