# YouTube Sentiment Analysis MLOps Project

Minimal backend + Chrome extension to analyze YouTube comments with a HuggingFace model.

**Features**
- Fetch top-level YouTube comments via YouTube Data API v3
- Sentiment analysis (multilingual model)
- FastAPI endpoint `/analyze`
- Chrome extension popup to trigger analysis and show counts

**Setup**
1. Create `.env` from the template:
   ```bash
   cp .env.template .env
   ```
2. Fill in your API key in `.env`:
   ```
   YOUTUBE_API_KEY=...
   YOUTUBE_VIDEO_ID=...   # optional for tests
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```

**Run the API**
```bash
uv run uvicorn --app-dir src youtube_sentiment.main:app --reload
```

**Test the API**
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_id":"VIDEO_ID","max_comments":50}'
```

**Run tests**
```bash
uv run pytest
```

Integration tests (require `.env` values):
```bash
RUN_INTEGRATION_TESTS=1 uv run pytest -m integration
```

**Chrome Extension**
1. Go to `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** and select this repo folder
4. Open a YouTube watch page
5. Click the extension icon → **Analyze Comments**

**Notes**
- The model cache is stored in `.hf-cache/` (override with `HF_CACHE_DIR`).
