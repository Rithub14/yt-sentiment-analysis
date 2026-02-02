const statusEl = document.getElementById("status");
const analyzeBtn = document.getElementById("analyze-btn");
const totalEl = document.getElementById("stat-total");
const posEl = document.getElementById("stat-pos");
const negEl = document.getElementById("stat-neg");
const neuEl = document.getElementById("stat-neu");

let currentVideoId = null;

function extractVideoId(url) {
  try {
    const parsed = new URL(url);
    const isYouTube =
      parsed.hostname === "www.youtube.com" ||
      parsed.hostname === "youtube.com" ||
      parsed.hostname.endsWith(".youtube.com");
    if (!isYouTube || parsed.pathname !== "/watch") {
      return null;
    }
    return parsed.searchParams.get("v");
  } catch {
    return null;
  }
}

chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const tab = tabs[0];
  if (!tab || !tab.url || tab.id == null) {
    statusEl.textContent = "No active tab.";
    return;
  }

  const videoId = extractVideoId(tab.url);
  if (!videoId) {
    statusEl.textContent = "Open a YouTube watch page.";
    analyzeBtn.disabled = true;
    return;
  }

  currentVideoId = videoId;
  statusEl.textContent = `Video ID: ${videoId}`;
  analyzeBtn.disabled = false;
});

analyzeBtn.addEventListener("click", () => {
  if (!currentVideoId) {
    return;
  }

  const originalText = analyzeBtn.textContent;
  analyzeBtn.textContent = "Analyzing...";
  analyzeBtn.disabled = true;
  statusEl.textContent = "Fetching comments...";

  fetch("http://localhost:8000/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: currentVideoId,
      max_comments: 50,
    }),
  })
    .then(async (response) => {
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error ${response.status}: ${errorText}`);
      }
      return response.json();
    })
    .then((data) => {
      console.log("Sentiment API response:", data);
      totalEl.textContent = String(data.total_comments ?? 0);
      posEl.textContent = String(data.sentiment_distribution?.POSITIVE ?? 0);
      negEl.textContent = String(data.sentiment_distribution?.NEGATIVE ?? 0);
      neuEl.textContent = String(data.sentiment_distribution?.NEUTRAL ?? 0);
      statusEl.textContent = "Done.";
    })
    .catch((error) => {
      console.error("Failed to analyze comments:", error);
      statusEl.textContent = "Error. Check console.";
    })
    .finally(() => {
      analyzeBtn.textContent = originalText;
      analyzeBtn.disabled = false;
    });
});
