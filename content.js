(() => {
  const PANEL_ID = "yt-sentiment-panel";
  const BUTTON_ID = "yt-sentiment-analyze-btn";

  if (document.getElementById(PANEL_ID)) {
    return;
  }

  const panel = document.createElement("div");
  panel.id = PANEL_ID;
  panel.className = "yt-sentiment-panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "YouTube Sentiment");

  const title = document.createElement("div");
  title.className = "yt-sentiment-title";
  title.textContent = "YouTube Sentiment";

  const stats = document.createElement("div");
  stats.className = "yt-sentiment-stats";
  stats.innerHTML = `
    <div class="yt-sentiment-row">
      <span>Total:</span>
      <span id="yt-sentiment-total">-</span>
    </div>
    <div class="yt-sentiment-row">
      <span>Positive:</span>
      <span id="yt-sentiment-pos">-</span>
    </div>
    <div class="yt-sentiment-row">
      <span>Negative:</span>
      <span id="yt-sentiment-neg">-</span>
    </div>
    <div class="yt-sentiment-row">
      <span>Neutral:</span>
      <span id="yt-sentiment-neu">-</span>
    </div>
  `;

  const status = document.createElement("div");
  status.className = "yt-sentiment-status";
  status.id = "yt-sentiment-status";
  status.textContent = "Ready.";

  const button = document.createElement("button");
  button.id = BUTTON_ID;
  button.type = "button";
  button.textContent = "Analyze Comments";
  button.className = "yt-sentiment-button";

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "yt-sentiment-close";
  closeBtn.setAttribute("aria-label", "Close");
  closeBtn.textContent = "×";
  closeBtn.addEventListener("click", () => {
    panel.remove();
  });

  button.addEventListener("click", async () => {
    const url = new URL(window.location.href);
    const videoId = url.searchParams.get("v");
    if (!videoId) {
      console.warn("No video ID found in URL.");
      return;
    }

    const originalText = button.textContent;
    button.textContent = "Analyzing...";
    button.disabled = true;
    button.classList.add("is-loading");
    status.textContent = "Fetching comments...";

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: videoId,
          max_comments: 50,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log("Sentiment API response:", data);

      const totalEl = panel.querySelector("#yt-sentiment-total");
      const posEl = panel.querySelector("#yt-sentiment-pos");
      const negEl = panel.querySelector("#yt-sentiment-neg");
      const neuEl = panel.querySelector("#yt-sentiment-neu");

      if (totalEl) totalEl.textContent = String(data.total_comments ?? 0);
      if (posEl) posEl.textContent = String(data.sentiment_distribution?.POSITIVE ?? 0);
      if (negEl) negEl.textContent = String(data.sentiment_distribution?.NEGATIVE ?? 0);
      if (neuEl) neuEl.textContent = String(data.sentiment_distribution?.NEUTRAL ?? 0);

      status.textContent = "Done.";
    } catch (error) {
      console.error("Failed to analyze comments:", error);
      status.textContent = "Error. Check console.";
    } finally {
      button.textContent = originalText;
      button.disabled = false;
      button.classList.remove("is-loading");
    }
  });

  panel.appendChild(closeBtn);
  panel.appendChild(title);
  panel.appendChild(stats);
  panel.appendChild(status);
  panel.appendChild(button);
  document.documentElement.appendChild(panel);
})();
