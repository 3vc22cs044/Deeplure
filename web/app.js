/**
 * DeepLure Color-Invariant Saree Recognition
 * Interactive Web Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs Switching
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Theme Switching Logic
  const themeSwitcher = document.getElementById("theme-switcher");
  if (themeSwitcher) {
    const savedTheme = localStorage.getItem("deeplure-theme") || "emerald-gold";
    document.documentElement.setAttribute("data-theme", savedTheme);
    themeSwitcher.value = savedTheme;

    themeSwitcher.addEventListener("change", (e) => {
      const selected = e.target.value;
      document.documentElement.setAttribute("data-theme", selected);
      localStorage.setItem("deeplure-theme", selected);
    });
  }

  // State
  let sampleQueriesData = {};
  let currentActiveQueryType = "cross_colorway_match";
  let selectedQueryFile = null;
  let selectedQueryPath = null;

  let verifyFileA = null, verifyPathA = null;
  let verifyFileB = null, verifyPathB = null;

  // Initialize
  loadSampleQueries();
  loadEvaluationMetrics();
  loadEfficiencyReport();

  // -------------------------------------------------------------
  // Tab 1: Identification Logic
  // -------------------------------------------------------------
  async function loadSampleQueries() {
    try {
      const res = await fetch("/api/sample_queries");
      sampleQueriesData = await res.json();
      renderSampleQueries(currentActiveQueryType);
    } catch (err) {
      console.error("Failed to load sample queries:", err);
    }
  }

  const sampleTags = document.querySelectorAll(".sample-tag");
  sampleTags.forEach(tag => {
    tag.addEventListener("click", () => {
      sampleTags.forEach(t => t.classList.remove("active"));
      tag.classList.add("active");
      currentActiveQueryType = tag.getAttribute("data-type");
      renderSampleQueries(currentActiveQueryType);
    });
  });

  function renderSampleQueries(category) {
    const list = document.getElementById("sample-queries-list");
    list.innerHTML = "";
    const items = sampleQueriesData[category] || [];

    if (items.length === 0) {
      list.innerHTML = `<span class="text-dim text-sm">No samples available</span>`;
      return;
    }

    items.forEach((item, idx) => {
      const card = document.createElement("div");
      card.className = "sample-thumb-card";
      card.innerHTML = `
        <img src="${item.image_url}" alt="Sample ${idx}">
        <span>${item.category}</span>
      `;
      card.addEventListener("click", () => selectSampleQuery(item));
      list.appendChild(card);
    });

    // Auto-select first item if nothing selected
    if (items.length > 0 && !selectedQueryPath && !selectedQueryFile) {
      selectSampleQuery(items[0]);
    }
  }

  function selectSampleQuery(item) {
    selectedQueryPath = item.file_path;
    selectedQueryFile = null;

    // Update preview
    const previewBox = document.getElementById("query-preview-box");
    const imgPreview = document.getElementById("query-img-preview");
    const heatmapPreview = document.getElementById("query-heatmap-preview");

    imgPreview.src = item.image_url;
    heatmapPreview.src = item.image_url; // will be replaced on run
    previewBox.style.display = "block";

    // Auto-trigger search for immediate wow factor
    runIdentification();
  }

  // Dropzone handling for Tab 1
  const queryDropzone = document.getElementById("query-dropzone");
  const queryFileInput = document.getElementById("query-file-input");

  queryDropzone.addEventListener("click", () => queryFileInput.click());
  queryFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleQueryFileUpload(e.target.files[0]);
    }
  });

  queryDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    queryDropzone.style.borderColor = "var(--accent-primary)";
  });
  queryDropzone.addEventListener("dragleave", () => {
    queryDropzone.style.borderColor = "";
  });
  queryDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    queryDropzone.style.borderColor = "";
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleQueryFileUpload(e.dataTransfer.files[0]);
    }
  });

  function handleQueryFileUpload(file) {
    selectedQueryFile = file;
    selectedQueryPath = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      const previewBox = document.getElementById("query-preview-box");
      const imgPreview = document.getElementById("query-img-preview");
      const heatmapPreview = document.getElementById("query-heatmap-preview");

      imgPreview.src = e.target.result;
      heatmapPreview.src = e.target.result;
      previewBox.style.display = "block";
    };
    reader.readAsDataURL(file);
  }

  document.getElementById("btn-run-identify").addEventListener("click", runIdentification);

  async function runIdentification() {
    const statusText = document.getElementById("identify-status");
    const resultsContainer = document.getElementById("identify-results-list");
    const heatmapPreview = document.getElementById("query-heatmap-preview");

    statusText.textContent = "Extracting color-invariant embeddings...";
    resultsContainer.innerHTML = `<div class="empty-state"><p>Searching gallery database...</p></div>`;

    const formData = new FormData();
    if (selectedQueryFile) {
      formData.append("file", selectedQueryFile);
    } else if (selectedQueryPath) {
      formData.append("image_path", selectedQueryPath);
    } else {
      statusText.textContent = "Please select or upload an image.";
      return;
    }
    formData.append("top_k", 5);

    try {
      const res = await fetch("/api/identify", { method: "POST", body: formData });
      const data = await res.json();

      if (data.error) {
        statusText.textContent = "Error: " + data.error;
        return;
      }

      // Display structural gradient heatmap
      if (data.query_structural_heatmap) {
        heatmapPreview.src = "data:image/jpeg;base64," + data.query_structural_heatmap;
      }

      statusText.textContent = `Found ${data.ranked_matches.length} ranked matches`;
      resultsContainer.innerHTML = "";

      data.ranked_matches.forEach(m => {
        const card = document.createElement("div");
        card.className = "match-card";
        const isMatch = m.similarity_score >= data.calibrated_threshold;
        const confColor = isMatch ? "text-emerald" : "text-dim";

        card.innerHTML = `
          <div class="match-rank-badge">#${m.rank}</div>
          <img src="${m.image_url}" class="match-img" alt="${m.design_id}">
          <div class="match-info">
            <div class="match-header">
              <div>
                <span class="match-title">${m.design_id}</span>
                <span class="match-category">&bull; ${m.category}</span>
              </div>
              <span class="text-sm ${confColor}"><strong>${m.confidence}</strong></span>
            </div>
            <div class="similarity-bar-wrap">
              <div class="similarity-label">
                <span>Cosine Similarity: <strong>${m.similarity_score}</strong></span>
                <span>${m.similarity_percent}%</span>
              </div>
              <div class="progress-bar">
                <div class="progress-fill" style="width: ${m.similarity_percent}%;"></div>
              </div>
            </div>
          </div>
        `;
        resultsContainer.appendChild(card);
      });

    } catch (err) {
      console.error(err);
      statusText.textContent = "Failed to run search.";
    }
  }

  // -------------------------------------------------------------
  // Tab 2: Verification Logic
  // -------------------------------------------------------------
  const verifyDropA = document.getElementById("verify-drop-a");
  const verifyFileInA = document.getElementById("verify-file-a");
  const verifyPreviewA = document.getElementById("verify-preview-a");
  const placeholderA = document.getElementById("placeholder-a");

  verifyDropA.addEventListener("click", () => verifyFileInA.click());
  verifyFileInA.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      verifyFileA = e.target.files[0];
      verifyPathA = null;
      showVerifyPreview(verifyFileA, verifyPreviewA, placeholderA);
    }
  });

  const verifyDropB = document.getElementById("verify-drop-b");
  const verifyFileInB = document.getElementById("verify-file-b");
  const verifyPreviewB = document.getElementById("verify-preview-b");
  const placeholderB = document.getElementById("placeholder-b");

  verifyDropB.addEventListener("click", () => verifyFileInB.click());
  verifyFileInB.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      verifyFileB = e.target.files[0];
      verifyPathB = null;
      showVerifyPreview(verifyFileB, verifyPreviewB, placeholderB);
    }
  });

  function showVerifyPreview(file, imgElem, placeholderElem) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imgElem.src = e.target.result;
      imgElem.style.display = "block";
      placeholderElem.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  function setVerifyPath(path, imgElem, placeholderElem, isSlotA = true) {
    if (isSlotA) {
      verifyPathA = path;
      verifyFileA = null;
    } else {
      verifyPathB = path;
      verifyFileB = null;
    }
    imgElem.src = "/" + path.replace("\\", "/");
    imgElem.style.display = "block";
    placeholderElem.style.display = "none";
  }

  // Preset Pair Buttons
  document.getElementById("btn-pair-pos").addEventListener("click", async () => {
    if (sampleQueriesData.cross_colorway_match && sampleQueriesData.cross_colorway_match.length > 0) {
      const q = sampleQueriesData.cross_colorway_match[0];
      setVerifyPath(q.file_path, verifyPreviewA, placeholderA, true);
      // Pair with canonical gallery
      const galRes = await fetch("/api/gallery");
      const galData = await galRes.json();
      const target = galData.gallery.find(g => g.design_id === q.design_id) || galData.gallery[0];
      setVerifyPath(target.image_url.slice(1), verifyPreviewB, placeholderB, false);
      runVerification();
    }
  });

  document.getElementById("btn-pair-distractor").addEventListener("click", async () => {
    if (sampleQueriesData.same_palette_diff_motif && sampleQueriesData.same_palette_diff_motif.length > 0) {
      const q = sampleQueriesData.same_palette_diff_motif[0];
      setVerifyPath(q.file_path, verifyPreviewA, placeholderA, true);
      const galRes = await fetch("/api/gallery");
      const galData = await galRes.json();
      const target = galData.gallery[0];
      setVerifyPath(target.image_url.slice(1), verifyPreviewB, placeholderB, false);
      runVerification();
    }
  });

  document.getElementById("btn-pair-neg").addEventListener("click", async () => {
    const galRes = await fetch("/api/gallery");
    const galData = await galRes.json();
    if (galData.gallery.length >= 2) {
      setVerifyPath(galData.gallery[1].image_url.slice(1), verifyPreviewA, placeholderA, true);
      setVerifyPath(galData.gallery[5].image_url.slice(1), verifyPreviewB, placeholderB, false);
      runVerification();
    }
  });

  document.getElementById("btn-run-verify").addEventListener("click", runVerification);

  async function runVerification() {
    if ((!verifyFileA && !verifyPathA) || (!verifyFileB && !verifyPathB)) {
      alert("Please select or upload both Image A and Image B.");
      return;
    }

    const formData = new FormData();
    if (verifyFileA) formData.append("file_a", verifyFileA);
    else formData.append("path_a", verifyPathA);

    if (verifyFileB) formData.append("file_b", verifyFileB);
    else formData.append("path_b", verifyPathB);

    try {
      const res = await fetch("/api/verify", { method: "POST", body: formData });
      const data = await res.json();

      const verdictBox = document.getElementById("verify-verdict-box");
      const badge = document.getElementById("verify-badge");
      const scoreText = document.getElementById("verify-score-text");
      const progressFill = document.getElementById("verify-progress-fill");
      const thrLabel = document.getElementById("verify-thr-label");
      const explanation = document.getElementById("verify-explanation");

      verdictBox.style.display = "block";
      thrLabel.textContent = data.operating_threshold;
      scoreText.textContent = data.cosine_similarity;
      progressFill.style.width = Math.min(100, Math.max(0, data.similarity_percent)) + "%";

      if (data.is_same_design) {
        badge.textContent = "MATCH (SAME DESIGN)";
        badge.className = "verdict-badge verdict-match";
        progressFill.style.background = "linear-gradient(90deg, #10b981, #059669)";
      } else {
        badge.textContent = "NON-MATCH (DIFFERENT)";
        badge.className = "verdict-badge verdict-non-match";
        progressFill.style.background = "linear-gradient(90deg, #f43f5e, #e11d48)";
      }

      explanation.textContent = data.explanation;
    } catch (err) {
      console.error(err);
      alert("Verification failed: " + err.message);
    }
  }

  // -------------------------------------------------------------
  // Tab 3: Color Invariance Stress Lab
  // -------------------------------------------------------------
  document.getElementById("btn-run-stress").addEventListener("click", runStressTest);

  async function runStressTest() {
    const container = document.getElementById("colorways-container");
    const matrixBox = document.getElementById("similarity-matrix-box");
    container.innerHTML = `<span class="text-dim">Generating colorway variants...</span>`;
    matrixBox.innerHTML = `<span class="text-dim">Computing hyperspherical distances...</span>`;

    const formData = new FormData();
    if (selectedQueryPath) formData.append("image_path", selectedQueryPath);

    try {
      const res = await fetch("/api/stress_test", { method: "POST", body: formData });
      const data = await res.json();

      // Render colorway thumbnails
      container.innerHTML = "";
      data.colorway_variants.forEach(cw => {
        const item = document.createElement("div");
        item.className = "colorway-item";
        item.innerHTML = `
          <img src="data:image/jpeg;base64,${cw.base64}" alt="${cw.label}">
          <span>${cw.label}</span>
        `;
        container.appendChild(item);
      });

      // Render Similarity Matrix Table
      let tableHtml = `<table class="data-table" style="text-align:center;"><thead><tr><th></th>`;
      for (let i = 0; i < data.similarity_matrix.length; i++) {
        tableHtml += `<th>CW ${i}</th>`;
      }
      tableHtml += `</tr></thead><tbody>`;

      data.similarity_matrix.forEach((row, rIdx) => {
        tableHtml += `<tr><td><strong>CW ${rIdx}</strong></td>`;
        row.forEach((val, cIdx) => {
          const bg = rIdx === cIdx ? "rgba(99, 102, 241, 0.25)" : (val >= 0.75 ? "rgba(16, 185, 129, 0.2)" : "rgba(255, 255, 255, 0.05)");
          const col = rIdx === cIdx ? "var(--text-main)" : (val >= 0.75 ? "var(--accent-emerald)" : "var(--accent-cyan)");
          tableHtml += `<td style="background:${bg}; color:${col}; font-weight:600; font-family:var(--font-mono);">${val.toFixed(3)}</td>`;
        });
        tableHtml += `</tr>`;
      });
      tableHtml += `</tbody></table>`;
      matrixBox.innerHTML = tableHtml;

      document.getElementById("stress-mean-sim").textContent = data.mean_cross_colorway_similarity;
      document.getElementById("stress-assessment").textContent = data.invariance_rating;

    } catch (err) {
      console.error(err);
      container.innerHTML = `<span class="text-crimson">Failed to run stress test.</span>`;
    }
  }

  // -------------------------------------------------------------
  // Tab 4: Evaluation Protocol & Metrics
  // -------------------------------------------------------------
  async function loadEvaluationMetrics() {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      if (data.identification) {
        document.getElementById("eval-top1").textContent = data.identification.top1_accuracy_percent + "%";
        document.getElementById("eval-top5").textContent = data.identification.top5_accuracy_percent + "%";
        document.getElementById("eval-map").textContent = data.identification.mean_average_precision_percent + "%";
      }
      if (data.verification) {
        document.getElementById("eval-auc").textContent = data.verification.auc;
        document.getElementById("eval-eer").textContent = data.verification.eer_percent + "%";
      }
      if (data.color_invariance_diagnostic) {
        const gap = data.color_invariance_diagnostic.color_bias_gap;
        document.getElementById("eval-gap").textContent = (gap >= 0 ? "+" : "") + gap;
      }
    } catch (err) {
      console.error("Failed to load metrics:", err);
    }
  }

  // -------------------------------------------------------------
  // Tab 5: Efficiency Benchmark
  // -------------------------------------------------------------
  async function loadEfficiencyReport() {
    try {
      const res = await fetch("/api/efficiency");
      const data = await res.json();
      if (data.parameters) {
        document.getElementById("eff-params").textContent = data.parameters.formatted_param_count;
      }
      if (data.computation) {
        document.getElementById("eff-flops").textContent = data.computation.gflops_per_image + " GFLOPs";
      }
      if (data.latency_and_throughput) {
        document.getElementById("eff-latency").textContent = data.latency_and_throughput.mean_latency_ms + " ms";
      }
    } catch (err) {
      console.error("Failed to load efficiency:", err);
    }
  }

  document.getElementById("btn-live-benchmark").addEventListener("click", async () => {
    const btn = document.getElementById("btn-live-benchmark");
    btn.textContent = "Benchmarking 20 iterations...";
    btn.disabled = true;

    try {
      const res = await fetch("/api/benchmark_live", { method: "POST" });
      const data = await res.json();
      document.getElementById("eff-latency").textContent = data.mean_latency_ms + " ms";
      alert(`Benchmark Completed!\nDevice: ${data.device}\nMean Latency: ${data.mean_latency_ms} ms\nThroughput: ${data.throughput_fps} FPS`);
    } catch (err) {
      alert("Benchmark error: " + err.message);
    } finally {
      btn.textContent = "Run Live Benchmark Now";
      btn.disabled = false;
    }
  });

  // Run initial stress test automatically once sample queries load
  setTimeout(() => {
    runStressTest();
  }, 1000);
});
