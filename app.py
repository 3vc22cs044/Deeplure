"""
FastAPI Server for Color-Invariant Saree Recognition System.
Provides REST APIs and serves the interactive Web Dashboard.
"""

import os
import io
import json
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from src.inference import SareeInferenceEngine
from src.benchmark import benchmark_latency, count_parameters

app = FastAPI(title="Color-Invariant Saree Recognition API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Engine
engine = SareeInferenceEngine(
    checkpoint_path="checkpoints/cis_net_best.pt",
    splits_path="data/splits.json"
)

# Mount static files
os.makedirs("web", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")
app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
def read_root():
    return FileResponse("web/index.html")


@app.get("/api/gallery")
def get_gallery():
    """Returns list of reference saree designs in the gallery."""
    items = []
    for item in engine.gallery_items:
        items.append({
            "gallery_id": item["gallery_id"],
            "design_id": item["design_id"],
            "category": item["category"],
            "source": item.get("source", "DeepLure"),
            "image_url": "/" + item["path"].replace("\\", "/"),
        })
    return {"gallery": items, "total_count": len(items)}


@app.get("/api/sample_queries")
def get_sample_queries():
    """Returns categorized sample queries for quick 1-click testing."""
    if not os.path.exists("data/splits.json"):
        return {"samples": []}

    with open("data/splits.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("query_items", [])
    grouped = {
        "cross_colorway_match": [],
        "same_palette_diff_motif": [],
        "open_set_novel": []
    }

    for q in queries:
        t = q["query_type"]
        if t in grouped and len(grouped[t]) < 8:
            grouped[t].append({
                "query_id": q["query_id"],
                "design_id": q["design_id"],
                "category": q["category"],
                "query_type": t,
                "target_gallery_id": q.get("target_gallery_id"),
                "image_url": "/" + q["path"].replace("\\", "/"),
                "file_path": q["path"].replace("\\", "/")
            })

    return grouped


@app.post("/api/identify")
async def identify_saree(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    top_k: int = Form(5)
):
    """
    1:N Saree Identification.
    Matches query saree against the gallery database.
    """
    try:
        if file is not None and file.filename:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        elif image_path:
            # Clean path
            cleaned_p = image_path.lstrip("/")
            if not os.path.exists(cleaned_p):
                raise HTTPException(status_code=404, detail=f"Image not found at {cleaned_p}")
            image = Image.open(cleaned_p)
        else:
            raise HTTPException(status_code=400, detail="Provide an image file or image_path.")

        # Run identification
        results = engine.identify(image, top_k=top_k)
        
        # Also generate structural heatmap to show how color is stripped
        heatmap_b64 = engine.get_structural_heatmap_base64(image)
        results["query_structural_heatmap"] = heatmap_b64

        # Fix relative image URLs for UI
        for m in results.get("ranked_matches", []):
            m["image_url"] = "/" + m["image_path"]

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify")
async def verify_sarees(
    file_a: Optional[UploadFile] = File(None),
    file_b: Optional[UploadFile] = File(None),
    path_a: Optional[str] = Form(None),
    path_b: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None)
):
    """
    1:1 Saree Verification.
    Determines whether Image A and Image B carry the same structural design.
    """
    try:
        # Load Image A
        if file_a is not None and file_a.filename:
            image_a = Image.open(io.BytesIO(await file_a.read()))
        elif path_a:
            image_a = Image.open(path_a.lstrip("/"))
        else:
            raise HTTPException(status_code=400, detail="Image A is required.")

        # Load Image B
        if file_b is not None and file_b.filename:
            image_b = Image.open(io.BytesIO(await file_b.read()))
        elif path_b:
            image_b = Image.open(path_b.lstrip("/"))
        else:
            raise HTTPException(status_code=400, detail="Image B is required.")

        result = engine.verify(image_a, image_b, custom_threshold=threshold)
        result["heatmap_a"] = engine.get_structural_heatmap_base64(image_a)
        result["heatmap_b"] = engine.get_structural_heatmap_base64(image_b)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stress_test")
async def stress_test_color_invariance(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None)
):
    """
    Color Invariance Stress-Test Lab.
    Recolors the motif into multiple authentic palettes and computes cross-colorway similarity.
    """
    try:
        if file is not None and file.filename:
            image = Image.open(io.BytesIO(await file.read()))
        elif image_path:
            image = Image.open(image_path.lstrip("/"))
        else:
            # Default to first gallery item
            if engine.gallery_items:
                image = Image.open(engine.gallery_items[0]["path"])
            else:
                raise HTTPException(status_code=400, detail="No image provided.")

        return engine.stress_test_colorways(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
def get_metrics():
    """Returns evaluation metrics, ROC curve, and CMC curve."""
    metrics_path = "artifacts/evaluation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return {"error": "Evaluation metrics not found."}


@app.get("/api/efficiency")
def get_efficiency():
    """Returns efficiency report (parameters, FLOPs, latency, memory)."""
    report_path = "artifacts/efficiency_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return {"error": "Efficiency report not found."}


@app.post("/api/benchmark_live")
def benchmark_live():
    """Executes live latency benchmark on current server."""
    mean_lat, med_lat, p95_lat, fps = benchmark_latency(engine.model, device=engine.device, num_warmup=5, num_runs=20)
    return {
        "device": str(engine.device),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(med_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 1),
    }


if __name__ == "__main__":
    import uvicorn
    print("Launching Saree Recognition Web App on http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
