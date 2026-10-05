"""
Data preparation and curation pipeline for Color-Invariant Saree Recognition.
Integrates DeepLure Handloom Sarees and Indian Saree Patterns into structured
design classes, synthesized colorways, and a documented Train/Gallery/Query split.
"""

import os
import glob
import json
import random
import shutil
from PIL import Image
from tqdm import tqdm
from src.colorway_engine import generate_colorways_for_motif, TEXTILE_PALETTES, apply_palette_transfer

# Fix random seed for reproducibility
random.seed(42)

def prepare_dataset(
    output_dir: str = "data/saree_dataset",
    num_train_designs: int = 40,
    num_gallery_designs: int = 25,
    num_unseen_designs: int = 15,
    colorways_per_design: int = 6,
):
    """
    Builds a curated dataset of saree designs with multi-colorway variations.
    Saves clean train, gallery, and query splits with explicit metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gallery"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "query"), exist_ok=True)

    # 1. Collect candidate images from Kaggle and DeepLure
    kaggle_base = r"C:\Users\USER\.cache\kagglehub\datasets\div456\indian-saree-patterns\versions\1"
    deeplure_base = "data/deeplure_corpus"

    candidates = []
    
    # Kaggle categories
    categories = ["Banarasi", "Bandhani", "Ikat", "Pichwai"]
    for cat in categories:
        for split in ["train", "valid", "test"]:
            folder = os.path.join(kaggle_base, split, cat)
            if os.path.exists(folder):
                imgs = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
                for img in imgs:
                    candidates.append({"path": img, "category": cat, "source": "kaggle"})

    # DeepLure handloom sarees
    dl_imgs = glob.glob(os.path.join(deeplure_base, "*", "*.jpg")) + glob.glob(os.path.join(deeplure_base, "*", "*.png"))
    for img in dl_imgs:
        candidates.append({"path": img, "category": "Handloom", "source": "deeplure"})

    print(f"Total raw candidate images found: {len(candidates)}")
    random.shuffle(candidates)

    total_designs_needed = num_train_designs + num_gallery_designs + num_unseen_designs
    if len(candidates) < total_designs_needed:
        print(f"Warning: requested {total_designs_needed} designs, but only {len(candidates)} candidates.")
        total_designs_needed = len(candidates)

    selected_candidates = candidates[:total_designs_needed]

    # Partition candidates into train, gallery, and unseen sets
    train_candidates = selected_candidates[:num_train_designs]
    gallery_candidates = selected_candidates[num_train_designs : num_train_designs + num_gallery_designs]
    unseen_candidates = selected_candidates[num_train_designs + num_gallery_designs :]

    metadata = {
        "train_designs": [],
        "gallery_items": [],
        "query_items": [],
    }

    # 2. Process Training Set (each design has multiple colorways)
    print("Generating Training set colorways...")
    for idx, item in enumerate(tqdm(train_candidates, desc="Train")):
        design_id = f"train_design_{idx:03d}_{item['category']}"
        design_dir = os.path.join(output_dir, "train", design_id)
        os.makedirs(design_dir, exist_ok=True)

        try:
            with Image.open(item["path"]) as orig_im:
                orig_im = orig_im.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
                colorways = generate_colorways_for_motif(orig_im, num_colorways=colorways_per_design)

                cw_paths = []
                for c_idx, cw in enumerate(colorways):
                    cw_filename = f"cw_{c_idx:02d}.jpg"
                    cw_path = os.path.join(design_dir, cw_filename)
                    cw.save(cw_path, quality=90)
                    cw_paths.append(cw_path)

                metadata["train_designs"].append({
                    "design_id": design_id,
                    "category": item["category"],
                    "source": item["source"],
                    "num_colorways": len(cw_paths),
                    "dir": design_dir,
                })
        except Exception as e:
            print(f"Skipping {item['path']}: {e}")

    # 3. Process Gallery (Reference Database of Known Designs)
    # Stored with canonical colorway (original)
    print("Populating Gallery (Reference Database)...")
    for idx, item in enumerate(tqdm(gallery_candidates, desc="Gallery")):
        design_id = f"design_{idx:03d}_{item['category']}"
        try:
            with Image.open(item["path"]) as orig_im:
                orig_im = orig_im.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
                colorways = generate_colorways_for_motif(orig_im, num_colorways=5)

                # Canonical reference image (Colorway 0)
                gallery_filename = f"{design_id}_canonical.jpg"
                gallery_path = os.path.join(output_dir, "gallery", gallery_filename)
                colorways[0].save(gallery_path, quality=92)

                gallery_record = {
                    "gallery_id": f"gal_{idx:03d}",
                    "design_id": design_id,
                    "category": item["category"],
                    "source": item["source"],
                    "path": gallery_path,
                }
                metadata["gallery_items"].append(gallery_record)

                # 4. Create Query Probes for this gallery design:
                # Type A: True Positives - same design in drastically different colorways (Colorways 1, 2, 3)
                for cw_idx in range(1, min(4, len(colorways))):
                    q_filename = f"q_match_{design_id}_cw{cw_idx}.jpg"
                    q_path = os.path.join(output_dir, "query", q_filename)
                    colorways[cw_idx].save(q_path, quality=90)

                    metadata["query_items"].append({
                        "query_id": f"q_pos_{idx:03d}_{cw_idx}",
                        "design_id": design_id,
                        "query_type": "cross_colorway_match",
                        "target_gallery_id": gallery_record["gallery_id"],
                        "category": item["category"],
                        "path": q_path,
                    })

                # Type B: Hard Distractors - take an adjacent/different design, recolor it with the EXACT SAME
                # palette as this design's canonical colorway!
                other_idx = (idx + 1) % len(gallery_candidates)
                other_item = gallery_candidates[other_idx]
                with Image.open(other_item["path"]) as other_im:
                    other_im = other_im.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
                    # Apply canonical palette of current design to other design
                    canonical_palette = TEXTILE_PALETTES[idx % len(TEXTILE_PALETTES)]
                    distractor_im = apply_palette_transfer(other_im, canonical_palette)
                    
                    distractor_filename = f"q_distractor_for_{design_id}.jpg"
                    distractor_path = os.path.join(output_dir, "query", distractor_filename)
                    distractor_im.save(distractor_path, quality=90)

                    metadata["query_items"].append({
                        "query_id": f"q_distractor_{idx:03d}",
                        "design_id": f"design_{other_idx:03d}_{other_item['category']}",
                        "query_type": "same_palette_diff_motif",  # Hard negative!
                        "target_gallery_id": None,
                        "intended_confuser_gallery_id": gallery_record["gallery_id"],
                        "category": other_item["category"],
                        "path": distractor_path,
                    })

        except Exception as e:
            print(f"Error processing gallery item {item['path']}: {e}")

    # 5. Process Unseen Novel Designs for Open-Set Verification
    print("Generating Unseen Open-Set verification queries...")
    for idx, item in enumerate(tqdm(unseen_candidates, desc="Unseen")):
        design_id = f"unseen_{idx:03d}_{item['category']}"
        try:
            with Image.open(item["path"]) as orig_im:
                orig_im = orig_im.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
                colorways = generate_colorways_for_motif(orig_im, num_colorways=3)

                for cw_idx, cw in enumerate(colorways[:2]):
                    q_filename = f"q_unseen_{design_id}_cw{cw_idx}.jpg"
                    q_path = os.path.join(output_dir, "query", q_filename)
                    cw.save(q_path, quality=90)

                    metadata["query_items"].append({
                        "query_id": f"q_unseen_{idx:03d}_{cw_idx}",
                        "design_id": design_id,
                        "query_type": "open_set_novel",
                        "target_gallery_id": None,
                        "category": item["category"],
                        "path": q_path,
                    })
        except Exception as e:
            print(f"Error processing unseen item {item['path']}: {e}")

    # Save documented split metadata
    splits_file = "data/splits.json"
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset Preparation Complete!")
    print(f"Train Designs: {len(metadata['train_designs'])} designs (total images: {sum(d['num_colorways'] for d in metadata['train_designs'])})")
    print(f"Gallery Designs: {len(metadata['gallery_items'])} reference designs")
    print(f"Query Probes: {len(metadata['query_items'])} total queries")
    q_types = {}
    for q in metadata["query_items"]:
        q_types[q["query_type"]] = q_types.get(q["query_type"], 0) + 1
    for k, v in q_types.items():
        print(f"  - {k}: {v}")

    return metadata

if __name__ == "__main__":
    prepare_dataset()
