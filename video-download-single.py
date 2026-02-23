import json
import os
from pathlib import Path, PurePosixPath
from datasets import load_dataset

# Dataset specific constants
DATASET_NAME = "lmms-lab/ActivityNetQA"
JSONS_DIR = os.path.join("data", "activitynet", "jsons")
OUTPUT_DIR = os.path.join("data", "activitynet", "videos")

def load_targets_from_json(json_path):
    """
    Expects a JSON list where each item has a 'video' or 'image' key 
    pointing to a filename like 'v_12345.mp4'.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    targets = {}
    for item in entries:
        # ActivityNetQA items usually use 'video', but checking 'image' for parity with your snippet
        target_filename = item.get("video") or item.get("image")
        if not target_filename:
            continue
            
        # Clean the filename and use the stem (ID) as the lookup key
        target_key = os.path.splitext(os.path.basename(target_filename))[0]
        targets[target_key] = target_filename
    return targets

def save_video_bytes(relative_path, video_data):
    """
    Handles both raw bytes or the 'bytes' key within a dataset video object.
    """
    out_path = os.path.join(OUTPUT_DIR, *PurePosixPath(relative_path).parts)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Extract bytes if it's a dict (common in HF video datasets)
    content = video_data['bytes'] if isinstance(video_data, dict) and 'bytes' in video_data else video_data
    
    with open(out_path, "wb") as f:
        f.write(content)
    return out_path

def process_json_file(json_path):
    json_filename = os.path.basename(json_path)
    # ActivityNetQA doesn't use sub-dirs for subsets the same way VISTA does, 
    # but we'll keep the logic if you're targeting specific splits.
    targets = load_targets_from_json(json_path)

    if not targets:
        print(f"[{json_filename}] No valid video entries found.")
        return

    print(f"[{json_filename}] targets={len(targets)}")
    
    # Note: ActivityNetQA usually has 'train' and 'test' splits
    dataset = load_dataset(DATASET_NAME, streaming=True, split="train")
    remaining_keys = set(targets.keys())

    for i, entry in enumerate(dataset, 1):
        if i % 500 == 0:
            found_count = len(targets) - len(remaining_keys)
            print(f"[{json_filename}] scanned={i}, found={found_count}/{len(targets)}")

        # ActivityNetQA schema uses 'video_id' or 'video' 
        current_key = entry.get("video_id") or entry.get("video")
        
        # If the key is a full path, strip it to match our target_key
        if current_key:
            current_key = os.path.splitext(os.path.basename(str(current_key)))[0]

        if current_key not in remaining_keys:
            continue

        # In ActivityNetQA, the video data is often in the 'video' column
        video_payload = entry.get("video")
        if video_payload:
            target_filename = targets[current_key]
            saved_path = save_video_bytes(target_filename, video_payload)
            print(f"[{json_filename}] Saved: {saved_path}")
            remaining_keys.remove(current_key)
        else:
            print(f"[{json_filename}] Missing video data for key={current_key}")

        if not remaining_keys:
            break

    if remaining_keys:
        print(f"[{json_filename}] Not found in stream: {len(remaining_keys)}")
    else:
        print(f"[{json_filename}] Completed. All targets saved.")

def main():
    if not os.path.isdir(JSONS_DIR):
        print(f"JSON directory not found: {JSONS_DIR}")
        # Create it to be helpful
        os.makedirs(JSONS_DIR, exist_ok=True)
        return

    json_paths = sorted(Path(JSONS_DIR).glob("*.json"))
    if not json_paths:
        print(f"No JSON files found in: {JSONS_DIR}")
        return

    for json_path in json_paths:
        process_json_file(str(json_path))

if __name__ == "__main__":
    main()