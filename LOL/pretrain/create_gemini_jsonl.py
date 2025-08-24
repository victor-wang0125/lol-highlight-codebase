import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 BLIP Image-Text Matching 模型
print("🔄 Loading BLIP ITM model...")
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip_image_text_matching",
    model_type="base",
    is_eval=True,
    device=device
)
print("✅ Model loaded.")

# 讀取 JSONL 檔案，每行一筆 JSON
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

# 主處理函數
@torch.no_grad()
def process_jsonl_and_images(image_dir, input_jsonl, output_jsonl):
    data = load_jsonl(input_jsonl)
    results = []

    for item in tqdm(data, desc="Processing images"):
        filename = item["filename"]
        query = item["description"]

        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue

        # 圖片與文字處理
        try:
            image_tensor = vis_processors["eval"](Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            text_tensor = txt_processors["eval"](query)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
            continue

        # 執行圖文匹配（ITM）
        itm_output = model({"image": image_tensor, "text_input": text_tensor}, match_head="itm")
        score = F.softmax(itm_output, dim=1)[:, 1].item()
        scaled_score = max(score * 2.5, 1.0)
        saliency_score = torch.clamp(torch.exp(torch.tensor(scaled_score)), min=2.0).item()

        # 解析 metadata
        qid = os.path.splitext(filename)[0]
        parts = qid.split("_")
        if len(parts) != 6:
            print(f"⚠️ Unexpected filename format: {filename}")
            continue

        vid = "_".join(parts[:4])
        start = float(parts[4])
        end = float(parts[5])
        duration = 150.0  # 可依情況修改

        clip_ids = list(range(int(start // 2), int(end // 2)))

        results.append({
            "qid": qid,
            "query": query,
            "vid": vid,
            "duration": duration,
            "split": "train",
            "relevant_windows": [[start, end]],
            "relevant_clip_ids": clip_ids,
            "saliency_scores": [saliency_score for _ in clip_ids]
        })

    # 輸出 JSONL 檔
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n✅ Saved {len(results)} entries to: {output_jsonl}")

# 執行主程式
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate VideoLights pretrain JSONL using BLIP ITM.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory with images")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to .jsonl file with filename and description")
    parser.add_argument("--output_jsonl", type=str, default="output.jsonl", help="Path to output .jsonl")

    args = parser.parse_args()

    process_jsonl_and_images(
        image_dir=args.image_dir,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl
    )
