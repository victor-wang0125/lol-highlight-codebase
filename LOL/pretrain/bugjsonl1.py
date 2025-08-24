import json

# 原始檔案路徑
input_path = "LOL/pretrain/pretrain_output.jsonl"
# 過濾後輸出的新檔案
output_path = "LOL/pretrain/pretrain_output_fixed.jsonl"

bad_qids = []

def is_valid_entry(clip_ids, saliency_scores):
    if not clip_ids or not saliency_scores:
        return False
    # 如果 saliency_scores 是 list of list
    if isinstance(saliency_scores[0], list):
        return len(clip_ids) == len(saliency_scores)
    # 如果 saliency_scores 是 list of floats
    if isinstance(saliency_scores[0], float) or isinstance(saliency_scores[0], int):
        return len(clip_ids) == len(saliency_scores)
    return False

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        clip_ids = data.get("relevant_clip_ids", [])
        scores = data.get("saliency_scores", [])

        if is_valid_entry(clip_ids, scores):
            outfile.write(json.dumps(data) + "\n")
        else:
            bad_qids.append(data.get("qid", "UNKNOWN"))

print("❌ Found bad entries with these QIDs:")
for qid in bad_qids:
    print(qid)

print(f"\n✅ Cleaned file saved to: {output_path}")
