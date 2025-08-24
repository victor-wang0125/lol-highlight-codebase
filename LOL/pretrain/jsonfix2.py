import json

input_path = "LOL/pretrain/pretrain_output_fixed.jsonl"
output_path = "LOL/pretrain/pretrain_output_fixed2.jsonl"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        scores = data.get("saliency_scores", [])
        # 如果是一個 list of float 而不是 list of list
        if scores and isinstance(scores[0], float):
            data["saliency_scores"] = [[s] for s in scores]
        # 確保長度一致
        if len(data["relevant_clip_ids"]) != len(data["saliency_scores"]):
            print(f"⚠️ mismatch: {data['qid']}")
            continue  # or fix/remove
        outfile.write(json.dumps(data) + "\n")
