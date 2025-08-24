import json

input_path = "LOL/pretrain/pretrain_output_fixed2.jsonl"
output_path = "LOL/pretrain/pretrain_output_final1.jsonl"

bad_qids = []

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        qid = data["qid"]
        clip_ids = data.get("relevant_clip_ids", [])
        duration = data.get("duration", 0)
        max_index = int(duration // 2)

        # 檢查是否有任何 index 超過最大值
        if any(idx >= max_index for idx in clip_ids):
            bad_qids.append(qid)
        else:
            outfile.write(json.dumps(data) + "\n")

# 印出被刪除的 qid
print("❌ QIDs with out-of-bounds clip_ids:")
for qid in bad_qids:
    print(qid)

print(f"\n✅ Cleaned file saved to: {output_path}")
