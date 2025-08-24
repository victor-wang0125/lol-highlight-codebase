import os
import json

jsonl_path = "LOL/pretrain/pretrain_output_final.jsonl"
query_feat_dir = "LOL/pretrain/clip_query_features"

missing_qids = []

with open(jsonl_path, "r") as f:
    for line in f:
        data = json.loads(line)
        qid = data["qid"]
        fname = f"qid{qid}.npz"
        fpath = os.path.join(query_feat_dir, fname)
        if not os.path.exists(fpath):
            missing_qids.append(qid)

print(f"❌ Missing query features for {len(missing_qids)} samples:")
for qid in missing_qids:
    print(qid)