import json

def check_jsonl_saliency_errors(jsonl_path):
    error_entries = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                qid = data.get("qid", f"line_{line_num}")
                relevant_clip_ids = data["relevant_clip_ids"]
                duration = data["duration"]
                ctx_l = duration // 2

                for clip_id in relevant_clip_ids:
                    if clip_id >= ctx_l:
                        print(f"[⚠️ ClipID 超出 ctx_l] qid={qid}, clip_id={clip_id}, ctx_l={ctx_l}")
                        error_entries.append({
                            "qid": qid,
                            "clip_id": clip_id,
                            "ctx_l": ctx_l,
                            "relevant_clip_ids": relevant_clip_ids,
                            "duration": duration
                        })
                        break  # 如果一筆有錯誤，不重複印出多次

            except Exception as e:
                print(f"[❌ JSON Parse Error] line {line_num}: {e}")
                continue

    print(f"\n🔍 共發現 {len(error_entries)} 筆有問題的資料。")
    return error_entries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Path to the input .jsonl file")
    args = parser.parse_args()

    error_list = check_jsonl_saliency_errors(args.jsonl_path)

    # Optional: 輸出錯誤清單為 JSON 檔
    if error_list:
        output_path = "saliency_error_report.json"
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(error_list, out_file, indent=2, ensure_ascii=False)
        print(f"✅ 詳細錯誤資訊已輸出到: {output_path}")
