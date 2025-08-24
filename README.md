# VideoLights 專案執行指南

本專案整理了論文程式碼在本機/伺服器上的**完整執行流程**，涵蓋：環境設定、特徵提取、模型微調、合成資料生成、快取清理、LLaVA 推論，以及常用的 FFmpeg 影片處理指令。你可以將本檔直接作為 GitHub 專案的 `README.md`。

> ⚠️ 注意：文中路徑（如 `/mnt/e/...`、`/mnt/c/...`、`G:\...`）與模型檔名請依你的實際環境調整。

---

## 目錄
- [環境設定](#環境設定)
- [特徵提取 (Feature Extraction)](#特徵提取-feature-extraction)
- [模型微調 (Fine-tuning)](#模型微調-fine-tuning)
- [合成資料生成 (Synthetic Data Generation)](#合成資料生成-synthetic-data-generation)
- [模型快取清理 (HuggingFace Cache)](#模型快取清理-huggingface-cache)
- [LLaVA 推論](#llava-推論)
- [影片處理 (FFmpeg/FFprobe)](#影片處理-ffmpegffprobe)
- [預訓練 (Pre-training)](#預訓練-pre-training)
- [備註](#備註)

---

## 環境設定
```bash
conda activate video_lights
export PYTHONPATH=$PYTHONPATH:/mnt/c/code/VideoLights
```

---

## 特徵提取 (Feature Extraction)

### 1) SlowFast 視覺特徵
```bash
# 產生影片路徑列表
python extract_feature/slowfast/extract_feature/gather_video_paths.py

# 抽取 SlowFast 特徵
python extract_feature/slowfast/extract_feature/extract.py \
  --dataflow \
  --csv extract_feature/slowfast/output/csv/slowfast_info.csv \
  --batch_size 45 \
  --num_decoding_thread 4 \
  --clip_len 2 \
  TEST.CHECKPOINT_FILE_PATH /models/SLOWFAST_8x8_R50.pkl
```

### 2) Query（文字/多模態）特徵
```bash
# 文字改寫 / 同義改寫（如使用 OpenAI 流程）
python extract_feature/openai_paraphraser.py

# CLIP Query 特徵
python extract_feature/extract_query_clip_features.py

# BLIP（QVH）Query 特徵
python extract_feature/extract_query_blip_features_qvhl.py
```

---

## 模型微調 (Fine-tuning)

```bash
# 預訓練權重路徑（一般）
export PRETRAIN_CHECKPOINT_PATH="results/pretrain/model_best.ckpt"

# 預訓練權重路徑（LOL 資料專用）
export PRETRAIN_CHECKPOINT_PATH="LOL/pretrain/results/V(sf_clip_blip)_T(clip_blip)_video(26)/model_best.ckpt"

# 啟動微調訓練
bash video_lights/scripts/qvhl/train.sh
```

---

## 合成資料生成 (Synthetic Data Generation)

### 1) 官方 BLIP JSONL 生成
```bash
python extract_feature/pretrain_data_generator_using_blip2.py
```

### 2) Gemini JSONL 生成
```bash
python LOL/pretrain/create_gemini_jsonl.py \
  --image_dir /mnt/e/coding/nkust_paper/2023_spring_LEC/W3D1_W3D3_frames \
  --input_jsonl LOL/pretrain/W3D1_1_3_6.jsonl \
  --output_jsonl LOL/pretrain/pretrain_output1.jsonl

# 修正/驗證 JSONL
python LOL/pretrain/bugjsonl1.py
python LOL/pretrain/jsonfix2.py
python LOL/pretrain/clip_ids_check3.py
```

### 3) Query 特徵生成（預訓練）
```bash
python extract_feature/extract_pretrain_query_features.py
```

### 4) JSONL Score 包裝（將 score 外層多套一層 list）
```bash
python LOL/pretrain/jsonfix.py
```

---

## 模型快取清理 (HuggingFace Cache)

```bash
# 確認 HuggingFace 模型快取位置
cd ~/.cache/huggingface/hub
du -h --max-depth=1 .

# 刪除指定模型
rm -rf ./models--google--flan-t5-xl
```

若刪除失敗，可嘗試以下方式（Ubuntu + Windows WSL）：

**Ubuntu**
```bash
sudo fstrim -av
```

**Windows（以系統管理員身分開啟 PowerShell）**
```powershell
diskpart
select vdisk file="C:\Users\victo\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```

---

## LLaVA 推論

```bash
cd LLaVA
conda activate llava
# 單張圖片推論（eval）：請依實際檔名調整
python run_llava_infer.py
# 原始 txt 註記為「run_llava_infer.p（單張圖片）（eval）」；若你的檔名為 .py，請改用上面指令。
```

---

## 影片處理 (FFmpeg/FFprobe)

```bash
# 檢查影片（輸出媒體資訊）
ffmpeg -i /mnt/e/coding/nkust_paper/2023_spring_LEC/W2D2/W2D2_1/W2D2_1_CUT/fixed_900_1050.mp4

# 修改影片 FPS 為 60（重新編碼）
ffmpeg -i /mnt/e/coding/nkust_paper/2023_spring_LEC/W2D2/W2D2_1/W2D2_1_CUT/W2D2_1_900.0_1050.0.mp4 \
  -r 60 -c:v libx264 -preset fast -crf 23 -c:a copy \
  /mnt/e/coding/nkust_paper/2023_spring_LEC/W2D2/W2D2_1/W2D2_1_CUT/fixed_900_1050.mp4

# 計算畫面幀數
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames \
  -of default=nokey=1:noprint_wrappers=1 output_9000frames.mp4

# 依時間範圍剪輯影片（無損 copy）
ffmpeg -ss 00:01:12 -to 00:11:12 -i G:\GROUPS_W1D2\GW1D2.mp4 -c copy \
  E:\coding\nkust_paper\2023_spring_LEC\GW1D2\GW1D2_1.mp4
```

---

## 預訓練 (Pre-training)
```bash
bash video_lights/scripts/pretrain/pretrain_sf_clip_blip.sh
```

---

## 備註
- 上述流程摘自原始 `執行.txt` 並做格式化彙整，方便快速上手與復現。
- 若需更完整的說明（資料夾結構、模型下載連結、指標定義、實驗結果、引用文獻等），可在此 README 後續段落補充。
- 若你要在 **GitHub Actions** 或 **雲端環境**自動化執行，建議再新增：
  - 需求檔（`requirements.txt` / `environment.yml`）
  - 範例設定（`config/*.yaml`）
  - 範例資料與輸出目錄結構（`data/`, `outputs/`）
