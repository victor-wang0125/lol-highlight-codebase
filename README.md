# VideoLights 專案執行指南

本專案整理了論文程式碼在本機/伺服器上的**完整執行流程**，涵蓋：環境設定、特徵提取、模型微調、合成資料生成。

> ⚠️ 注意：文中路徑（如 `/mnt/e/...`、`/mnt/c/...`、`G:\...`）與模型檔名請依你的實際環境調整。

---

## 目錄
- [環境設定](#環境設定)
- [特徵提取 (Feature Extraction)](#特徵提取-feature-extraction)
- [模型微調 (Fine-tuning)](#模型微調-fine-tuning)
- [合成資料生成 (Synthetic Data Generation)](#合成資料生成-synthetic-data-generation)
- [影片處理 (FFmpeg/FFprobe)](#影片處理-ffmpegffprobe)
- [預訓練 (Pre-training)](#預訓練-pre-training)
- [備註](#備註)

---

## 快速開始
### 下載專案
```bash
git clone https://github.com/victor-wang0125/lol-highlight-codebase.git
cd lol-highlight-codebase
```
---


### 環境設定
```bash
conda create -n lol-highlight-codebase python=3.10 -y
conda activate lol-highlight-codebase
# 安裝套件
pip install -r requirements.txt
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
# 預訓練權重路徑
export PRETRAIN_CHECKPOINT_PATH="results/pretrain/model_best.ckpt"


# 啟動微調訓練
bash video_lights/scripts/qvhl/train.sh
```

---

## 合成資料生成 (Synthetic Data Generation )

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


## 預訓練 (Pre-training)
```bash
bash video_lights/scripts/pretrain/pretrain_sf_clip_blip.sh
```

---

## 備註
