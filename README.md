# 英雄聯盟賽事影片精華偵測與片段檢索 Codebase

此專案為碩士論文研究程式碼，研究主題為《英雄聯盟》電子競技賽事影片中的  
**Moment Retrieval (片段檢索)** 與 **Highlight Detection (精華偵測)** 任務。  

本專案基於 **VideoLights** 架構，結合 **SlowFast 視訊特徵** 與 **CLIP / BLIP 語言特徵**，  
並支援 **Gemini、BLIP-2、LLaVA** 等大型視覺語言模型生成的資料進行預訓練。

---

## 📌 環境設定

建立與啟用環境：
```bash
conda activate video_lights
export PYTHONPATH=$PYTHONPATH:/mnt/c/code/VideoLights
