# 📰🤖 Fake News & Deepfake Detector (Gen-AI)

An **AI-powered detection system** that helps identify misinformation in **news articles** and detects **deepfake videos**.  
It uses **LLM-powered claim verification** for textual content and **deepfake detection models** for video analysis.  

🚨 In today’s world, misinformation spreads faster than ever. This tool empowers people to **fact-check content** and **protect themselves from AI-generated deception**.  



## ✨ Features

✅ **Article Verification** – Extracts claims from articles and verifies them against trusted sources.  
✅ **Deepfake Video Detection** – Analyzes video frames for manipulations & face artifacts.  
✅ **Multi-Modal AI** – Combines text + video AI pipelines.  
✅ **Streamlit Dashboard** – Simple, interactive web UI for uploading and checking content.  
✅ **Modular Design** – Easy to extend with stronger LLMs or detection models.  



## 🛠️ Tech Stack

- **LLM** → Claim verification using OpenAI / any LLM API  
- **Whisper / OCR (optional)** → Extract speech-to-text from videos  
- **Deepfake Detection Models** → (Face artifacts, CNN-based classifier, placeholder ready)  
- **Streamlit** → Frontend dashboard  
- **Python** → Glue code for article & video pipelines  



## 📂 Project Structure

```
fake-news-deepfake-detector-gen-ai/
│── .env.example              # API keys
│── requirements.txt          # Dependencies
│── README.md                 # Documentation
│── streamlit_app.py          # Streamlit UI
└── src/
    ├── __init__.py
    ├── config.py             # Env + settings
    ├── llm.py                # LLM wrapper (claim verification)
    ├── article_checker.py    # Article ingestion, claim extraction & verification
    ├── video_checker.py      # Deepfake detection pipeline
    ├── models.py             # Model loaders (placeholders)
    └── utils.py              # Helpers (downloader, cleaners)
```



## 🚀 Getting Started

### 1️⃣ Clone the repo  
```bash
git clone https://github.com/your-username/fake-news-deepfake-detector-gen-ai.git
cd fake-news-deepfake-detector-gen-ai
```

### 2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Setup environment variables  
Create a `.env` file from the provided `.env.example`:  
```
OPENAI_API_KEY=your_key_here
```

### 4️⃣ Run the app  
```bash
streamlit run streamlit_app.py
```



## 🖥️ Usage

1. **For Articles 📰**
   - Paste a **news article link** or **raw text**.  
   - The system extracts claims and checks them via the LLM.  
   - Output → *Fact-check summary* with confidence score.  

2. **For Videos 🎥**
   - Upload a short video file.  
   - Frames are analyzed for **deepfake artifacts**.  
   - Output → *Deepfake likelihood score + evidence frames*.  



## 📊 Example Output

✅ Article:  
```
Claim: "XYZ company announced free vaccines."
Verdict: ❌ Misleading – no verified source found.
Confidence: 83%
```

🎥 Video:  
```
Result: 72% probability of deepfake detected.
Key Evidence: Blinking inconsistencies, facial warping.
```

