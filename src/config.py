# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "6"))
VIDEO_MODEL_WEIGHTS_PATH = os.getenv("VIDEO_MODEL_WEIGHTS_PATH", "./weights/video_deepfake_model.pt")
