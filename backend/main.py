# ==========================================================
# FINAL main.py
# AUTO SWITCH MODELS + RETRY + CACHE + TRANSLATION + NO BLANK UI
# ==========================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import io
import os
import requests
from dotenv import load_dotenv

# ==========================================================
# ENV
# ==========================================================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ==========================================================
# APP
# ==========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================
# LANGUAGES
# ==========================================================
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "pa": "Punjabi"
}

# ==========================================================
# CACHE
# ==========================================================
CACHE_FILE = os.path.join(BASE_DIR, "ai_cache.json")

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            AI_CACHE = json.load(f)
    except:
        AI_CACHE = {}
else:
    AI_CACHE = {}

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(AI_CACHE, f, indent=2, ensure_ascii=False)

# ==========================================================
# MODEL LOAD
# ==========================================================
with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "crop_disease_model.pt"),
        map_location=device
    )
)

model = model.to(device)
model.eval()

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==========================================================
# HELPERS
# ==========================================================
def clean_json(text):
    return text.replace("```json", "").replace("```", "").strip()

def fallback_solution(label):
    return {
        "prediction": label,
        "cause": "Disease detected in crop leaf.",
        "symptoms": [
            "Leaf damage visible",
            "Spots / yellowing present",
            "Growth may reduce"
        ],
        "organic_treatment": [
            "Use neem oil spray",
            "Remove infected leaves"
        ],
        "chemical_treatment": [
            "Use recommended fungicide",
            "Consult agriculture shop"
        ],
        "prevention": [
            "Keep field clean",
            "Avoid excess water",
            "Use healthy seeds"
        ],
        "extra_tip": "Upload clearer image for better AI result.",
        "warning": "Use gloves during chemical spray."
    }

# ==========================================================
# OPENROUTER MODELS
# ==========================================================
MODELS = [
    "deepseek/deepseek-chat",
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free"
]

# ==========================================================
# AI SOLUTION
# ==========================================================
def get_ai_solution(raw_label, language):

    selected_language = LANG_MAP.get(language, "English")
    cache_key = f"{raw_label}_{language}"

    if cache_key in AI_CACHE:
        return AI_CACHE[cache_key]

    if not OPENROUTER_API_KEY:
        return fallback_solution(raw_label)

    prompt = f"""
You are expert agriculture doctor AI.

Detected disease: {raw_label}

Give complete treatment in {selected_language} language.

Return ONLY valid JSON:

{{
 "prediction":"translated disease name",
 "cause":"short reason",
 "symptoms":["point1","point2","point3"],
 "organic_treatment":["point1","point2"],
 "chemical_treatment":["point1","point2"],
 "prevention":["point1","point2"],
 "extra_tip":"farmer tip",
 "warning":"safety warning"
}}

No markdown.
Only JSON.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Crop Doctor AI"
    }

    for model_name in MODELS:

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        try:
            print("TRYING MODEL:", model_name)

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )

            data = response.json()
            print("MODEL RESPONSE:", data)

            if "choices" not in data:
                continue

            text = data["choices"][0]["message"]["content"]
            text = clean_json(text)

            parsed = json.loads(text)

            final_data = {
                "prediction": parsed.get("prediction", raw_label),
                "cause": parsed.get("cause", ""),
                "symptoms": parsed.get("symptoms", []),
                "organic_treatment": parsed.get("organic_treatment", []),
                "chemical_treatment": parsed.get("chemical_treatment", []),
                "prevention": parsed.get("prevention", []),
                "extra_tip": parsed.get("extra_tip", ""),
                "warning": parsed.get("warning", "")
            }

            if not final_data["symptoms"]:
                continue

            AI_CACHE[cache_key] = final_data
            save_cache()

            print("SUCCESS MODEL:", model_name)

            return final_data

        except Exception as e:
            print("FAILED MODEL:", model_name, e)
            continue

    return fallback_solution(raw_label)

# ==========================================================
# ROUTES
# ==========================================================
@app.get("/")
def home():
    return {"message": "Crop Doctor AI Running"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Form("hi")
):

    if not file.content_type.startswith("image/"):
        return {"error": "Upload image only"}

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except:
        return {"error": "Invalid image"}

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    raw_label = class_names[pred.item()]
    confidence_val = float(confidence.item())

    ai_data = get_ai_solution(raw_label, language)

    warning = None
    if confidence_val < 0.70:
        warning = "Low confidence. Upload clearer image."

    return {
        "prediction": ai_data["prediction"],
        "confidence": confidence_val,
        "warning": warning,
        "solution_source": "openrouter",
        "solution": {
            "cause": ai_data["cause"],
            "symptoms": ai_data["symptoms"],
            "organic_treatment": ai_data["organic_treatment"],
            "chemical_treatment": ai_data["chemical_treatment"],
            "prevention": ai_data["prevention"],
            "extra_tip": ai_data["extra_tip"],
            "warning": ai_data["warning"]
        },
        "language_selected": language
    }

