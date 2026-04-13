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

# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print("OpenRouter Loaded:", "YES" if OPENROUTER_API_KEY else "NO")

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# BASE PATH
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# LANGUAGE MAP
# =====================================================
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

# =====================================================
# CACHE
# =====================================================
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


# =====================================================
# LOAD ML MODEL
# =====================================================
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

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =====================================================
# HELPERS
# =====================================================
def clean_json(text):
    return text.replace("```json", "").replace("```", "").strip()


# =====================================================
# OPENROUTER ONE CALL
# =====================================================
def get_ai_solution(raw_label, language):

    selected_language = LANG_MAP.get(language, "Hindi")

    cache_key = f"{raw_label}_{language}"

    if cache_key in AI_CACHE:
        return AI_CACHE[cache_key]

    if not OPENROUTER_API_KEY:
        return {
            "prediction": raw_label,
            "cause": "API key missing.",
            "symptoms": [],
            "organic_treatment": [],
            "chemical_treatment": [],
            "prevention": [],
            "extra_tip": "",
            "warning": ""
        }

    prompt = f"""
You are agriculture expert AI.

Disease detected: {raw_label}

Give complete solution in {selected_language} language.

Return ONLY JSON:

{{
 "prediction":"Translated disease name",
 "cause":"Reason",
 "symptoms":["...","..."],
 "organic_treatment":["...","..."],
 "chemical_treatment":["...","..."],
 "prevention":["...","..."],
 "extra_tip":"Farmer tip",
 "warning":"Use gloves and mask"
}}

No markdown.
No explanation.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Crop Doctor AI"
    }

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        data = response.json()

        text = data["choices"][0]["message"]["content"]

        text = clean_json(text)

        print("AI RAW:", text)

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

        AI_CACHE[cache_key] = final_data
        save_cache()

        return final_data

    except Exception as e:
        print("OpenRouter Error:", e)

        return {
            "prediction": raw_label,
            "cause": "Solution unavailable.",
            "symptoms": [],
            "organic_treatment": [],
            "chemical_treatment": [],
            "prevention": [],
            "extra_tip": "",
            "warning": ""
        }


# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def home():
    return {"message": "Crop Doctor AI OpenRouter Running"}


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