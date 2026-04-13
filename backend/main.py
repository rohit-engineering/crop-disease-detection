from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import io
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Language Mapping
# -----------------------------
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

# -----------------------------
# Gemini Setup
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

print("Gemini API Loaded:", "YES" if GEMINI_API_KEY else "NO")

# -----------------------------
# Cache file
# -----------------------------
CACHE_FILE = os.path.join(BASE_DIR, "gemini_cache.json")

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        try:
            GEMINI_CACHE = json.load(f)
        except:
            GEMINI_CACHE = {}
else:
    GEMINI_CACHE = {}


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(GEMINI_CACHE, f, indent=2, ensure_ascii=False)


# -----------------------------
# Load class names
# -----------------------------
with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model architecture
# -----------------------------
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

model_path = os.path.join(BASE_DIR, "crop_disease_model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)
model.eval()

# -----------------------------
# Image Transform (same as training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# -----------------------------
# Clean Gemini output
# -----------------------------
def clean_gemini_text(text: str):
    if not text:
        return ""
    return text.replace("```json", "").replace("```", "").strip()


# -----------------------------
# Translate Prediction Label
# -----------------------------
def translate_prediction_label(clean_label: str, language: str):

    if language == "en":
        return clean_label

    if not GEMINI_API_KEY or client is None:
        return clean_label

    selected_language = LANG_MAP.get(language, "Hindi")

    cache_key = f"prediction_{clean_label}_{language}"

    if cache_key in GEMINI_CACHE:
        return GEMINI_CACHE[cache_key]

    prompt = f"""
Translate the following crop disease prediction into {selected_language}.

Text: "{clean_label}"

Rules:
- Output only translated text
- Do not add explanation
- Keep meaning accurate
"""

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        translated_text = response.text.strip()

        GEMINI_CACHE[cache_key] = translated_text
        save_cache()

        return translated_text

    except:
        return clean_label


# -----------------------------
# Gemini Solution Function (Language supported)
# -----------------------------
def get_solution_from_gemini(raw_label: str, language: str):

    selected_language = LANG_MAP.get(language, "Hindi")

    cache_key = f"{raw_label}_solution_{language}"

    if cache_key in GEMINI_CACHE:
        return GEMINI_CACHE[cache_key]

    if not GEMINI_API_KEY or client is None:
        return {"error": "Gemini API key not found. Please set GEMINI_API_KEY in .env file."}

    prompt = f"""
You are an agriculture expert.

Disease Detected: {raw_label}

IMPORTANT RULE:
Give solution ONLY in {selected_language}.
Do not mix languages.
Use very simple farmer-friendly words.

Keep solutions short, effective, and safe.

Return output strictly in JSON format:

{{
  "crop": "...",
  "disease": "...",
  "cause": "...",
  "symptoms": ["...", "..."],
  "organic_treatment": ["...", "...", "..."],
  "chemical_treatment": ["...", "...", "..."],
  "prevention": ["...", "...", "..."],
  "extra_tip": "...",
  "warning": "Mention safety (gloves/mask), avoid spray in strong sunlight, and consult Krishi Vigyan Kendra if severe."
}}

Extra Rules:
- Give chemical names only if common and safe.
- Mention dosage only if standard.
- Mention wearing gloves/mask.
- Mention not to spray in strong sunlight.
- Mention "If infection is severe consult Krishi Vigyan Kendra".
"""

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        result_text = clean_gemini_text(response.text)

        try:
            result_json = json.loads(result_text)
        except:
            result_json = {
                "message": "Gemini response parsing failed (not proper JSON)",
                "raw_response": result_text
            }

        GEMINI_CACHE[cache_key] = result_json
        save_cache()

        return result_json

    except Exception as e:
        return {
            "error": "Gemini API failed",
            "details": str(e)
        }

def is_leaf_with_gemini(image_bytes):
    if not GEMINI_API_KEY or client is None:
        return True  # fallback (skip check)

    prompt = """
Check if the given image contains a plant leaf.

Rules:
- Reply ONLY with: YES or NO
- YES → if plant leaf is clearly visible
- NO → if not a plant leaf
"""

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
            ]
        )

        result = response.text.strip().upper()
        return "YES" in result

    except:
        return True
# -----------------------------
# API Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Crop Disease Detection API Running"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Form("hi")
):

    if not file.content_type.startswith("image/"):
        return {"error": "Invalid file type. Please upload an image."}

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except:
        return {"error": "Invalid image file."}
    

    if not is_leaf_with_gemini(image_bytes):
      return {
        "error": "❌ This is not a plant leaf image.",
        "message": "Please upload a clear plant leaf image."
    }

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    raw_label = class_names[predicted.item()]
    confidence_value = float(confidence.item())

    clean_label = raw_label.replace("___", " - ").replace("_", " ")

    # Translate prediction name
    translated_prediction = translate_prediction_label(clean_label, language)

    warning = None
    if confidence_value < 0.70:
        warning = "⚠️ Low confidence prediction. Please upload a clear leaf image in good lighting."
    # -----------------------------
    # Gemini fallback
    # -----------------------------
    gemini_solution = get_solution_from_gemini(raw_label, language)

    return {
        "prediction": translated_prediction,
        "confidence": confidence_value,
        "warning": warning,
        "solution_source": "gemini",
        "solution": gemini_solution,
        "language_selected": language
    }