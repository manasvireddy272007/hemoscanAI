"""
HemoScan AI — Anemia Detection & Risk Analysis Backend
FastAPI REST API | Python 3.9+
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum
import uvicorn


# ─────────────────────────────────────────────
#  Enums & Constants
# ─────────────────────────────────────────────

class Gender(str, Enum):
    male   = "male"
    female = "female"
    other  = "other"

class PregnancyStatus(str, Enum):
    yes = "yes"
    no  = "no"

class DietType(str, Enum):
    balanced   = "balanced"
    vegetarian = "vegetarian"
    vegan      = "vegan"
    low_iron   = "low_iron"
    irregular  = "irregular"

VALID_SYMPTOMS = {
    "fatigue", "pallor", "dizziness",
    "shortness", "headache", "coldness"
}

VALID_GENETICS = {
    "iron_deficiency", "sickle_cell", "b12",
    "folate", "hemolytic", "thalassemia"
}

# ─────────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────────

class PatientInput(BaseModel):
    name:              str              = Field(..., min_length=1, max_length=100, example="Priya Sharma")
    age:               int              = Field(..., ge=1, le=120, example=28)
    gender:            Gender           = Field(..., example="female")
    pregnancy_status:  PregnancyStatus  = Field(PregnancyStatus.no, example="no")
    hemoglobin:        float            = Field(..., ge=1.0, le=25.0, example=10.2)
    rbc_count:         float            = Field(..., ge=1.0, le=8.0,  example=3.8)
    mcv:               float            = Field(..., ge=50.0, le=130.0, example=72.0)
    symptoms:          List[str]        = Field(default=[], example=["fatigue", "pallor"])
    genetic_history:   List[str]        = Field(default=[], example=["iron_deficiency"])
    diet_type:         DietType         = Field(DietType.balanced, example="vegetarian")

    @validator("symptoms", each_item=True)
    def validate_symptoms(cls, v):
        if v not in VALID_SYMPTOMS:
            raise ValueError(f"'{v}' is not a valid symptom. Valid: {VALID_SYMPTOMS}")
        return v

    @validator("genetic_history", each_item=True)
    def validate_genetics(cls, v):
        if v not in VALID_GENETICS:
            raise ValueError(f"'{v}' is not a valid genetic history entry. Valid: {VALID_GENETICS}")
        return v


class AnalysisResponse(BaseModel):
    patient_name:           str
    risk_level:             str
    risk_score:             int
    anemia_detected:        bool
    probable_anemia_type:   Optional[str]
    explanation:            str
    parameter_flags:        dict
    dietary_recommendations: List[dict]
    disclaimer:             str


# ─────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="HemoScan AI — Anemia Detection API",
    description="REST API for AI-driven anemia risk analysis based on CBC parameters and patient history.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Core Analysis Engine
# ─────────────────────────────────────────────

def compute_risk_score(data: PatientInput) -> tuple[int, dict]:
    """
    Rule-based risk scoring engine.
    Returns (score: int 0–100, parameter_flags: dict)
    """
    score = 0
    flags = {}

    # --- Hemoglobin thresholds (WHO standards) ---
    if data.gender == Gender.female or data.pregnancy_status == PregnancyStatus.yes:
        hgb_normal = 12.0
    else:
        hgb_normal = 13.0

    hgb_drop = hgb_normal - data.hemoglobin
    if hgb_drop >= 4:
        score += 40; flags["hemoglobin"] = "critically_low"
    elif hgb_drop >= 2:
        score += 25; flags["hemoglobin"] = "low"
    elif hgb_drop >= 0.5:
        score += 12; flags["hemoglobin"] = "borderline"
    else:
        flags["hemoglobin"] = "normal"

    # --- Pregnancy penalty ---
    if data.pregnancy_status == PregnancyStatus.yes:
        score += 8
        flags["pregnancy"] = "elevated_risk"

    # --- RBC Count ---
    rbc_normal_low = 3.9 if data.gender == Gender.female else 4.3
    rbc_drop = rbc_normal_low - data.rbc_count
    if rbc_drop >= 1.0:
        score += 20; flags["rbc_count"] = "critically_low"
    elif rbc_drop >= 0.4:
        score += 10; flags["rbc_count"] = "low"
    else:
        flags["rbc_count"] = "normal"

    # --- MCV (cell size) ---
    if data.mcv < 70:
        score += 20; flags["mcv"] = "microcytic_severe"
    elif data.mcv < 80:
        score += 10; flags["mcv"] = "microcytic"
    elif data.mcv > 100:
        score += 8;  flags["mcv"] = "macrocytic"
    else:
        flags["mcv"] = "normal"

    # --- Symptoms (5 pts each) ---
    score += len(data.symptoms) * 5
    flags["symptoms_count"] = len(data.symptoms)

    # --- Genetic history (7 pts each) ---
    score += len(data.genetic_history) * 7
    flags["genetic_flags"] = data.genetic_history

    # --- Diet risk ---
    diet_scores = {
        "vegan": 8, "vegetarian": 5,
        "low_iron": 12, "irregular": 12, "balanced": 0
    }
    score += diet_scores.get(data.diet_type, 0)
    flags["diet_risk"] = data.diet_type

    # --- Age extremes ---
    if data.age < 5 or data.age > 65:
        score += 5
        flags["age_risk"] = "elevated"
    else:
        flags["age_risk"] = "normal"

    return min(score, 100), flags


def classify_risk(score: int) -> tuple[str, bool, str]:
    """Returns (risk_level, anemia_detected, explanation)"""
    if score < 15:
        return (
            "No Anemia",
            False,
            "Your blood parameters are within normal range. No significant anemia risk detected. "
            "Maintain a balanced iron-rich diet and schedule a CBC checkup annually."
        )
    elif score < 40:
        return (
            "Low Risk",
            True,
            "Mild anemia indicators are present. Your hemoglobin or RBC levels are slightly below optimal. "
            "Dietary improvements and a follow-up CBC in 3 months are recommended."
        )
    elif score < 65:
        return (
            "Moderate Risk",
            True,
            "Multiple anemia risk factors detected across CBC values, symptoms, and history. "
            "Consult a physician promptly. Lab confirmation and possible supplementation are advised."
        )
    else:
        return (
            "High Risk",
            True,
            "Critical anemia indicators detected. Hemoglobin and/or RBC levels are significantly below "
            "normal thresholds. Immediate medical evaluation by a hematologist is strongly recommended."
        )


def predict_anemia_type(data: PatientInput, score: int) -> Optional[str]:
    """Rule-based anemia type classification."""
    if score < 15:
        return None

    genetics = set(data.genetic_history)

    if "sickle_cell" in genetics:
        return "Sickle Cell Anemia"

    if "thalassemia" in genetics or (data.mcv < 78 and data.rbc_count > 4.5):
        return "Thalassemia"

    if "hemolytic" in genetics:
        return "Hemolytic Anemia"

    if data.mcv > 100 or "b12" in genetics:
        return "Vitamin B12 Deficiency Anemia"

    if "folate" in genetics:
        return "Folate Deficiency Anemia"

    # Default: most common
    return "Iron Deficiency Anemia"


def get_dietary_recommendations(risk_level: str, anemia_type: Optional[str]) -> List[dict]:
    """Returns structured dietary recommendation cards."""

    base = [
        {
            "category": "Iron-Rich Foods",
            "icon": "🥩",
            "foods": ["Spinach", "Red meat", "Lentils", "Tofu", "Pumpkin seeds", "Fortified cereals"],
            "tip": "Consume at least one iron-rich food at every main meal."
        },
        {
            "category": "Vitamin C Pairing",
            "icon": "🍋",
            "foods": ["Oranges", "Lemon juice", "Bell peppers", "Tomatoes", "Strawberries"],
            "tip": "Pair with iron sources to enhance absorption by up to 3×."
        },
        {
            "category": "Foods to Avoid",
            "icon": "🚫",
            "foods": ["Tea", "Coffee", "Milk (with iron meals)", "Antacids"],
            "tip": "Avoid these within 1–2 hours of iron-rich meals as they inhibit absorption."
        },
        {
            "category": "Hydration",
            "icon": "💧",
            "foods": ["8–10 glasses of water daily"],
            "tip": "Proper hydration supports blood volume and nutrient transport."
        },
    ]

    if anemia_type in ("Vitamin B12 Deficiency Anemia",):
        return [
            {
                "category": "Vitamin B12 Sources",
                "icon": "🥚",
                "foods": ["Eggs", "Dairy products", "Salmon", "Tuna", "Fortified plant milk", "Nutritional yeast"],
                "tip": "B12 is found almost exclusively in animal products. Vegans must supplement."
            },
            {
                "category": "Folate Sources",
                "icon": "🥦",
                "foods": ["Kale", "Broccoli", "Avocado", "Chickpeas", "Fortified grains"],
                "tip": "Folate works alongside B12 in red blood cell maturation."
            },
            {
                "category": "Supplement Guidance",
                "icon": "💊",
                "foods": ["B12 supplements (methylcobalamin)", "B-complex vitamins"],
                "tip": "Consult your doctor for correct B12 dosage — deficiency is correctable."
            },
            base[3],
        ]

    if anemia_type == "Folate Deficiency Anemia":
        return [
            {
                "category": "Folate-Rich Foods",
                "icon": "🥦",
                "foods": ["Dark leafy greens", "Avocado", "Lentils", "Beans", "Asparagus", "Fortified grains"],
                "tip": "Cook vegetables lightly — prolonged cooking destroys folate."
            },
            {
                "category": "Vitamin B12 Support",
                "icon": "🥚",
                "foods": ["Eggs", "Milk", "Fish", "Fortified cereals"],
                "tip": "B12 and folate work together; deficiency in one affects the other."
            },
            {
                "category": "Protein Sources",
                "icon": "🫘",
                "foods": ["Quinoa", "Legumes", "Eggs", "Lean meat"],
                "tip": "Amino acids from protein are essential for hemoglobin synthesis."
            },
            base[3],
        ]

    if anemia_type in ("Sickle Cell Anemia", "Hemolytic Anemia", "Thalassemia"):
        return [
            {
                "category": "Folate Priority",
                "icon": "🥦",
                "foods": ["Dark leafy greens", "Avocado", "Lentils", "Fortified grains"],
                "tip": "Sickle cell and hemolytic conditions increase folate demand significantly."
            },
            {
                "category": "Antioxidant Foods",
                "icon": "🍇",
                "foods": ["Berries", "Grapes", "Nuts", "Dark chocolate", "Green tea"],
                "tip": "Antioxidants protect red blood cells from oxidative stress and early breakdown."
            },
            {
                "category": "Hydration (Critical)",
                "icon": "💧",
                "foods": ["Water", "Coconut water", "Electrolyte drinks"],
                "tip": "Dehydration triggers sickling episodes — maintain 10–12 glasses/day."
            },
            {
                "category": "Supplement Guidance",
                "icon": "💊",
                "foods": ["Folic acid 5mg/day (prescription)", "Hydroxyurea (physician-guided)"],
                "tip": "Management requires close physician monitoring. Never self-medicate."
            },
        ]

    if risk_level == "No Anemia":
        return [
            base[0],
            base[1],
            {
                "category": "Balanced Nutrition",
                "icon": "🥗",
                "foods": ["Whole grains", "Legumes", "Lean protein", "Colorful vegetables"],
                "tip": "A diverse diet is the best prevention against nutritional deficiencies."
            },
            base[3],
        ]

    # Default (Iron Deficiency / general risk)
    return base + [
        {
            "category": "Supplement Guidance",
            "icon": "💊",
            "foods": ["Ferrous sulfate (on prescription)", "Iron bisglycinate", "Multivitamin with iron"],
            "tip": "Supplements should only be taken after confirming iron deficiency via serum ferritin test."
        },
        {
            "category": "Protein & B-vitamins",
            "icon": "🫘",
            "foods": ["Eggs", "Legumes", "Quinoa", "Fish", "Chicken"],
            "tip": "Protein and B-vitamins are co-factors in hemoglobin production."
        },
    ]


# ─────────────────────────────────────────────
#  API Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "HemoScan AI — Anemia Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoint": "POST /analyze"
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze(patient: PatientInput):
    """
    Analyze patient CBC data and return anemia risk assessment.

    - Accepts hemoglobin, RBC count, MCV + clinical context
    - Returns risk level, probable anemia type, and dietary plan
    """
    try:
        # Step 1: Score
        score, flags = compute_risk_score(patient)

        # Step 2: Classify
        risk_level, anemia_detected, explanation = classify_risk(score)

        # Step 3: Type prediction
        anemia_type = predict_anemia_type(patient, score)

        # Step 4: Diet plan
        diet_recs = get_dietary_recommendations(risk_level, anemia_type)

        return AnalysisResponse(
            patient_name=patient.name,
            risk_level=risk_level,
            risk_score=score,
            anemia_detected=anemia_detected,
            probable_anemia_type=anemia_type,
            explanation=explanation,
            parameter_flags=flags,
            dietary_recommendations=diet_recs,
            disclaimer=(
                "This analysis is generated by a rule-based AI screening system and is intended "
                "for informational purposes only. It does not constitute a medical diagnosis. "
                "Please consult a qualified healthcare professional for clinical evaluation and treatment."
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
