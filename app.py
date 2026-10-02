# ==============================================================================
# UNIVERSITY STUDENT WELLBEING VOICE & TEXT TRIAGE SYSTEM
# Track 1: "Help students reach the right support"
#
# ARCHITECTURAL SPECIFICATION:
# 1. Ingest student voice recording (via microphone/upload) or text statement.
# 2. Local Hugging Face Safety Classifier (bert-suicide-detection) as an
#    emergency circuit breaker with a conservative decision threshold (p > 0.40).
# 3. Local Hugging Face Zero-Shot Classifier (DeBERTa-v3-base-mnli) to rank
#    the 12 candidate university support departments.
# 4. Official google-genai SDK (gemini-2.5-flash with resilient fallback) using
#    strict Pydantic JSON schema enforcement (TriagePayload).
# 5. Fail-Safe Override mechanism activating red emergency circuit breaker for
#    acute crisis/self-harm with 988 lifeline instructions and 0-hour CAPS dispatch.
# 6. Production Gradio web interface with dual-channel inputs and structured outputs.
#
# PIP INSTALLATION INSTRUCTIONS:
# pip install google-genai gradio transformers torch pydantic python-dotenv
# ============================================================================== 

import os
import sys
import json
import logging
import uuid
import mimetypes
from typing import Optional, List, Dict, Any, Tuple

from dotenv import load_dotenv
# Ensure stdout and stderr handle utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure Windows finds PyTorch C++ runtime DLLs (c10.dll, torch_cpu.dll) if applicable
_torch_lib_candidates = [
    os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib"),
    os.path.join(os.path.dirname(sys.executable), "lib", "site-packages", "torch", "lib"),
]
for _candidate in _torch_lib_candidates:
    if os.path.isdir(_candidate):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_candidate)
            except Exception:
                pass
        os.environ["PATH"] = _candidate + os.pathsep + os.environ.get("PATH", "")
        break

from transformers import pipeline
import gradio as gr
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("WellbeingTriageSystem")

# ------------------------------------------------------------------------------
# GEMINI API CONFIGURATION & CLIENT INITIALIZATION
# ------------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    logger.error("CRITICAL: GEMINI_API_KEY is not set. Please add it to your .env file or environment variables.")
    client = None
else:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Official google-genai Client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize google-genai Client: {e}")
        client = None

# Model hierarchy: try requested model first, then resilient fallbacks
PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

# ------------------------------------------------------------------------------
# THE 12 TARGET UNIVERSITY DEPARTMENTS
# ------------------------------------------------------------------------------
TARGET_DEPARTMENTS: List[str] = [
    "Counseling & Psychological Services (CAPS)",
    "Student Health / Campus Medical Clinic",
    "Sexual Assault & Trauma Prevention (Title IX)",
    "Academic Advising & Course Drops",
    "Disability & Accessibility Resources",
    "Academic Tutoring & Learning Skills Center",
    "Housing & Residence Life",
    "Dean of Students / CARE Team",
    "Campus Security & Emergency Transport",
    "Financial Aid & Emergency Relief",
    "International Student Services",
    "Multicultural & Identity Resource Centers",
]

# ------------------------------------------------------------------------------
# REQUIRED PYDANTIC SCHEMAS FOR STRUCTURED TRIAGE OUTPUT
# ------------------------------------------------------------------------------
class SafetyAssessment(BaseModel):
    is_crisis: bool = Field(
        ...,
        description="Whether immediate acute crisis, active self-harm ideation, or severe danger is detected.",
    )
    urgency_level: int = Field(
        ...,
        ge=1,
        le=5,
        description="Clinical urgency rating from 1 (routine informational) to 5 (immediate life safety crisis).",
    )
    primary_emotion: str = Field(
        ...,
        description="Primary detected affective state of the student (e.g., overwhelmed, despondent, anxious, calm).",
    )
    risk_indicators: List[str] = Field(
        ...,
        description="List of clinical risk factors, behavioral flags, or environmental vulnerabilities identified.",
    )


class ReferralTicket(BaseModel):
    ticket_id: str = Field(
        ...,
        description="Structured unique ticket identifier, formatted as 'TICK-XXXX'.",
    )
    assigned_department: str = Field(
        ...,
        description="The primary target department from the authorized 12 university departments.",
    )
    secondary_department: Optional[str] = Field(
        None,
        description="Optional secondary department if cross-functional support is needed, or null.",
    )
    intake_summary: str = Field(
        ...,
        description="Clinical summary note for the receiving department advisor detailing key stressors, facts, and context.",
    )
    suggested_action: str = Field(
        ...,
        description="Immediate actionable next step for the receiving advisor or intake officer.",
    )
    estimated_sla_hours: int = Field(
        ...,
        description="Recommended service level agreement (SLA) response window in hours (0 for crisis, 12-24 for elevated, 48-72 for routine).",
    )


class StudentFacingResponse(BaseModel):
    spoken_dialogue: str = Field(
        ...,
        description="Warm, empathetic, natural, trauma-informed spoken response under 3 sentences for direct voice readout to the student.",
    )


class TriagePayload(BaseModel):
    safety_assessment: SafetyAssessment = Field(
        ...,
        description="Clinical safety evaluation and urgency scoring.",
    )
    referral_ticket: ReferralTicket = Field(
        ...,
        description="Formal intake ticket dispatched to campus resources.",
    )
    student_facing_response: StudentFacingResponse = Field(
        ...,
        description="Direct student-facing spoken dialogue.",
    )


# ------------------------------------------------------------------------------
# LOCAL HUGGING FACE MODEL MANAGEMENT (CIRCUIT BREAKER & ZERO-SHOT)
# ------------------------------------------------------------------------------
_circuit_breaker_pipe = None
_dept_classifier_pipe = None


def get_circuit_breaker():
    """Lazy-load and cache the local Akashpaul123/bert-suicide-detection pipeline."""
    global _circuit_breaker_pipe
    if _circuit_breaker_pipe is None:
        logger.info("Initializing local Hugging Face Circuit Breaker (Akashpaul123/bert-suicide-detection)...")
        try:
            _circuit_breaker_pipe = pipeline(
                "text-classification",
                model="Akashpaul123/bert-suicide-detection",
                top_k=None,
            )
            logger.info("Circuit Breaker pipeline loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load bert-suicide-detection pipeline directly: {e}. Using fallback heuristic.")
            _circuit_breaker_pipe = None
    return _circuit_breaker_pipe


def get_department_classifier():
    """Lazy-load and cache the local MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli pipeline."""
    global _dept_classifier_pipe
    if _dept_classifier_pipe is None:
        logger.info("Initializing local Hugging Face Zero-Shot Classifier (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)...")
        try:
            _dept_classifier_pipe = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            )
            logger.info("DeBERTa-v3 zero-shot department classifier loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load DeBERTa-v3 pipeline directly: {e}. Using heuristic ranking.")
            _dept_classifier_pipe = None
    return _dept_classifier_pipe


def run_circuit_breaker(text: str) -> Tuple[float, bool]:
    """
    Run the student transcript through the Akashpaul123/bert-suicide-detection model.
    Conservative decision threshold: p > 0.40 flags crisis.
    Returns (crisis_probability, hf_crisis_flag).
    """
    if not text or not text.strip():
        return 0.0, False

    pipe = get_circuit_breaker()
    crisis_prob = 0.0

    if pipe is not None:
        try:
            # Model returns list of lists when top_k=None: [[{'label': 'LABEL_0', 'score': ...}, {'label': 'LABEL_1', 'score': ...}]]
            preds = pipe(text[:512])
            if preds and isinstance(preds[0], list):
                preds = preds[0]

            for item in preds:
                label = str(item.get("label", "")).upper()
                score = float(item.get("score", 0.0))
                # In bert-suicide-detection: LABEL_1 = Suicide, LABEL_0 = Non-suicide
                if label in ["LABEL_1", "SUICIDE"]:
                    crisis_prob = score
                    break
                elif label in ["LABEL_0", "NON-SUICIDE", "NON_SUICIDE"]:
                    crisis_prob = max(0.0, 1.0 - score)

            logger.info(f"Local HF Circuit Breaker evaluated text. Crisis score: {crisis_prob:.4f}")
        except Exception as e:
            logger.error(f"Error during circuit breaker inference: {e}")
            crisis_prob = _heuristic_suicide_score(text)
    else:
        crisis_prob = _heuristic_suicide_score(text)

    # Conservative threshold: p > 0.40 flags crisis
    hf_crisis_flag = bool(crisis_prob > 0.40)
    return crisis_prob, hf_crisis_flag


def run_department_screening(text: str) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Run DeBERTa-v3 against the 12 candidate departments.
    Returns (top_department, top_score, all_rankings).
    """
    if not text or not text.strip():
        return TARGET_DEPARTMENTS[0], 0.0, []

    pipe = get_department_classifier()
    if pipe is not None:
        try:
            res = pipe(text[:512], candidate_labels=TARGET_DEPARTMENTS)
            labels = res.get("labels", [])
            scores = res.get("scores", [])
            rankings = [{"department": lbl, "score": float(sc)} for lbl, sc in zip(labels, scores)]
            top_dept = labels[0] if labels else TARGET_DEPARTMENTS[0]
            top_score = float(scores[0]) if scores else 0.0
            return top_dept, top_score, rankings[:5]
        except Exception as e:
            logger.error(f"Error during zero-shot department screening: {e}")

    # Fallback heuristic screening if pipeline is offline
    return _heuristic_department_screening(text)


def _heuristic_suicide_score(text: str) -> float:
    """High-recall safety net heuristic if local neural weights encounter unexpected runtime fault."""
    lower = text.lower()
    high_risk_keywords = [
        "kill myself", "suicide", "end my life", "end it all", "want to die",
        "better off dead", "no reason to live", "cannot live anymore", "bottle of pills",
        "hanging myself", "cut myself", "shoot myself", "jump off"
    ]
    moderate_risk = ["hopeless", "can't go on", "giving up", "no point in living", "nobody cares"]

    for kw in high_risk_keywords:
        if kw in lower:
            return 0.95
    for kw in moderate_risk:
        if kw in lower:
            return 0.65
    return 0.05


def _heuristic_department_screening(text: str) -> Tuple[str, float, List[Dict[str, Any]]]:
    """Keyword-based zero-shot fallback for candidate ranking."""
    lower = text.lower()
    scores = {}
    keywords = {
        "Counseling & Psychological Services (CAPS)": ["depressed", "anxiety", "crying", "mental", "therapy", "hopeless", "sad", "panic"],
        "Financial Aid & Emergency Relief": ["tuition", "aid", "scholarship", "money", "loan", "broke", "rent", "afford", "grant"],
        "Academic Advising & Course Drops": ["drop", "withdraw", "major", "advisor", "credits", "gpa", "course", "schedule", "probation"],
        "Housing & Residence Life": ["dorm", "roommate", "eviction", "homeless", "lease", "housing", "apartment", "hall"],
        "Academic Tutoring & Learning Skills Center": ["tutor", "failing", "quiz", "homework", "exam", "chemistry", "study", "learning"],
        "Student Health / Campus Medical Clinic": ["sick", "doctor", "prescription", "fever", "medication", "clinic", "flu", "injury"],
        "Sexual Assault & Trauma Prevention (Title IX)": ["assault", "harassment", "title ix", "consent", "abuse", "stalking", "trauma"],
        "Disability & Accessibility Resources": ["accommodation", "adhd", "dyslexia", "extended time", "disability", "accessible"],
        "Dean of Students / CARE Team": ["distress", "emergency leave", "bereavement", "family emergency", "complaint", "care team"],
        "Campus Security & Emergency Transport": ["threat", "stalker", "danger", "escort", "security", "safe walk", "transport"],
        "International Student Services": ["visa", "i-20", "f-1", "opt", "cpt", "immigration", "embassy", "international"],
        "Multicultural & Identity Resource Centers": ["lgbtq", "identity", "cultural", "belonging", "pride", "discrimination", "diversity"],
    }
    for dept, kws in keywords.items():
        match_count = sum(1 for kw in kws if kw in lower)
        scores[dept] = 0.1 + (match_count * 0.25)

    sorted_ranks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_dept = sorted_ranks[0][0]
    top_score = min(0.99, sorted_ranks[0][1])
    rankings = [{"department": d, "score": min(0.99, s)} for d, s in sorted_ranks[:5]]
    return top_dept, top_score, rankings


# ------------------------------------------------------------------------------
# MULTIMODAL AUDIO TRANSCRIPTION VIA GEMINI
# ------------------------------------------------------------------------------
def transcribe_audio_with_gemini(audio_path: str) -> str:
    """
    Pass student voice recording directly to Gemini to extract high-fidelity verbatim text.
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""

    if client is None:
        raise ValueError("Gemini API client is not initialized. Please configure GEMINI_API_KEY.")

    logger.info(f"Transcribing student voice input from: {audio_path}")

    # Determine mime-type
    mime_type, _ = mimetypes.guess_type(audio_path)
    if not mime_type:
        ext = os.path.splitext(audio_path)[1].lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mp3",
            ".m4a": "audio/m4a",
            ".ogg": "audio/ogg",
            ".webm": "audio/webm",
            ".flac": "audio/flac",
        }
        mime_type = mime_map.get(ext, "audio/wav")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    transcribe_prompt = (
        "You are an accurate, compassionate medical and student intake transcriber. "
        "Listen to this audio recording of a university student. "
        "Transcribe the student's spoken words verbatim. "
        "Do NOT add introductory text, formatting tags, quotation marks, or assumptions. "
        "Return ONLY the spoken transcription text."
    )

    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting audio transcription using model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    transcribe_prompt,
                ],
            )
            transcript = response.text.strip() if response and response.text else ""
            if transcript:
                logger.info(f"Audio transcription completed ({len(transcript)} chars): '{transcript}'")
                return transcript
        except Exception as e:
            logger.warning(f"Audio transcription failed with model {model_name}: {e}")

    raise RuntimeError("Failed to transcribe audio across all available Gemini models.")


# ------------------------------------------------------------------------------
# GEMINI CLINICAL TRIAGE SYNTHESIS (STRUCTURED JSON OUTPUT)
# ------------------------------------------------------------------------------
def synthesize_triage_decision(
    student_statement: str,
    hf_crisis_score: float,
    hf_crisis_flag: bool,
    hf_predicted_dept: str,
    hf_dept_score: float,
) -> TriagePayload:
    """
    Call Gemini with strict Pydantic JSON schema enforcement to evaluate clinical
    urgency (levels 1-5), route to one of the 12 authorized departments, and craft
    a warm, trauma-informed spoken response.
    """
    if client is None:
        raise ValueError("Gemini API client is not configured. Please supply a valid GEMINI_API_KEY.")

    departments_list_str = "\n".join([f"- {d}" for d in TARGET_DEPARTMENTS])

    triage_prompt = f"""
You are the Chief Clinical AI Triage Officer for a mid-sized University Student Wellbeing Network.
Your mission is to perform an empathetic, evidence-based triage evaluation on a student's statement
and route them to the single best resource among the 12 authorized campus departments.

### AUTHORIZED TARGET DEPARTMENTS (You MUST select assigned_department from this list):
{departments_list_str}

### INCOMING STUDENT STATEMENT:
"{student_statement}"

### PRE-COMPUTED LOCAL SAFETY & SCREENING SIGNALS:
1. Local Hugging Face BERT Suicide Detector Score: {hf_crisis_score:.4f} (Flagged Crisis: {hf_crisis_flag})
2. Local Hugging Face DeBERTa-v3 Department Suggestion: "{hf_predicted_dept}" (Confidence: {hf_dept_score:.4f})

### CLINICAL PROTOCOL & GUIDELINES:
1. Urgency Level Rubric:
   - Level 1 (Routine): General administrative inquiries, standard advising, informational questions. SLA: 72 hrs.
   - Level 2 (Mild Stress): Normal collegiate adjustment, exam anxiety, non-emergency study skills. SLA: 48 hrs.
   - Level 3 (Moderate Impairment): Sustained depressive mood, course failure panic, sleep disturbance, isolation. SLA: 24 hrs.
   - Level 4 (Severe Distress): Impending eviction/homelessness, panic attacks, trauma disclosure, acute breakdown risk. SLA: 12 hrs.
   - Level 5 (Acute Life Crisis): Expressed suicidal ideation, intent/plan, self-harm, physical violence, medical emergency. SLA: 0 hrs.

2. Safety Circuit Breaker Rule:
   - If the HF Suicide Score is > 0.40 OR the student text expresses thoughts of suicide, ending life, or severe self-harm:
     You MUST assign urgency_level = 5, is_crisis = true, and assigned_department = "Counseling & Psychological Services (CAPS)".
     Set estimated_sla_hours = 0.

3. Department Routing Rules:
   - Financial distress / eviction -> "Financial Aid & Emergency Relief" or "Housing & Residence Life".
   - Exam panic / course drops -> "Academic Advising & Course Drops" or "Academic Tutoring & Learning Skills Center".
   - Disability accommodations / ADHD -> "Disability & Accessibility Resources".
   - International visa / immigration -> "International Student Services".
   - Identity / community -> "Multicultural & Identity Resource Centers".
   - Medical physical symptoms -> "Student Health / Campus Medical Clinic".
   - Trauma / Title IX -> "Sexual Assault & Trauma Prevention (Title IX)".
   - Imminent physical danger / stalking -> "Campus Security & Emergency Transport".

4. Student-Facing Spoken Dialogue:
   - Must be warm, natural, trauma-informed, and direct.
   - Strictly UNDER 3 SENTENCES (concise for voice synthesis).
   - Acknowledge their feeling without judgment and reassure them of immediate human support.

5. Structured Referral Ticket:
   - Generate ticket_id format: "TICK-" followed by a 4-digit uppercase alphanumeric code (e.g. TICK-78A4).
   - Write a professional clinical intake note for the receiving university advisor.
   - Provide an immediate suggested action for the staff member.
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TriagePayload,
        temperature=0.1,
    )

    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS
    last_error = None

    for model_name in models_to_try:
        try:
            logger.info(f"Calling Gemini ({model_name}) with strict TriagePayload response_schema...")
            response = client.models.generate_content(
                model=model_name,
                contents=triage_prompt,
                config=config,
            )
            if response and response.text:
                payload_dict = json.loads(response.text)
                triage_payload = TriagePayload.model_validate(payload_dict)
                logger.info(f"Gemini triage synthesis succeeded with model: {model_name}")
                return triage_payload
        except Exception as e:
            logger.warning(f"Triage generation with {model_name} failed: {e}")
            last_error = e

    raise RuntimeError(f"All Gemini models failed to generate triage payload: {last_error}")


# ------------------------------------------------------------------------------
# FAIL-SAFE OVERRIDE & COMPOSITE ORCHESTRATION PIPELINE
# ------------------------------------------------------------------------------
def execute_triage_pipeline(
    audio_path: Optional[str],
    text_input: Optional[str],
) -> Tuple[str, str, str, str, Dict[str, Any]]:
    """
    Main triage coordinator function executed by Gradio.
    Returns:
    - transcribed_text: str
    - alert_banner_html: str
    - spoken_response: str
    - ticket_card_html: str
    - full_payload_json: dict
    """
    # Step 1: Input Resolution & Audio Transcription
    effective_text = (text_input or "").strip()

    if audio_path and os.path.exists(audio_path):
        try:
            audio_transcript = transcribe_audio_with_gemini(audio_path)
            if audio_transcript:
                effective_text = audio_transcript
        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            if not effective_text:
                error_banner = (
                    "<div style='background-color:#fee2e2; border:2px solid #ef4444; border-radius:10px; padding:16px; color:#991b1b;'>"
                    "<strong>Audio Transcription Failed:</strong> Could not transcribe the voice recording. Please check your microphone or type your statement in the text box below."
                    f"<p style='margin:4px 0 0 0; font-size:12px; color:#b91c1c;'>Error: {str(e)}</p></div>"
                )
                return "", error_banner, "I'm having trouble hearing the audio. Please type your message into the text box so I can assist you right away.", "", {}

    if not effective_text:
        warning_banner = (
            "<div style='background-color:#fef3c7; border:2px solid #f59e0b; border-radius:10px; padding:16px; color:#92400e; font-weight:600;'>"
            "⚠️ Please speak into the microphone or enter a student statement to begin triage."
            "</div>"
        )
        return "", warning_banner, "Please share what you are experiencing so I can connect you with the right campus support.", "", {}

    # Step 2: Local Hugging Face Fast Circuit Breaker (Suicide Detection)
    hf_crisis_score, hf_crisis_flag = run_circuit_breaker(effective_text)

    # Step 3: Local Hugging Face Department Zero-Shot Screening
    hf_top_dept, hf_dept_score, hf_rankings = run_department_screening(effective_text)

    # Step 4: Gemini LLM Clinical Synthesis
    try:
        triage = synthesize_triage_decision(
            student_statement=effective_text,
            hf_crisis_score=hf_crisis_score,
            hf_crisis_flag=hf_crisis_flag,
            hf_predicted_dept=hf_top_dept,
            hf_dept_score=hf_dept_score,
        )
    except Exception as e:
        logger.critical(f"LLM Synthesis failed completely: {e}. Activating autonomous emergency fallback.")
        # Emergency heuristic fallback construction
        is_crisis_fallback = hf_crisis_flag or (hf_crisis_score > 0.40)
        triage = TriagePayload(
            safety_assessment=SafetyAssessment(
                is_crisis=is_crisis_fallback,
                urgency_level=5 if is_crisis_fallback else 3,
                primary_emotion="distressed" if is_crisis_fallback else "uncertain",
                risk_indicators=["Neural synthesis fallback triggered", f"HF Suicide Score: {hf_crisis_score:.2f}"],
            ),
            referral_ticket=ReferralTicket(
                ticket_id=f"TICK-{uuid.uuid4().hex[:4].upper()}",
                assigned_department="Counseling & Psychological Services (CAPS)" if is_crisis_fallback else hf_top_dept,
                secondary_department="Dean of Students / CARE Team",
                intake_summary=f"Automated fallback referral for student presenting: {effective_text[:200]}...",
                suggested_action="Review intake statement immediately and contact student via registered campus emergency contact.",
                estimated_sla_hours=0 if is_crisis_fallback else 24,
            ),
            student_facing_response=StudentFacingResponse(
                spoken_dialogue=(
                    "Thank you for reaching out. We take what you're going through seriously and are immediately connecting you with caring campus support."
                    if not is_crisis_fallback else
                    "I hear how much pain you are in right now, and you do not have to carry this alone. Please stay with us while we connect you to immediate 24/7 crisis support."
                )
            ),
        )

    # Step 5: Fail-Safe Override Evaluation
    # If HF or Gemini detects crisis (urgency == 5 or is_crisis == True or hf_crisis_flag == True)
    is_acute_crisis = (
        hf_crisis_flag
        or (hf_crisis_score > 0.40)
        or triage.safety_assessment.is_crisis
        or (triage.safety_assessment.urgency_level == 5)
    )

    if is_acute_crisis:
        logger.warning("🚨 EMERGENCY CIRCUIT BREAKER ACTIVATED: Acute crisis flagged by safety layer.")
        triage.safety_assessment.is_crisis = True
        triage.safety_assessment.urgency_level = 5
        triage.referral_ticket.assigned_department = "Counseling & Psychological Services (CAPS)"
        triage.referral_ticket.estimated_sla_hours = 0
        triage.referral_ticket.suggested_action = (
            "🚨 CRITICAL INTERRUPT: Initiate immediate warm handoff to CAPS on-call crisis counselor or Campus Security. "
            "Do not leave student unmonitored. Deploy on-call emergency protocol."
        )

    # Step 6: Render Visual Banners & Ticket Formatting
    alert_banner_html = render_status_banner(
        urgency_level=triage.safety_assessment.urgency_level,
        is_crisis=triage.safety_assessment.is_crisis,
        hf_crisis_score=hf_crisis_score,
        hf_flag=hf_crisis_flag,
    )

    ticket_card_html = render_ticket_card(triage.referral_ticket, triage.safety_assessment, hf_top_dept, hf_dept_score)

    return (
        effective_text,
        alert_banner_html,
        triage.student_facing_response.spoken_dialogue,
        ticket_card_html,
        triage.model_dump(),
    )


# ------------------------------------------------------------------------------
# UI HTML RENDERERS (CRISIS / AMBER / GREEN BANNERS & TICKET CARD)
# ------------------------------------------------------------------------------
def render_status_banner(urgency_level: int, is_crisis: bool, hf_crisis_score: float, hf_flag: bool) -> str:
    """
    Generate the visual status banner:
    - RED: Acute Crisis Interrupt (Urgency 5 or is_crisis=True)
    - AMBER: Elevated Urgency (Urgency 3-4)
    - GREEN: Routine Support (Urgency 1-2)
    """
    if is_crisis or urgency_level >= 5:
        return f"""
        <div style="background: linear-gradient(135deg, #7f1d1d, #991b1b, #b91c1c); color: #ffffff; padding: 22px; border-radius: 12px; border: 3px solid #ef4444; box-shadow: 0 10px 25px rgba(239, 68, 68, 0.4); animation: pulse 2s infinite;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 28px;">🚨</span>
                    <span style="font-size: 20px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;">EMERGENCY CIRCUIT BREAKER ACTIVATED</span>
                </div>
                <span style="background-color: #fef2f2; color: #b91c1c; font-weight: 800; padding: 4px 12px; border-radius: 20px; font-size: 13px;">URGENCY: LEVEL 5 (CRITICAL)</span>
            </div>
            <p style="font-size: 15px; margin: 8px 0; font-weight: 600;">
                Acute distress or self-harm risk detected. The system has initiated an immediate fail-safe lock on the triage pipeline.
            </p>
            <div style="background: rgba(0,0,0,0.3); padding: 14px; border-radius: 8px; margin: 12px 0;">
                <div style="font-size: 16px; font-weight: 700; margin-bottom: 6px;">📞 24/7 IMMEDIATE CRISIS INTERVENTION LIFELINES:</div>
                <ul style="margin: 0; padding-left: 20px; font-size: 14px; line-height: 1.6;">
                    <li><strong>National Suicide & Crisis Lifeline:</strong> Call or Text <span style="background:#ffffff; color:#991b1b; padding:2px 8px; border-radius:4px; font-weight:800;">988</span> (Free, confidential, 24/7)</li>
                    <li><strong>Campus On-Call Emergency Crisis Team:</strong> <span style="background:#ffffff; color:#991b1b; padding:2px 8px; border-radius:4px; font-weight:800;">(555) 019-9111</span> (Immediate dispatch)</li>
                    <li><strong>CAPS Crisis Walk-In:</strong> Student Health Center, 2nd Floor, Rm 204 (Mon-Fri 8am-5pm)</li>
                    <li><strong>Crisis Text Line:</strong> Text <code>HOME</code> to <code>741741</code></li>
                </ul>
            </div>
            <div style="font-size: 12px; opacity: 0.9; display: flex; justify-content: space-between;">
                <span>HF Suicide Detector Score: <strong>{hf_crisis_score:.4f}</strong> (Threshold: 0.40)</span>
                <span>Circuit Breaker State: <strong>HARD OVERRIDE ENGAGED</strong></span>
            </div>
        </div>
        """
    elif urgency_level in [3, 4]:
        return f"""
        <div style="background: linear-gradient(135deg, #78350f, #92400e, #b45309); color: #ffffff; padding: 18px; border-radius: 12px; border: 2px solid #f59e0b; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.25);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.25); padding-bottom: 6px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 22px;">⚡</span>
                    <span style="font-size: 17px; font-weight: 700;">ELEVATED TRIAGE PRIORITY (LEVEL {urgency_level})</span>
                </div>
                <span style="background-color: #fef3c7; color: #92400e; font-weight: 700; padding: 3px 10px; border-radius: 20px; font-size: 12px;">SLA: EXPEDITED REVIEW</span>
            </div>
            <p style="font-size: 14px; margin: 6px 0;">
                Significant academic, psychological, or situational stress identified. Expedited case management ticket has been queued.
            </p>
            <div style="font-size: 12px; opacity: 0.9;">
                Local HF Suicide Classifier: <strong>{hf_crisis_score:.4f}</strong> (Normal range < 0.40) | Standard crisis protocols remain on standby.
            </div>
        </div>
        """
    else:
        return f"""
        <div style="background: linear-gradient(135deg, #064e3b, #065f46, #047857); color: #ffffff; padding: 18px; border-radius: 12px; border: 2px solid #10b981; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.25); padding-bottom: 6px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 22px;">✅</span>
                    <span style="font-size: 17px; font-weight: 700;">ROUTINE INTAKE & REFERRAL (LEVEL {urgency_level})</span>
                </div>
                <span style="background-color: #d1fae5; color: #065f46; font-weight: 700; padding: 3px 10px; border-radius: 20px; font-size: 12px;">STANDARD SLA: 48-72 HRS</span>
            </div>
            <p style="font-size: 14px; margin: 6px 0;">
                No acute safety risks detected. Ticket dispatched to primary university services for standard scheduled appointment.
            </p>
            <div style="font-size: 12px; opacity: 0.9;">
                Circuit Breaker Score: <strong>{hf_crisis_score:.4f}</strong> | Safe status confirmed by dual neural classifier consensus.
            </div>
        </div>
        """


def render_ticket_card(ticket: ReferralTicket, safety: SafetyAssessment, hf_dept: str, hf_score: float) -> str:
    """Render a clean, responsive HTML Referral Ticket Card."""
    sla_badge_color = "#dc2626" if ticket.estimated_sla_hours == 0 else ("#f59e0b" if ticket.estimated_sla_hours <= 24 else "#10b981")
    sla_text = "0 HOURS (IMMEDIATE DISPATCH)" if ticket.estimated_sla_hours == 0 else f"{ticket.estimated_sla_hours} HOURS"

    risk_tags = "".join([
        f"<span style='background:#e2e8f0; color:#334155; padding:3px 8px; border-radius:6px; font-size:11px; margin-right:4px; font-weight:600;'>{tag}</span>"
        for tag in safety.risk_indicators
    ]) or "<span style='color:#64748b; font-size:12px;'>None identified</span>"

    secondary = f"<div style='margin-top:4px; font-size:13px; color:#475569;'><strong>Cross-Department:</strong> {ticket.secondary_department}</div>" if ticket.secondary_department else ""

    return f"""
    <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 14px;">
            <div>
                <span style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Official University Referral Ticket</span>
                <div style="font-size: 22px; font-weight: 800; color: #0f172a;">{ticket.ticket_id}</div>
            </div>
            <div style="text-align: right;">
                <span style="background-color: {sla_badge_color}; color: #ffffff; font-weight: 800; padding: 4px 12px; border-radius: 16px; font-size: 12px;">
                    SLA: {sla_text}
                </span>
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Urgency Score: <strong>{safety.urgency_level} / 5</strong></div>
            </div>
        </div>

        <div style="margin-bottom: 14px; background: #f8fafc; border-left: 4px solid #2563eb; padding: 12px; border-radius: 0 8px 8px 0;">
            <div style="font-size: 11px; font-weight: 700; color: #2563eb; text-transform: uppercase;">Primary Receiving Department</div>
            <div style="font-size: 18px; font-weight: 800; color: #1e293b; margin-top: 2px;">{ticket.assigned_department}</div>
            {secondary}
            <div style="font-size: 11px; color: #64748b; margin-top: 6px;">
                HF DeBERTa Initial Suggestion: <em>{hf_dept}</em> (Confidence: {hf_score*100:.1f}%)
            </div>
        </div>

        <div style="margin-bottom: 12px;">
            <div style="font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 4px;">Advisor Intake Clinical Summary</div>
            <div style="font-size: 13.5px; color: #1e293b; line-height: 1.5; background: #fdfdfd; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px;">
                {ticket.intake_summary}
            </div>
        </div>

        <div style="margin-bottom: 14px;">
            <div style="font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 4px;">Recommended Action Protocol</div>
            <div style="font-size: 13.5px; color: #047857; font-weight: 600; line-height: 1.5; background: #ecfdf5; border: 1px solid #a7f3d0; padding: 10px; border-radius: 8px;">
                👉 {ticket.suggested_action}
            </div>
        </div>

        <div style="border-top: 1px solid #f1f5f9; padding-top: 10px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 11px; font-weight: 700; color: #64748b;">Primary Emotion:</span>
                <span style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: capitalize; margin-left: 4px;">{safety.primary_emotion}</span>
            </div>
            <div>
                {risk_tags}
            </div>
        </div>
    </div>
    """


# ------------------------------------------------------------------------------
# GRADIO APPLICATION BUILDER & INTERFACE
# ------------------------------------------------------------------------------
custom_css = """
.container { max-width: 1280px; margin: auto; }
.header-box { text-align: center; margin-bottom: 18px; padding: 16px; border-radius: 12px; background: linear-gradient(135deg, #1e3a8a, #1e40af); color: white; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.008); }
    100% { transform: scale(1); }
}
"""

with gr.Blocks(title="University Wellbeing Triage Engine") as demo:
    with gr.Column(elem_classes=["container"]):
        gr.HTML(
            """
            <div class="header-box">
                <div style="display:flex; justify-content:center; align-items:center; gap:12px; margin-bottom:4px;">
                    <span style="font-size:32px;">🎓</span>
                    <h1 style="font-size:26px; font-weight:800; margin:0; letter-spacing:-0.5px;">University Student Wellbeing Voice & Text Triage System</h1>
                </div>
                <p style="font-size:14px; margin:4px 0 0 0; opacity:0.9;">
                    Track 1: "Help students reach the right support" — Intelligent multi-department routing & emergency circuit breaker engine
                </p>
                <div style="margin-top:10px; display:flex; justify-content:center; gap:8px; flex-wrap:wrap; font-size:11px;">
                    <span style="background:rgba(255,255,255,0.2); padding:3px 10px; border-radius:12px;">🛡️ Circuit Breaker: bert-suicide-detection</span>
                    <span style="background:rgba(255,255,255,0.2); padding:3px 10px; border-radius:12px;">🎯 Routing: DeBERTa-v3 12-Department MNLI</span>
                    <span style="background:rgba(255,255,255,0.2); padding:3px 10px; border-radius:12px;">🧠 Synthesis: google-genai gemini-2.5-flash</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            # LEFT COLUMN: INPUT CONTROLS
            with gr.Column(scale=5):
                gr.Markdown("### 📥 Student Intake Channel (Voice or Text)")
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎙️ Voice Intake (Speak into microphone or upload audio)",
                )

                text_input = gr.Textbox(
                    lines=4,
                    placeholder="Type student statement here, or record your voice above...",
                    label="📝 Text Statement (Fallback / Direct Input)",
                )

                with gr.Row():
                    submit_btn = gr.Button("⚡ Run Intelligent Triage", variant="primary", scale=2)
                    clear_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)

                gr.Markdown("#### 🧪 Test Benchmark Scenarios (Click to Load)")
                example_scenarios = [
                    [
                        "I don't think I can keep doing this anymore. Everything is way too much, I feel completely hopeless and don't want to wake up tomorrow. I have a bottle of pills on my desk.",
                    ],
                    [
                        "I need to apply for university emergency financial assistance and off-campus housing relief because my semester stipend is delayed and rent is due next Monday. Can you guide me through the emergency aid process?",
                    ],
                    [
                        "I am having debilitating panic attacks before my organic chemistry exams. I failed the second midterm and I'm really scared of losing my merit scholarship if my GPA drops. Can someone help me see if I can drop the course or get tutoring?",
                    ],
                ]

                gr.Examples(
                    examples=example_scenarios,
                    inputs=[text_input],
                    label="Clinical Benchmark Scenarios",
                )

            # RIGHT COLUMN: OUTPUTS & CLINICAL DASHBOARD
            with gr.Column(scale=7):
                gr.Markdown("### 📊 Triage Assessment & Official Dispatch")

                # Emergency Status Banner
                status_banner_output = gr.HTML(
                    value="""
                    <div style="background:#f1f5f9; border:2px dashed #cbd5e1; border-radius:10px; padding:18px; text-align:center; color:#64748b;">
                        <em>System Ready. Record audio or enter a statement to evaluate safety and generate referral dispatch.</em>
                    </div>
                    """
                )

                # Spoken Dialogue Response
                spoken_dialogue_output = gr.Textbox(
                    lines=3,
                    interactive=False,
                    label="🗣️ Empathetic Spoken-Dialogue Response (For Direct Student Voice Playback)",
                    placeholder="Empathetic, trauma-informed spoken response will generate here...",
                )

                # Formatted Referral Ticket Card
                ticket_card_output = gr.HTML(
                    value="""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; text-align:center; color:#94a3b8; font-size:13px;">
                        Referral ticket details and receiving department SLA will render here upon submission.
                    </div>
                    """
                )

                with gr.Accordion("🔍 Diagnostic Inspector & Structured Raw JSON", open=False):
                    json_output = gr.JSON(label="Full Structured TriagePayload JSON")

        # EVENT BINDINGS
        submit_btn.click(
            fn=execute_triage_pipeline,
            inputs=[audio_input, text_input],
            outputs=[
                text_input,
                status_banner_output,
                spoken_dialogue_output,
                ticket_card_output,
                json_output,
            ],
        )

        def reset_fields():
            empty_banner = """
            <div style="background:#f1f5f9; border:2px dashed #cbd5e1; border-radius:10px; padding:18px; text-align:center; color:#64748b;">
                <em>System Ready. Record audio or enter a statement to evaluate safety and generate referral dispatch.</em>
            </div>
            """
            empty_card = """
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; text-align:center; color:#94a3b8; font-size:13px;">
                Referral ticket details and receiving department SLA will render here upon submission.
            </div>
            """
            return None, "", empty_banner, "", empty_card, {}

        clear_btn.click(
            fn=reset_fields,
            inputs=[],
            outputs=[
                audio_input,
                text_input,
                status_banner_output,
                spoken_dialogue_output,
                ticket_card_output,
                json_output,
            ],
        )

# ------------------------------------------------------------------------------
# PROTOTYPE LAUNCH ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting University Student Wellbeing Voice & Text Triage System...")
    logger.info("Initializing models and compiling Gradio UI...")
    logger.info("=" * 70)

    # Launch Gradio interface
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=custom_css,
    )
