from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

import os
import re
import math
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union
from collections import Counter
from datetime import datetime
from uuid import uuid4
import base64

import fitz  # PyMuPDF

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Content,
    GenerationConfig,
)

# Document AI ingest helper (optional file)
try:
    from docai_ingest import docai_extract_pdf_to_text
    _DOCAI_HELPER_AVAILABLE = True
except Exception:
    docai_extract_pdf_to_text = None
    _DOCAI_HELPER_AVAILABLE = False


# ----------------------------
# Setup
# ----------------------------

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))

CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "30"))
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "22000"))

GCP_PROJECT_ID = (os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or "").strip()
GCP_LOCATION = (os.getenv("GCP_LOCATION") or os.getenv("VERTEX_LOCATION") or "europe-west4").strip()

GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

MODEL_CHAT = (os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-001") or "").strip()
MODEL_COMPLIANCE = (os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash-001") or "").strip()

# Fire Cert report model (keeps chat/TGD behaviour unchanged)
MODEL_FIRECERT = (os.getenv("GEMINI_MODEL_FIRECERT", "gemini-2.0-pro-001") or "").strip()

# Verification pass toggle (default ON)
VERIFY_NUMERIC = (os.getenv("VERIFY_NUMERIC", "true") or "true").strip().lower() in ("1", "true", "yes", "y", "on")

app = FastAPI(docs_url="/swagger", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://raheemai.pages.dev",
        "https://raheem-ai.pages.dev",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

DOCAI_DIR = BASE_DIR / "parsed_docai"
DOCAI_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Vertex init
# ----------------------------

_VERTEX_READY = False
_VERTEX_ERR: Optional[str] = None


def ensure_vertex_ready() -> None:
    global _VERTEX_READY, _VERTEX_ERR
    if _VERTEX_READY:
        return
    try:
        if GOOGLE_CREDENTIALS_JSON and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            creds = json.loads(GOOGLE_CREDENTIALS_JSON)
            fd, path = tempfile.mkstemp(prefix="gcp-sa-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(creds, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

        if not GCP_PROJECT_ID:
            raise RuntimeError("Missing VERTEX_PROJECT_ID (or GCP_PROJECT_ID)")
        if not GCP_LOCATION:
            raise RuntimeError("Missing VERTEX_LOCATION (or GCP_LOCATION)")

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        _VERTEX_READY = True
        _VERTEX_ERR = None
    except Exception as e:
        _VERTEX_READY = False
        _VERTEX_ERR = str(e)


def get_model(model_name: str, system_prompt: str) -> GenerativeModel:
    ensure_vertex_ready()
    return GenerativeModel(model_name, system_instruction=[Part.from_text(system_prompt)])


def get_generation_config(is_compliance: bool) -> GenerationConfig:
    if is_compliance:
        return GenerationConfig(
            temperature=0.35,
            top_p=0.85,
            max_output_tokens=950,
        )
    return GenerationConfig(
        temperature=0.85,
        top_p=0.9,
        max_output_tokens=900,
    )


def get_verify_generation_config() -> GenerationConfig:
    return GenerationConfig(
        temperature=0.1,
        top_p=0.6,
        max_output_tokens=450,
    )


def get_firecert_generation_config() -> GenerationConfig:
    # Report drafting: lower creativity, longer output
    return GenerationConfig(
        temperature=0.35,
        top_p=0.85,
        max_output_tokens=2400,
    )


# ----------------------------
# Chat memory (server-side)
# ----------------------------

CHAT_STORE: Dict[str, List[Dict[str, str]]] = {}


def _trim_chat(chat_id: str) -> None:
    msgs = CHAT_STORE.get(chat_id, [])
    if not msgs:
        return

    if len(msgs) > CHAT_MAX_MESSAGES:
        msgs = msgs[-CHAT_MAX_MESSAGES:]

    total = 0
    kept_rev: List[Dict[str, str]] = []
    for m in reversed(msgs):
        total += len(m.get("content", ""))
        if total > CHAT_MAX_CHARS:
            break
        kept_rev.append(m)
    kept_rev.reverse()
    CHAT_STORE[chat_id] = kept_rev


def remember(chat_id: str, role: str, content: str) -> None:
    if not chat_id:
        return
    content = (content or "").strip()
    if not content:
        return
    CHAT_STORE.setdefault(chat_id, []).append({"role": role, "content": content})
    _trim_chat(chat_id)


def get_history(chat_id: str) -> List[Dict[str, str]]:
    return CHAT_STORE.get(chat_id, [])


def _normalize_incoming_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    hist: List[Dict[str, str]] = []
    for m in messages or []:
        r = (m.get("role") or "").lower().strip()
        c = (m.get("content") or "").strip()
        if r in ("user", "assistant") and c:
            hist.append({"role": r, "content": c})
    return hist


def _ensure_last_user_message(hist: List[Dict[str, str]], message: str) -> List[Dict[str, str]]:
    msg = (message or "").strip()
    if not msg:
        return hist
    if not hist:
        return [{"role": "user", "content": msg}]
    last = hist[-1]
    if last.get("role") != "user" or (last.get("content") or "").strip() != msg:
        hist = hist + [{"role": "user", "content": msg}]
    return hist


# ----------------------------
# Fire Cert (separate from chat)
# ----------------------------

FIRECERT_STORE: Dict[str, Dict[str, Any]] = {}

FIRECERT_MAX_IMAGES = int(os.getenv("FIRECERT_MAX_IMAGES", "40"))
FIRECERT_MAX_IMAGE_MB = int(os.getenv("FIRECERT_MAX_IMAGE_MB", "8"))


def _fc_get_project(project_id: str) -> Dict[str, Any]:
    project_id = (project_id or "").strip()
    if not project_id:
        raise ValueError("Missing project_id")

    if project_id not in FIRECERT_STORE:
        FIRECERT_STORE[project_id] = {
            "project_id": project_id,
            "created_utc": datetime.utcnow().isoformat(),
            "meta": {},
            "answers": {},   # qid -> {value, na, notes, refs}
            "images": [],    # [{id, name, mime, b64}]
        }
    return FIRECERT_STORE[project_id]


def _fc_meta(project: Dict[str, Any]) -> Dict[str, Any]:
    m = project.get("meta") or {}
    return m if isinstance(m, dict) else {}


def _boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1", "on"):
            return True
        if s in ("false", "no", "n", "0", "off"):
            return False
    return None


def _in_list(v: Any, options: List[str]) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v in options
    if isinstance(v, list):
        return any(x in options for x in v)
    return False


def _evaluate_show_if(show_if: Dict[str, Any], meta: Dict[str, Any], answers: Dict[str, Any]) -> bool:
    """
    Very simple gating for show_if:
      - {"some_qid": True/False/value}
      - {"some_qid__in": [..]}
      - {"some_qid__not_in": [..]}
    It checks answers first, then meta.
    """
    if not show_if:
        return True

    def get_val(key: str) -> Any:
        if key in answers:
            a = answers.get(key) or {}
            if isinstance(a, dict):
                return a.get("value")
        return meta.get(key)

    for k, expected in show_if.items():
        if k.endswith("__in"):
            base = k[:-4]
            val = get_val(base)
            if isinstance(expected, list):
                if not _in_list(val, expected):
                    return False
            else:
                if val != expected:
                    return False

        elif k.endswith("__not_in"):
            base = k[:-8]
            val = get_val(base)
            if isinstance(expected, list):
                if _in_list(val, expected):
                    return False
            else:
                if val == expected:
                    return False

        else:
            val = get_val(k)
            eb = _boolish(expected)
            vb = _boolish(val)
            if eb is not None:
                if vb is None or vb != eb:
                    return False
            else:
                if val != expected:
                    return False

    return True


def _firecert_visible_questions(schema: Dict[str, Any], meta: Dict[str, Any], answers: Dict[str, Any]) -> List[Dict[str, Any]]:
    visible: List[Dict[str, Any]] = []
    for step in schema.get("steps", []):
        for q in step.get("questions", []):
            show_if = q.get("show_if") or {}
            if _evaluate_show_if(show_if, meta, answers):
                visible.append(q)
    return visible


def _firecert_apply_auto_na(schema: Dict[str, Any], project: Dict[str, Any]) -> None:
    """
    Auto-NA any question that is not visible given meta + already-answered values.
    This keeps users away from irrelevant questions.
    """
    meta = _fc_meta(project)
    answers = project.get("answers") or {}
    if not isinstance(answers, dict):
        project["answers"] = {}
        answers = project["answers"]

    # Build a set of visible IDs
    visible = set()
    for step in schema.get("steps", []):
        for q in step.get("questions", []):
            if _evaluate_show_if(q.get("show_if") or {}, meta, answers):
                visible.add(q.get("id"))

    # For all questions in schema: if not visible and not answered -> set NA
    for step in schema.get("steps", []):
        for q in step.get("questions", []):
            qid = q.get("id")
            if not qid:
                continue
            if qid in visible:
                continue
            if qid in answers:
                continue
            answers[qid] = {
                "value": None,
                "na": True,
                "notes": "Auto-marked not applicable based on project inputs.",
                "refs": "",
                "updated_utc": datetime.utcnow().isoformat(),
            }


# ----------------------------
# Fire Cert schema (canonical universal schema + gates)
# ----------------------------

def firecert_schema_v1() -> Dict[str, Any]:
    return {
        "version": "v1",
        "steps": [
            {
                "id": "S0",
                "title": "STEP 0 — PROJECT SETUP & SCOPE",
                "questions": [
                    {"id": "s0_fsc", "label": "Is this a Fire Safety Certificate application under the Building Control Regulations?", "type": "bool",
                     "help": "If you are submitting through BCMS as an FSC application, choose Yes."},
                    {"id": "s0_tgd_edition", "label": "TGD B edition used", "type": "text",
                     "default": "TGD B 2024 – Volume 1 (Non-Dwellings)", "help": "Set the edition you are basing the report on."},

                    {"id": "s0_topmost_60m", "label": "Topmost floor height ≤ 60 m?", "type": "bool",
                     "help": "If the top occupied floor is above 60m, the building may be outside typical guidance scope."},
                    {"id": "s0_within_scope", "label": "Within scope of TGD B Volume 1 (not very unusual / non-complex)?", "type": "bool",
                     "help": "If the building is complex/unusual you may need a performance-based approach."},
                    {"id": "s0_alt_solution", "label": "If not within scope, is an alternative / performance-based approach proposed?", "type": "bool",
                     "show_if": {"s0_within_scope": False}, "help": "Choose Yes if you will use fire engineering / alternative codes."},
                    {"id": "s0_alt_standard", "label": "If Yes, describe method/standard", "type": "text",
                     "show_if": {"s0_alt_solution": True}, "help": "Example: Fire engineered solution; BS 9999; EN 12101; etc."},

                    {"id": "s0_works_type", "label": "Works type", "type": "select",
                     "options": ["New building", "Extension", "Material alteration", "Material change of use", "Combination"],
                     "help": "Select what kind of project this is."},
                    {"id": "s0_existing_reasonably_achievable",
                     "label": "If existing building works: are Sections 1–6 reasonably achievable?",
                     "type": "bool",
                     "show_if": {"s0_works_type__in": ["Extension", "Material alteration", "Material change of use", "Combination"]},
                     "help": "Existing constraints can justify compensatory measures."},

                    {"id": "s0_description", "label": "Brief building description", "type": "text",
                     "help": "Short narrative: use, key features, anything unusual."},
                    {"id": "s0_location", "label": "Site location / address", "type": "text",
                     "help": "Used for the report header."},
                    {"id": "s0_client", "label": "Client / Owner (if known)", "type": "text",
                     "help": "Used for the report header."},
                    {"id": "s0_prepared_by", "label": "Prepared by (name/company)", "type": "text",
                     "help": "Used for the report header."},

                    {"id": "s0_storeys_ag", "label": "Number of storeys above ground", "type": "number",
                     "help": "Enter storeys above ground level."},
                    {"id": "s0_storeys_bg", "label": "Number of basement storeys", "type": "number",
                     "help": "Enter basement levels (0 if none)."},
                    {"id": "s0_height_building", "label": "Height of building (per Appendix C definition) (m)", "type": "number",
                     "help": "Use the definition in TGD B Appendix C."},
                    {"id": "s0_height_top_storey", "label": "Height of topmost storey (m)", "type": "number",
                     "help": "Height of top occupied storey above ground level."},

                    {"id": "s0_separated_parts", "label": "Any separated parts (per 3.4.2)?", "type": "bool",
                     "help": "Separated parts can be treated as independent parts of a building for compartmentation logic."},
                    {"id": "s0_atrium", "label": "Any atrium?", "type": "bool",
                     "help": "Atria often trigger smoke control and special provisions."},
                    {"id": "s0_carpark", "label": "Any car park?", "type": "select",
                     "options": ["None", "Open", "Enclosed"],
                     "help": "Car parks can trigger smoke control and firefighting access requirements."},

                    {"id": "s0_purpose_groups", "label": "Purpose groups present (Table 1)", "type": "multiselect",
                     "options": [
                         "1(a) Office",
                         "1(b) Shop/Commercial",
                         "2(a) Residential care",
                         "2(b) Hotel/Student accommodation",
                         "3 Assembly/Leisure",
                         "4(a) Industrial (normal)",
                         "4(b) Shopping centre",
                         "5 Storage (normal)",
                         "6(b) Industrial high hazard",
                         "7(b) Storage high hazard"
                     ],
                     "help": "Select all that apply. If multiple uses exist, select them all."},

                    {"id": "s0_external_standards",
                     "label": "Any external codes/standards used in addition to TGD B?",
                     "type": "bool",
                     "help": "Select Yes if using BS/EN design guides beyond TGD."},
                    {"id": "s0_external_standards_list",
                     "label": "If Yes, list standards",
                     "type": "text",
                     "show_if": {"s0_external_standards": True},
                     "help": "Example: BS 9999, EN 12101, etc."},
                ],
            },

            {
                "id": "B1",
                "title": "SECTION 1 — MEANS OF WARNING & ESCAPE (B1)",
                "questions": [
                    {"id": "b1_occ_storey", "label": "Design occupancy per storey / compartment", "type": "text",
                     "help": "Provide occupancy assumptions used for exit and stair capacity checks."},
                    {"id": "b1_sleeping", "label": "Sleeping accommodation present?", "type": "bool",
                     "help": "Typically Yes for hotels/student accommodation and residential care."},
                    {"id": "b1_strategy", "label": "Evacuation strategy", "type": "select",
                     "options": ["Simultaneous", "Phased", "Progressive horizontal"],
                     "help": "Select the intended evacuation strategy."},

                    {"id": "b1_travel_actual", "label": "Maximum actual travel distance per storey (m)", "type": "number",
                     "help": "Measured travel distance to an exit / protected route."},
                    {"id": "b1_travel_basis", "label": "Travel distance basis used (single / dual direction)", "type": "select",
                     "options": ["Single direction", "Dual direction"],
                     "help": "Single direction usually has stricter limits."},
                    {"id": "b1_travel_ref", "label": "Table/paragraph reference used for travel distance check", "type": "text",
                     "help": "Enter what you relied on, e.g. TGD B Section 1 Table X."},

                    {"id": "b1_storey_exits", "label": "Number of storey exits per storey", "type": "text",
                     "help": "State exits by storey if it varies."},
                    {"id": "b1_final_exits", "label": "Number of final exits", "type": "number",
                     "help": "Total final exits to open air."},

                    {"id": "b1_exit_req_width", "label": "Exit capacity required width (mm)", "type": "number",
                     "help": "Based on occupancy and guidance references you used."},
                    {"id": "b1_exit_prov_width", "label": "Exit capacity provided width (mm)", "type": "number",
                     "help": "Sum of effective exit widths provided."},

                    {"id": "b1_stairs_count", "label": "Number of escape stairs", "type": "number",
                     "help": "Total escape stairs serving upper storeys."},
                    {"id": "b1_stairs_widths", "label": "Clear width of each stair (mm)", "type": "text",
                     "help": "List each stair width, e.g. Stair A 1200; Stair B 1200."},
                    {"id": "b1_stairs_persons", "label": "Persons served by each stair", "type": "text",
                     "help": "List how many persons rely on each stair."},
                    {"id": "b1_single_stair", "label": "Single stair building?", "type": "bool",
                     "help": "If yes, travel distance/height limits may be stricter."},
                    {"id": "b1_stair_smoke_control", "label": "Stair smoke ventilation/control provided?", "type": "bool",
                     "help": "If you have a smoke shaft / AOV / pressurisation, choose Yes."},

                    {"id": "b1_inner_rooms", "label": "Inner rooms present?", "type": "bool",
                     "help": "An inner room only has escape through another room (access room)."},
                    {"id": "b1_inner_detect", "label": "If inner rooms: automatic detection in access room?", "type": "bool",
                     "show_if": {"b1_inner_rooms": True},
                     "help": "Detection can be a compensatory measure for inner rooms."},
                    {"id": "b1_inner_vision", "label": "If inner rooms: vision panel / glazing provided?", "type": "bool",
                     "show_if": {"b1_inner_rooms": True},
                     "help": "Allows occupants to see if escape route is affected."},

                    {"id": "b1_fire_doors", "label": "Fire doors on escape routes?", "type": "bool",
                     "help": "Includes doors to protected corridors and stairs."},
                    {"id": "b1_hold_open", "label": "Hold-open devices provided?", "type": "bool",
                     "help": "If yes, they should release on alarm."},
                    {"id": "b1_electronic_locks", "label": "Electronically locked doors on escape routes?", "type": "bool",
                     "help": "Includes maglocks and access control doors on escape routes."},
                    {"id": "b1_unlock_alarm_power", "label": "If yes: unlock on fire alarm and power failure?", "type": "bool",
                     "show_if": {"b1_electronic_locks": True},
                     "help": "Escape doors must fail-safe / release."},

                    {"id": "b1_refuges", "label": "Refuge areas provided?", "type": "bool",
                     "help": "Usually expected in multi-storey buildings for assisted evacuation."},
                    {"id": "b1_refuge_location", "label": "If yes: location of refuges", "type": "text",
                     "show_if": {"b1_refuges": True},
                     "help": "Example: At each protected stair lobby level."},
                    {"id": "b1_refuge_comm", "label": "If yes: refuge communication system provided?", "type": "bool",
                     "show_if": {"b1_refuges": True},
                     "help": "Two-way communication may be required depending on strategy."},
                    {"id": "b1_evac_lift", "label": "Evacuation lift provided?", "type": "bool",
                     "help": "If yes, describe it in the report."},

                    {"id": "b1_alarm", "label": "Fire detection and alarm system provided?", "type": "bool",
                     "help": "If yes, describe category/standard if known."},
                    {"id": "b1_alarm_category", "label": "If yes: alarm category (if known)", "type": "text",
                     "show_if": {"b1_alarm": True},
                     "help": "Example: Category L2 (only if you know it)."},
                    {"id": "b1_emergency_lighting", "label": "Emergency lighting provided?", "type": "bool",
                     "help": "Emergency lighting on escape routes."},
                    {"id": "b1_signage", "label": "Fire safety signage provided?", "type": "bool",
                     "help": "Exit signs, directional signage, fire action notices."},

                    {"id": "b1_outcome", "label": "Section outcome flag", "type": "select",
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },

            {
                "id": "B2",
                "title": "SECTION 2 — INTERNAL FIRE SPREAD (LININGS) (B2)",
                "questions": [
                    {"id": "b2_linings_escape", "label": "Reaction-to-fire class of wall/ceiling linings on escape routes", "type": "text",
                     "help": "Enter the Euroclass (e.g. B-s2,d0) if known."},
                    {"id": "b2_linings_rooms", "label": "Reaction-to-fire class of wall/ceiling linings in rooms", "type": "text",
                     "help": "Enter the Euroclass if known."},
                    {"id": "b2_linings_circ", "label": "Reaction-to-fire class of wall/ceiling linings in circulation spaces", "type": "text",
                     "help": "Enter the Euroclass if known."},
                    {"id": "b2_table_ref", "label": "Table reference used for linings classification (e.g. Table 13)", "type": "text",
                     "help": "Enter the table/paragraph you relied on."},

                    {"id": "b2_deviation", "label": "Any deviations from tabulated classes?", "type": "bool",
                     "help": "Choose Yes if your lining class differs from guidance."},
                    {"id": "b2_compensatory", "label": "If Yes: compensatory measures proposed", "type": "text",
                     "show_if": {"b2_deviation": True},
                     "help": "Describe measures used to justify the deviation."},

                    {"id": "b2_special_risk", "label": "Any places of special fire risk (kitchens, plant, etc.)?", "type": "bool",
                     "help": "Choose Yes if present anywhere in the building."},

                    {"id": "b2_thermo_rooflights", "label": "Thermoplastic rooflights present?", "type": "bool",
                     "help": "Rooflights can have specific restrictions."},
                    {"id": "b2_thermo_diffusers", "label": "Thermoplastic lighting diffusers present?", "type": "bool",
                     "help": "Diffusers can contribute to internal fire spread."},
                    {"id": "b2_plastic_glazing_escape", "label": "Plastic glazing on escape routes?", "type": "bool",
                     "help": "Plastic glazing on escape routes can have restrictions."},

                    {"id": "b2_outcome", "label": "Section outcome flag", "type": "select",
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },

            {
                "id": "B3",
                "title": "SECTION 3 — INTERNAL FIRE SPREAD (STRUCTURE) (B3)",
                "questions": [
                    {"id": "b3_struct_system", "label": "Structural system type", "type": "text",
                     "help": "Example: steel frame, RC frame, timber, loadbearing masonry."},

                    {"id": "b3_required_fr", "label": "Required fire resistance (mins) per element (table reference)", "type": "text",
                     "help": "State required periods and the table/appendix used."},
                    {"id": "b3_proposed_fr_frame", "label": "Proposed fire resistance (mins) for frame", "type": "number",
                     "help": "Enter the proposed rating in minutes if known."},
                    {"id": "b3_proposed_fr_floors", "label": "Proposed fire resistance (mins) for floors", "type": "number",
                     "help": "Enter the proposed rating in minutes if known."},
                    {"id": "b3_proposed_fr_comp_walls", "label": "Proposed fire resistance (mins) for compartment walls", "type": "number",
                     "help": "Enter the proposed rating in minutes if known."},

                    {"id": "b3_compartment_areas", "label": "Compartment areas and heights (actual)", "type": "text",
                     "help": "List each compartment area/height and its use."},
                    {"id": "b3_compartment_max_ref", "label": "Maximum permitted compartment sizes (table reference)", "type": "text",
                     "help": "Enter the table/paragraph reference used."},
                    {"id": "b3_compartment_compliant", "label": "Compartmentation compliant with guidance?", "type": "bool",
                     "help": "Choose Yes if compartment sizes and separations match your chosen guidance basis."},

                    {"id": "b3_separated_parts", "label": "Any separated parts?", "type": "bool",
                     "help": "Choose Yes if separated parts are proposed."},
                    {"id": "b3_separated_parts_desc", "label": "If yes: describe separated parts strategy", "type": "text",
                     "show_if": {"b3_separated_parts": True},
                     "help": "Explain how separated parts are formed and separated."},

                    {"id": "b3_special_risk_rooms", "label": "List high-risk rooms (boilers, plant, stores, kitchens)", "type": "text",
                     "help": "List rooms considered higher fire risk."},
                    {"id": "b3_special_risk_enclosure", "label": "Fire-resisting enclosure and fire doors provided for high-risk rooms?", "type": "bool",
                     "help": "Choose Yes if enclosed in fire-resisting construction with appropriate doors."},

                    {"id": "b3_junction_strategy", "label": "Strategy at compartment floor/external wall junctions", "type": "text",
                     "help": "Describe approach, e.g. fire-stopping zone, balcony/recess arrangement."},

                    {"id": "b3_cavities_present", "label": "Cavities present?", "type": "bool",
                     "help": "Cavities can exist in walls, roofs, floors, and service zones."},
                    {"id": "b3_cavity_barriers", "label": "If cavities: cavity barriers provided?", "type": "bool",
                     "show_if": {"b3_cavities_present": True},
                     "help": "Choose Yes if cavity barriers are provided at required locations."},

                    {"id": "b3_services_penetrations", "label": "Services penetrating fire-resisting elements?", "type": "bool",
                     "help": "Penetrations require suitable fire stopping."},
                    {"id": "b3_firestopping_strategy", "label": "Fire stopping strategy defined?", "type": "bool",
                     "show_if": {"b3_services_penetrations": True},
                     "help": "Choose Yes if a fire stopping specification/strategy exists."},
                    {"id": "b3_fire_dampers", "label": "Fire dampers in ducts (where required)?", "type": "bool",
                     "help": "Choose Yes if dampers are included where ducts cross fire-resisting elements."},

                    {"id": "b3_outcome", "label": "Section outcome flag", "type": "select",
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },

            {
                "id": "B4",
                "title": "SECTION 4 — EXTERNAL FIRE SPREAD (B4)",
                "questions": [
                    {"id": "b4_wall_system", "label": "External wall system type", "type": "text",
                     "help": "Describe façade build-up in plain terms."},
                    {"id": "b4_facade_class", "label": "Reaction-to-fire classification of façade system", "type": "text",
                     "help": "Enter classification/test basis if known."},
                    {"id": "b4_combustible_cladding", "label": "Combustible insulation or cladding present?", "type": "bool",
                     "help": "Choose Yes if any combustible insulation/cladding is proposed."},
                    {"id": "b4_test_evidence", "label": "Fire test / evidence available?", "type": "bool",
                     "show_if": {"b4_combustible_cladding": True},
                     "help": "Choose Yes if you have test evidence/assessment for the façade system."},

                    {"id": "b4_boundary_distances", "label": "Boundary distances per elevation (m)", "type": "text",
                     "help": "List each elevation and its distance to boundary."},
                    {"id": "b4_unprotected_area", "label": "Percentage of unprotected area per elevation", "type": "text",
                     "help": "List glazing/openings proportion per elevation."},
                    {"id": "b4_space_sep_ref", "label": "Table/diagram used for space separation assessment", "type": "text",
                     "help": "Enter the reference used."},

                    {"id": "b4_attachments", "label": "Elements fixed to external wall present (balconies, PV, green walls)?", "type": "bool",
                     "help": "Choose Yes if any attachments could affect external spread."},
                    {"id": "b4_attachments_desc", "label": "If yes: describe external attachments", "type": "text",
                     "show_if": {"b4_attachments": True},
                     "help": "Describe balconies/PV supports/green walls etc."},

                    {"id": "b4_roof_covering_class", "label": "Roof covering classification", "type": "text",
                     "help": "Enter roof covering class if known."},
                    {"id": "b4_roof_required_class_ref", "label": "Required roof classification (table reference)", "type": "text",
                     "help": "Enter the guidance reference you used."},
                    {"id": "b4_plastic_rooflights", "label": "Plastic rooflights present?", "type": "bool",
                     "help": "Rooflights can affect external spread/performance."},
                    {"id": "b4_roof_pv_plant", "label": "Roof-mounted PV or plant present?", "type": "bool",
                     "help": "Choose Yes if PV/plant/amenity on roof is present."},
                    {"id": "b4_roof_pv_plant_desc", "label": "If yes: describe construction/fire resistance provisions", "type": "text",
                     "show_if": {"b4_roof_pv_plant": True},
                     "help": "Describe enclosures, separation, materials."},

                    {"id": "b4_outcome", "label": "Section outcome flag", "type": "select",
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },

            {
                "id": "B5",
                "title": "SECTION 5 — ACCESS & FACILITIES FOR FIRE SERVICE (B5)",
                "questions": [
                    {"id": "b5_access_route", "label": "Fire appliance access route provided?", "type": "bool",
                     "help": "Choose Yes if fire appliance access/turning is provided."},
                    {"id": "b5_appliance_distance", "label": "Distance from appliance parking to fire service access level (m)", "type": "number",
                     "help": "Measure from appliance hardstanding to access point."},
                    {"id": "b5_top_storey_vs_access", "label": "Height of topmost storey relative to access (m)", "type": "number",
                     "help": "Height above fire service access level."},

                    {"id": "b5_fire_main_required", "label": "Internal fire main required?", "type": "bool",
                     "help": "Choose Yes if a rising main is required/provided."},
                    {"id": "b5_fire_main_type", "label": "Type of internal fire main", "type": "select",
                     "show_if": {"b5_fire_main_required": True},
                     "options": ["Dry riser", "Wet riser"],
                     "help": "Select dry/wet riser type."},
                    {"id": "b5_fire_main_locations", "label": "Location of fire mains (per shaft)", "type": "text",
                     "show_if": {"b5_fire_main_required": True},
                     "help": "Describe where risers are located."},

                    {"id": "b5_ff_shaft_required", "label": "Firefighting shaft required?", "type": "bool",
                     "help": "Choose Yes if firefighting shafts are required/provided."},
                    {"id": "b5_ff_shaft_count", "label": "Number and location of firefighting shafts", "type": "text",
                     "show_if": {"b5_ff_shaft_required": True},
                     "help": "Describe shafts and their locations."},
                    {"id": "b5_ff_shaft_storeys", "label": "Storeys served by firefighting shafts", "type": "text",
                     "show_if": {"b5_ff_shaft_required": True},
                     "help": "List which storeys are served."},

                    {"id": "b5_hydrants", "label": "External hydrants provided?", "type": "bool",
                     "help": "Choose Yes if hydrants are present/available."},
                    {"id": "b5_hydrant_distance", "label": "Distance to nearest hydrant (m)", "type": "number",
                     "show_if": {"b5_hydrants": True},
                     "help": "Approximate distance to nearest hydrant."},
                    {"id": "b5_hydrant_flow_known", "label": "Hydrant flow/pressure known?", "type": "bool",
                     "show_if": {"b5_hydrants": True},
                     "help": "Choose Yes if flow/pressure data is available."},

                    {"id": "b5_boiler_fuel", "label": "Boiler rooms / fuel stores present?", "type": "bool",
                     "help": "Choose Yes if boiler/fuel store exists."},
                    {"id": "b5_elec_isolation", "label": "Electrical isolation for firefighting provided?", "type": "bool",
                     "help": "Choose Yes if firefighting electrical isolation is provided."},

                    {"id": "b5_outcome", "label": "Section outcome flag", "type": "select",
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },

            {
                "id": "B6",
                "title": "SECTION 6 — SMOKE CONTROL SYSTEMS",
                "questions": [
                    {"id": "b6_basements", "label": "Basements present?", "type": "bool",
                     "help": "Choose Yes if any basement storeys exist."},
                    {"id": "b6_atria", "label": "Atria present?", "type": "bool",
                     "help": "Choose Yes if atrium exists."},
                    {"id": "b6_lobbies_corridors", "label": "Protected lobbies/corridors present?", "type": "bool",
                     "help": "Choose Yes if protected lobbies/corridors exist."},
                    {"id": "b6_carparks", "label": "Car parks present?", "type": "bool",
                     "help": "Choose Yes if car parks exist (open or enclosed)."},
                    {"id": "b6_system_type", "label": "Smoke control system type per zone", "type": "text",
                     "show_if": {"b6_basements__in": [True, "true", "yes", "1"]},
                     "help": "Describe smoke control per zone (exhaust, pressurisation, AOV, etc.)."},
                    {"id": "b6_standard", "label": "Design standard used (EN 12101 / other)", "type": "text",
                     "help": "Enter standard/guide used for smoke control design."},
                    {"id": "b6_vent_defined", "label": "Vent sizes and locations defined?", "type": "bool",
                     "help": "Choose Yes if vents are sized and located."},
                    {"id": "b6_zones_defined", "label": "Smoke zones defined?", "type": "bool",
                     "help": "Choose Yes if smoke zones are defined."},
                    {"id": "b6_emergency_power", "label": "Emergency power supply provided?", "type": "bool",
                     "help": "Choose Yes if smoke control has emergency power."},
                    {"id": "b6_fail_safe", "label": "Fail-safe operation on power loss/alarm?", "type": "bool",
                     "help": "Choose Yes if vents/fans default to safe position on failure."},

                    {"id": "b6_outcome", "label": "Section outcome flag", "type": "select",
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },

            {
                "id": "B7",
                "title": "SECTION 7 — EXISTING BUILDINGS (IF APPLICABLE)",
                "questions": [
                    {"id": "b7_existing_fsc", "label": "Existing Fire Safety Certificate available?", "type": "bool",
                     "show_if": {"s0_works_type__in": ["Extension", "Material alteration", "Material change of use", "Combination"]},
                     "help": "Choose Yes if an existing FSC exists."},
                    {"id": "b7_existing_drawings", "label": "Existing fire safety drawings available?", "type": "bool",
                     "show_if": {"s0_works_type__in": ["Extension", "Material alteration", "Material change of use", "Combination"]},
                     "help": "Choose Yes if legacy fire strategy drawings exist."},
                    {"id": "b7_existing_fire_eng", "label": "Existing fire engineering / alternative solution retained?", "type": "bool",
                     "show_if": {"s0_works_type__in": ["Extension", "Material alteration", "Material change of use", "Combination"]},
                     "help": "Choose Yes if an existing alternative approach is being retained."},
                    {"id": "b7_comp_measures", "label": "Compensatory measures proposed? (Describe)", "type": "text",
                     "show_if": {"s0_works_type__in": ["Extension", "Material alteration", "Material change of use", "Combination"]},
                     "help": "Describe compensatory measures due to constraints."},

                    {"id": "b7_outcome", "label": "Section outcome flag", "type": "select",
                     "show_if": {"s0_works_type__in": ["Extension", "Material alteration", "Material change of use", "Combination"]},
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },

            {
                "id": "B8",
                "title": "SECTION 8 — SPRINKLERS",
                "questions": [
                    {"id": "b8_sprinklers", "label": "Sprinklers provided or required?", "type": "bool",
                     "help": "Choose Yes if sprinklers are included or required."},
                    {"id": "b8_hazard_class", "label": "Hazard classification (LH/OH/HH or equivalent)", "type": "text",
                     "show_if": {"b8_sprinklers": True},
                     "help": "Enter sprinkler hazard class if known."},
                    {"id": "b8_supply_type", "label": "Water supply type (town main / tank / pumped)", "type": "text",
                     "show_if": {"b8_sprinklers": True},
                     "help": "Describe sprinkler supply arrangement."},
                    {"id": "b8_tank_capacity", "label": "Tank capacity (if applicable)", "type": "number",
                     "show_if": {"b8_sprinklers": True},
                     "help": "Enter capacity if a tank is provided."},
                    {"id": "b8_omitted_heads", "label": "Any omitted areas or heads?", "type": "bool",
                     "show_if": {"b8_sprinklers": True},
                     "help": "Choose Yes if any areas are omitted or special."},
                    {"id": "b8_omitted_desc", "label": "If yes: describe omitted areas/heads", "type": "text",
                     "show_if": {"b8_omitted_heads": True},
                     "help": "Describe where sprinklers are omitted and why."},

                    {"id": "b8_outcome", "label": "Section outcome flag", "type": "select",
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },

            {
                "id": "B12",
                "title": "B12 — PROVISION OF INFORMATION",
                "questions": [
                    {"id": "b12_systems_list", "label": "Fire safety systems installed (list)", "type": "text",
                     "help": "List systems: alarm, emergency lighting, smoke control, sprinklers, fire mains, etc."},
                    {"id": "b12_commissioning", "label": "Commissioning certificates to be provided?", "type": "bool",
                     "help": "Choose Yes if certificates will be provided at handover."},
                    {"id": "b12_om", "label": "O&M manuals to be provided?", "type": "bool",
                     "help": "Choose Yes if O&M manuals will be provided."},
                    {"id": "b12_asbuilt", "label": "As-built fire safety drawings/report to be provided?", "type": "bool",
                     "help": "Choose Yes if as-built strategy drawings/report will be provided."},
                    {"id": "b12_logbook", "label": "Fire safety logbook / management plan provided?", "type": "bool",
                     "help": "Choose Yes if a logbook/management plan will be provided."},

                    {"id": "b12_outcome", "label": "Section outcome flag", "type": "select",
                     "options": ["Compliant using TGD tables only", "Compliant with compensatory measures", "Non-compliant (alternative solution required)", "Not assessed (pending info)"],
                     "help": "Used for report conclusions."},
                ],
            },
        ],
    }


# ----------------------------
# Prompts (tone stays the same)
# ----------------------------

FORMAT_RULES = """
Formatting rules (strict):
- Do NOT use Markdown.
- Do NOT use bullet points, numbered lists, asterisks, or hyphen lists.
- Write in short, clear paragraphs only.
- Never mention page numbers.
- If citing TGDs, cite only: document + section number and/or table number.
- Never invent section numbers, clause numbers, or table numbers.
- Only cite a section/table if it is visible in the provided SOURCES text.
- Do not reproduce large chunks of the TGDs. If quoting, quote only 1–2 short lines.
""".strip()

SYSTEM_PROMPT_NORMAL = f"""
You are Raheem AI.

Tone:
Friendly, calm, confident. Professional.
Humour style: dry, light, and occasional — think “architect on coffee #2”, not stand-up comedy.
You may add a short witty line when it fits, but never spam jokes and never derail the answer.
No emojis unless the user uses emojis first.

Memory:
You have access to the conversation history for THIS chat. Use it.
Never claim “this is our first conversation” if there is prior chat history.

{FORMAT_RULES}

General mode rules:
Answer general questions naturally using common knowledge.
If the user asks about Irish building regulations or TGDs, be cautious and do not guess numeric limits unless you can ground them in SOURCES.
""".strip()

SYSTEM_PROMPT_COMPLIANCE = f"""
You are Raheem AI in Evidence Mode for Irish building regulations, TGDs, BER/DEAP, fire safety, and accessibility.

Tone:
Still calm and professional, with minimal humour.
If you add humour at all, it must be a single short line and only when the user is not asking for strict numeric compliance.
When giving compliance numbers, be straight to the point.

Memory:
You have access to the conversation history for THIS chat. Use it.

{FORMAT_RULES}

Compliance rules (strict):
- Use provided SOURCES text as your primary grounding.
- If SOURCES do not support an exact numeric requirement (distances, widths, U-values, rise/going, travel distances), do NOT guess.
- If you cannot confirm a number from SOURCES, say so plainly and ask the user to clarify building type / situation or upload the relevant TGD.
- Do not invent “optimum” values unless the TGD explicitly states them in the SOURCES.
- If the user challenges you, re-check SOURCES and correct yourself.

Proof rule (critical):
- If you give a numeric requirement from a TGD, you MUST include a short verbatim quote (1–2 lines) from the SOURCES text you used.
- If you cannot quote it from SOURCES, do not give the number.
- Never mention page numbers in the quote.
""".strip()

VERIFY_PROMPT = f"""
You are a strict verifier for Irish TGD compliance answers.

Rules:
- You ONLY use the SOURCES text provided.
- If an answer includes any numeric requirement (mm, m, %, W/m²K, etc.), the exact number with its unit must be explicitly present in SOURCES.
- If the answer cites a section or table, the cited section/table label must be visible in SOURCES. If not visible, it must be removed or the answer must say it cannot confirm.
- If the answer gives numbers but does not include a 1–2 line verbatim quote from SOURCES that contains the number, it is NOT allowed.
- Output MUST be valid JSON only. No markdown.

Output JSON schema:
{{
  "ok": true/false,
  "reason": "short reason",
  "safe_answer": "a corrected safe answer that follows the formatting rules, includes quote: if numeric"
}}

{FORMAT_RULES}
""".strip()

SYSTEM_PROMPT_FIRECERT_REPORT = """
You are Raheem AI in Fire Safety Report drafting mode for an Irish Fire Safety Certificate (FSC) application.

Goal:
Draft a professional Fire Safety Report aligned to Part B / TGD B (Non-Dwellings, Volume 1) structure.
The report must be suitable for professional review (architect/fire consultant), not a final certified document.

Input you will receive:
- Project metadata (description, works type, purpose group(s), geometry)
- A structured set of Q&A answers (including "NA" where not applicable)
- Optional image attachments (descriptions + placeholders)

Rules:
- Do NOT use Markdown.
- Do NOT use bullet points.
- Use clear section headings as plain text lines (e.g. "SECTION 1 — MEANS OF ESCAPE (B1)").
- For each section, explicitly state applicability:
  - If NA: write "Not applicable to this project" and give a one-sentence justification based on inputs.
  - If applicable: describe the compliance strategy based strictly on user-provided inputs.
- Do NOT invent numeric limits or claim compliance with a table unless the user provided the reference.
- Where the user provides a clause/table/diagram reference, include it in a short "Reference:" sentence.
- If key information is missing, add a short "Information required:" sentence.
- Treat user answers as the source of truth. Never hallucinate drawings, systems, or dimensions.

Tone:
Professional, calm, clear. Typical Irish fire report tone. No jokes.

Output:
A complete report draft with:
- Cover-style header block (Project, Client, Location, Date, Prepared by)
- Introduction, scope, method
- SECTION 1 to SECTION 8 and B12 (where applicable)
- Summary / conclusions and list of outstanding items (plain paragraphs, no bullets)
""".strip()


def build_gemini_contents(history: List[Dict[str, str]], user_message: str, sources_text: str) -> List[Content]:
    contents: List[Content] = []
    for m in history or []:
        r = (m.get("role") or "").lower().strip()
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if r == "user":
            contents.append(Content(role="user", parts=[Part.from_text(c)]))
        elif r == "assistant":
            contents.append(Content(role="model", parts=[Part.from_text(c)]))

    final_user = (user_message or "").strip()
    if sources_text and sources_text.strip():
        final_user = (final_user + "\n\nSOURCES (use for evidence):\n" + sources_text).strip()

    if not contents:
        contents.append(Content(role="user", parts=[Part.from_text(final_user)]))
    else:
        if contents[-1].role == "user":
            contents[-1] = Content(role="user", parts=[Part.from_text(final_user)]))
        else:
            contents.append(Content(role="user", parts=[Part.from_text(final_user)]))

    return contents


# ----------------------------
# Retrieval helpers
# ----------------------------

STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "a", "an", "for", "on", "with", "is", "are", "be", "as", "at", "from", "by",
    "that", "this", "it", "your", "you", "we", "they", "their", "there", "what", "which", "when", "where", "how",
    "can", "shall", "should", "must", "may", "not", "than", "then", "into", "onto", "also", "such"
}


def clean_text(s: str) -> str:
    s = (s or "").replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-/\.%]*", t)
    toks = [x for x in toks if len(x) >= 2 and x not in STOPWORDS]
    return toks


PDF_INDEX: Dict[str, Dict[str, Any]] = {}
DOCAI_INDEX: Dict[str, Dict[str, Any]] = {}


def list_pdfs() -> List[str]:
    files = []
    for p in PDF_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p.name)
    files.sort()
    return files


def auto_pin_pdf(question: str) -> Optional[str]:
    q = (question or "").lower()
    pdfs = set(list_pdfs())

    def pick_any(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in pdfs:
                return c
        return None

    if any(w in q for w in ["stairs", "stair", "staircase", "riser", "rise", "going", "pitch", "headroom", "handrail", "balustrade", "landing"]):
        return pick_any([
            "Technical Guidance Document K.pdf",
            "TGD K.pdf",
            "Technical Guidance Document K (2020).pdf",
            "Technical Guidance Document K.pdf".replace("  ", " ")
        ])

    if any(w in q for w in ["fire", "escape", "means of escape", "travel distance", "sprinkler", "smoke", "compartment", "car park", "corridor", "protected route"]):
        return pick_any([
            "Technical Guidance Document B Non Dwellings.pdf",
            "Technical Guidance Document B Dwellings.pdf",
        ])

    if any(w in q for w in ["u-value", "u value", "y-value", "thermal", "insulation", "ber", "deap", "energy", "renovation", "major renovation"]):
        return pick_any([
            "Technical Guidance Document L Dwellings.pdf",
            "Technical Guidance Document L Non Dwellings.pdf",
        ])

    if any(w in q for w in ["access", "accessible", "wheelchair", "ramp", "dac", "part m"]):
        return pick_any([
            "Technical Guidance Document M.pdf",
        ])

    return None


def _docai_chunk_files_for(pdf_name: str) -> List[Path]:
    stem = Path(pdf_name).stem
    out = []
    for p in DOCAI_DIR.iterdir():
        if p.is_file() and p.name.startswith(stem + "_p") and p.suffix.lower() == ".txt":
            out.append(p)
    out.sort(key=lambda x: x.name)
    return out


def _parse_range_from_chunk_filename(path: Path) -> Optional[Tuple[int, int]]:
    m = re.search(r"_p(\d+)\-(\d+)\.txt$", path.name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def ensure_docai_indexed(pdf_name: str) -> None:
    if pdf_name in DOCAI_INDEX:
        return
    chunk_files = _docai_chunk_files_for(pdf_name)
    if not chunk_files:
        return

    chunks = []
    df = Counter()

    for p in chunk_files:
        rng = _parse_range_from_chunk_filename(p)
        if not rng:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        toks = tokenize(txt.lower())
        tf = Counter(toks)
        df.update(set(tf.keys()))

        chunks.append({"range": rng, "text": txt, "tf": tf, "len": len(toks)})

    if not chunks:
        return

    avgdl = sum(c["len"] for c in chunks) / max(1, len(chunks))
    DOCAI_INDEX[pdf_name] = {"chunks": chunks, "df": df, "avgdl": avgdl, "N": len(chunks)}


def bm25_score(query_toks: List[str], tf: Counter, df: Counter, N: int, dl: int, avgdl: float) -> float:
    k1 = 1.2
    b = 0.75
    score = 0.0
    for t in query_toks:
        if t not in tf:
            continue
        n = df.get(t, 0)
        idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
        f = tf[t]
        denom = f + k1 * (1 - b + b * (dl / (avgdl or 1.0)))
        score += idf * (f * (k1 + 1) / (denom or 1.0))
    return score


def search_docai_chunks(pdf_name: str, question: str, k: int = 3) -> List[Dict[str, Any]]:
    ensure_docai_indexed(pdf_name)
    idx = DOCAI_INDEX.get(pdf_name)
    if not idx:
        return []

    qt = tokenize(question)
    if not qt:
        return []

    df = idx["df"]
    N = idx["N"]
    avgdl = idx["avgdl"]

    scored = []
    for i, ch in enumerate(idx["chunks"]):
        s = bm25_score(qt, ch["tf"], df, N, ch["len"], avgdl)
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    out = []
    for i, _ in scored[:k]:
        out.append({"range": idx["chunks"][i]["range"], "text": idx["chunks"][i]["text"]})
    return out


def index_pdf(pdf_path: Path) -> None:
    name = pdf_path.name
    doc = fitz.open(pdf_path)
    try:
        page_tf: List[Counter] = []
        df = Counter()
        page_len: List[int] = []

        for i in range(doc.page_count):
            txt = clean_text(doc.load_page(i).get_text("text") or "")
            toks = tokenize(txt.lower())
            tf = Counter(toks)
            page_tf.append(tf)
            df.update(set(tf.keys()))
            page_len.append(len(toks))

        avgdl = (sum(page_len) / len(page_len)) if page_len else 0.0

        PDF_INDEX[name] = {
            "page_tf": page_tf,
            "df": df,
            "page_len": page_len,
            "avgdl": avgdl,
            "pages": doc.page_count,
        }
    finally:
        doc.close()


def ensure_indexed(pdf_name: str) -> None:
    if pdf_name in PDF_INDEX:
        return
    p = PDF_DIR / pdf_name
    if p.exists():
        index_pdf(p)


def search_pages(pdf_name: str, question: str, k: int = 5) -> List[int]:
    ensure_indexed(pdf_name)
    idx = PDF_INDEX.get(pdf_name)
    if not idx:
        return []

    qt = tokenize(question)
    if not qt:
        return []

    N = idx["pages"]
    df = idx["df"]
    avgdl = idx["avgdl"]
    page_tf = idx["page_tf"]
    page_len = idx["page_len"]

    scored = []
    for i, tf in enumerate(page_tf):
        s = bm25_score(qt, tf, df, N, page_len[i], avgdl)
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:k]]


def extract_pages_text(pdf_path: Path, pages: List[int], max_chars_per_page: int = 2200) -> str:
    doc = fitz.open(pdf_path)
    try:
        chunks = []
        for p in pages:
            if p < 0 or p >= doc.page_count:
                continue
            txt = clean_text(doc.load_page(p).get_text("text") or "")
            if len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + "..."
            chunks.append(f"[{pdf_path.name} excerpt]\n{txt}")
        return "\n\n".join(chunks).strip()
    finally:
        doc.close()


def build_sources_bundle(selected: List[Tuple[str, Union[List[int], List[Tuple[int, int]]], str]]) -> str:
    pieces = []
    for pdf_name, items, mode in selected:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            continue

        if mode == "docai":
            ensure_docai_indexed(pdf_name)
            idx = DOCAI_INDEX.get(pdf_name)
            if not idx:
                continue

            for rng in items:  # type: ignore
                a, b = rng
                txt = ""
                for ch in idx["chunks"]:
                    if ch["range"] == (a, b):
                        txt = ch["text"]
                        break
                if txt:
                    clipped = txt.strip()
                    if len(clipped) > 6500:
                        clipped = clipped[:6500] + "..."
                    pieces.append(f"[{pdf_name} excerpt]\n{clipped}")
        else:
            pages = items  # type: ignore
            pieces.append(extract_pages_text(pdf_path, pages))

    return ("\n\n".join([p for p in pieces if p]).strip())


def select_sources(
    question: str,
    pdf_pin: Optional[str] = None,
    max_docs: int = 3,
    pages_per_doc: int = 3,
    docai_chunks_per_doc: int = 2,
) -> List[Tuple[str, Union[List[int], List[Tuple[int, int]]], str]]:
    pdfs = list_pdfs()
    if not pdfs:
        return []

    chosen: List[Tuple[str, Union[List[int], List[Tuple[int, int]]], str]] = []
    know_docai = {name for name in pdfs if _docai_chunk_files_for(name)}

    def pick_for_pdf(pdf_name: str) -> Optional[Tuple[str, Union[List[int], List[Tuple[int, int]]], str]]:
        if pdf_name in know_docai:
            chunks = search_docai_chunks(pdf_name, question, k=docai_chunks_per_doc)
            if chunks:
                ranges = [c["range"] for c in chunks]
                return (pdf_name, ranges, "docai")

        pages = search_pages(pdf_name, question, k=pages_per_doc)
        if pages:
            return (pdf_name, pages, "pymupdf")
        return None

    if pdf_pin and pdf_pin in pdfs:
        picked = pick_for_pdf(pdf_pin)
        return [picked] if picked else []

    picks = []
    for pdf_name in pdfs:
        picked = pick_for_pdf(pdf_name)
        if picked:
            score = 10.0 if picked[2] == "docai" else 5.0
            picks.append((score, picked))
    picks.sort(key=lambda x: x[0], reverse=True)

    for _, picked in picks[:max_docs]:
        chosen.append(picked)

    return chosen


# ----------------------------
# Compliance intent detection
# ----------------------------

COMPLIANCE_KEYWORDS = [
    "tgd", "technical guidance", "building regulations", "building regs",
    "part a", "part b", "part c", "part d", "part e", "part f", "part g", "part h",
    "part j", "part k", "part l", "part m",
    "fire", "escape", "travel distance", "compartment", "smoke", "fire cert",
    "access", "accessible", "wheelchair", "dac",
    "ber", "deap", "u-value", "y-value", "airtight", "thermal bridge",
    "means of escape", "compartmentation", "evacuation",
]

PART_PATTERN = re.compile(r"\bpart\s*[a-m]\b", re.I)
TECH_PATTERN = re.compile(
    r"\b(tgd|technical guidance|building regulations|building regs|ber|deap|fire cert|dac|means of escape)\b",
    re.I
)

NUMERIC_COMPLIANCE_TRIGGERS = [
    "minimum", "maximum", "max", "min", "limit", "limits",
    "distance", "width", "widths", "lengths", "length", "height", "u-value", "y-value",
    "rise", "riser", "going", "pitch",
    "stairs", "stair", "staircase", "landing", "headroom",
    "dimension", "dimensions",
    "escape", "travel distance"
]


def is_numeric_compliance_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in NUMERIC_COMPLIANCE_TRIGGERS)


def should_use_docs(q: str, force_docs: bool = False) -> bool:
    if force_docs:
        return True
    ql = (q or "").lower()
    if is_numeric_compliance_question(ql):
        return True
    if PART_PATTERN.search(ql):
        return True
    if TECH_PATTERN.search(ql):
        return True
    return any(k in ql for k in COMPLIANCE_KEYWORDS)


# ----------------------------
# Output sanitizers
# ----------------------------

PAGE_MENTION_RE = re.compile(r"\bpage\s*\d+\b", re.I)
BULLET_RE = re.compile(r"(^|\n)\s*(\*|-|•|\d+\.)\s+", re.M)
REF_RE = re.compile(r"\b(Section|Table|Clause|Figure|Diagram)\s*[A-Za-z0-9][A-Za-z0-9\.\-]*\b", re.I)

ANY_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")
NUM_WITH_UNIT_RE = re.compile(r"\b\d+(\.\d+)?\s*(mm|m|metre|meter|w/m²k|w\/m2k|%)\b", re.I)


def strip_bullets_streaming(delta: str) -> str:
    return BULLET_RE.sub(lambda m: m.group(1), delta)


def _sources_contain_any_number(sources_text: str) -> bool:
    return bool(ANY_NUMBER_RE.search(sources_text or ""))


def _looks_like_numeric_answer(text: str) -> bool:
    return bool(NUM_WITH_UNIT_RE.search(text or ""))


def postprocess_final_answer(final_text: str, sources_text: str, compliance: bool) -> str:
    t = final_text or ""
    t = PAGE_MENTION_RE.sub("", t)
    t = BULLET_RE.sub(lambda m: m.group(1), t)

    if compliance:
        src = (sources_text or "").lower()
        if src.strip():
            if ("section" not in src) and ("table" not in src) and ("clause" not in src) and ("diagram" not in src) and ("figure" not in src):
                if REF_RE.search(t):
                    t = REF_RE.sub("", t).strip()
                    t = (t + "\n\nI can’t confidently point to the exact section or table from the extracted text I have right now. If you re-upload a clearer extract, I’ll re-check and cite it properly.").strip()
        else:
            t = REF_RE.sub("", t).strip()

    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


# ----------------------------
# Verification (silent double-check)
# ----------------------------

def _run_verifier(user_msg: str, draft_answer: str, sources_text: str) -> Tuple[bool, str]:
    """
    Returns (ok, safe_answer). If verifier fails, safe_answer is a corrected safe output.
    """
    if not VERIFY_NUMERIC:
        return True, draft_answer

    # Only verify when we have sources
    if not (sources_text and sources_text.strip()):
        return True, draft_answer

    # Only verify when answer contains numbers OR question is numeric compliance
    needs = is_numeric_compliance_question(user_msg) or _looks_like_numeric_answer(draft_answer) or bool(ANY_NUMBER_RE.search(draft_answer or ""))
    if not needs:
        return True, draft_answer

    try:
        ensure_vertex_ready()
        if not _VERTEX_READY:
            return True, draft_answer

        verifier = get_model(MODEL_COMPLIANCE, VERIFY_PROMPT)

        verify_input = (
            "USER QUESTION:\n" + (user_msg or "").strip() + "\n\n"
            "DRAFT ANSWER:\n" + (draft_answer or "").strip() + "\n\n"
            "SOURCES:\n" + (sources_text or "").strip()
        )

        resp = verifier.generate_content(
            [Content(role="user", parts=[Part.from_text(verify_input)])],
            generation_config=get_verify_generation_config(),
            stream=False
        )

        raw = (getattr(resp, "text", "") or "").strip()

        # Try to extract JSON object from any surrounding text (just in case)
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            raw = m.group(0)

        data = json.loads(raw)
        ok = bool(data.get("ok", False))
        safe_answer = (data.get("safe_answer") or "").strip()

        if ok and safe_answer:
            return True, safe_answer

        # If not ok, enforce a safe fallback message
        if safe_answer:
            return False, safe_answer

        fallback = (
            "I can’t safely confirm that numeric requirement from the extracted TGD text I have available right now.\n\n"
            "If you upload a clearer copy or tell me the exact table or section you want checked, I’ll answer directly from the text and include a short quote."
        )
        return False, fallback

    except Exception:
        # If verifier fails for any reason, don't break the app; just return draft.
        return True, draft_answer


# ----------------------------
# Fire Cert report assembly helpers
# ----------------------------

def _fc_build_report_input(project: Dict[str, Any]) -> str:
    meta = project.get("meta") or {}
    answers = project.get("answers") or {}
    images = project.get("images") or []

    lines = []
    lines.append("PROJECT META (user provided):")
    if isinstance(meta, dict):
        for k, v in meta.items():
            lines.append(f"{k}: {v}")

    lines.append("\nANSWERS (qid -> value / NA / notes / refs):")
    if isinstance(answers, dict):
        for qid, a in answers.items():
            if not isinstance(a, dict):
                continue
            na = bool(a.get("na", False))
            val = a.get("value", None)
            notes = (a.get("notes") or "").strip()
            refs = (a.get("refs") or "").strip()
            lines.append(f"{qid}:")
            lines.append(f"value: {val}")
            lines.append(f"na: {na}")
            if notes:
                lines.append(f"notes: {notes}")
            if refs:
                lines.append(f"refs: {refs}")

    if images:
        lines.append("\nIMAGES (placeholders):")
        for img in images:
            lines.append(f"[IMAGE {img.get('id')} | {img.get('name')} | {img.get('mime')}]")

    return "\n".join(lines).strip()


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def root():
    return {"ok": True, "app": "Raheem AI", "time": datetime.utcnow().isoformat()}


@app.get("/health")
def health():
    ensure_vertex_ready()
    return {
        "ok": True,
        "vertex_ready": _VERTEX_READY,
        "vertex_error": _VERTEX_ERR,
        "pdf_count": len(list_pdfs()),
        "models": {"chat": MODEL_CHAT, "compliance": MODEL_COMPLIANCE, "firecert": MODEL_FIRECERT},
        "project": GCP_PROJECT_ID,
        "location": GCP_LOCATION,
        "docai_helper": _DOCAI_HELPER_AVAILABLE,
        "docai_processor_id_present": bool((os.getenv("DOCAI_PROCESSOR_ID") or "").strip()),
        "docai_location_present": bool((os.getenv("DOCAI_LOCATION") or "").strip()),
        "verify_numeric": VERIFY_NUMERIC,
    }


@app.get("/pdfs")
def pdfs():
    return {"pdfs": list_pdfs()}


def _split_docai_combined_to_chunks(combined: str) -> List[Tuple[Tuple[int, int], str]]:
    if not combined:
        return []
    pattern = re.compile(r"---\s*DOC_AI_PAGES\s+(\d+)\-(\d+)\s*---", re.IGNORECASE)
    parts = pattern.split(combined)
    out: List[Tuple[Tuple[int, int], str]] = []
    if len(parts) < 4:
        return out

    i = 1
    while i + 2 < len(parts):
        a = int(parts[i])
        b = int(parts[i + 1])
        txt = parts[i + 2].strip()
        out.append(((a, b), txt))
        i += 3
    return out


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"ok": False, "error": "Only PDF files are allowed."}, status_code=400)

    raw = file.file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse({"ok": False, "error": f"File too large. Max {MAX_UPLOAD_MB}MB."}, status_code=400)

    safe_name = Path(file.filename).name
    dest = PDF_DIR / safe_name
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        i = 2
        while True:
            cand = PDF_DIR / f"{stem}_{i}{suffix}"
            if not cand.exists():
                dest = cand
                break
            i += 1

    with open(dest, "wb") as f:
        f.write(raw)

    PDF_INDEX.pop(dest.name, None)
    DOCAI_INDEX.pop(dest.name, None)

    docai_ok = False
    docai_chunks_saved = 0
    docai_error = None

    try:
        if _DOCAI_HELPER_AVAILABLE and docai_extract_pdf_to_text:
            if (os.getenv("DOCAI_PROCESSOR_ID") or "").strip() and (os.getenv("DOCAI_LOCATION") or "").strip():
                combined_text, _chunk_ranges = docai_extract_pdf_to_text(str(dest), chunk_pages=15)
                chunks = _split_docai_combined_to_chunks(combined_text)
                stem = dest.stem
                for (a, b), txt in chunks:
                    out_path = DOCAI_DIR / f"{stem}_p{a}-{b}.txt"
                    out_path.write_text(txt, encoding="utf-8", errors="ignore")
                    docai_chunks_saved += 1
                docai_ok = docai_chunks_saved > 0
    except Exception as e:
        docai_error = repr(e)

    return {
        "ok": True,
        "stored_as": dest.name,
        "pdf_count": len(list_pdfs()),
        "docai": {
            "attempted": bool(_DOCAI_HELPER_AVAILABLE and docai_extract_pdf_to_text),
            "ok": docai_ok,
            "chunks_saved": docai_chunks_saved,
            "error": docai_error,
        }
    }


# ----------------------------
# Fire Cert endpoints (new feature, isolated from chat)
# ----------------------------

@app.get("/firecert/schema")
def firecert_get_schema():
    return {"ok": True, "schema": firecert_schema_v1()}


@app.post("/firecert/project/init")
def firecert_project_init(payload: Dict[str, Any] = Body(...)):
    project_id = (payload.get("project_id") or "").strip() or str(uuid4())
    p = _fc_get_project(project_id)

    meta = payload.get("meta") or {}
    if isinstance(meta, dict) and meta:
        p["meta"].update(meta)

    # Auto-NA based on current meta state
    _firecert_apply_auto_na(firecert_schema_v1(), p)

    return {"ok": True, "project": {"project_id": project_id, "created_utc": p["created_utc"], "meta": p["meta"]}}


@app.get("/firecert/project")
def firecert_project_get(project_id: str = Query("")):
    p = _fc_get_project(project_id)
    return {"ok": True, "project": p}


@app.post("/firecert/project/meta")
def firecert_project_meta(payload: Dict[str, Any] = Body(...)):
    project_id = (payload.get("project_id") or "").strip()
    if not project_id:
        return JSONResponse({"ok": False, "error": "Missing project_id"}, status_code=400)

    p = _fc_get_project(project_id)
    meta_patch = payload.get("meta") or {}
    if isinstance(meta_patch, dict) and meta_patch:
        p["meta"].update(meta_patch)

    _firecert_apply_auto_na(firecert_schema_v1(), p)
    return {"ok": True, "project_id": project_id, "meta": p["meta"]}


@app.post("/firecert/answer")
def firecert_save_answer(payload: Dict[str, Any] = Body(...)):
    project_id = (payload.get("project_id") or "").strip()
    qid = (payload.get("qid") or "").strip()
    if not project_id or not qid:
        return JSONResponse({"ok": False, "error": "Missing project_id or qid"}, status_code=400)

    p = _fc_get_project(project_id)

    value = payload.get("value")
    na = bool(payload.get("na", False))
    notes = (payload.get("notes") or "").strip()
    refs = (payload.get("refs") or "").strip()

    p["answers"][qid] = {"value": value, "na": na, "notes": notes, "refs": refs, "updated_utc": datetime.utcnow().isoformat()}

    # Re-run auto-NA after each answer (keeps relevance tight)
    _firecert_apply_auto_na(firecert_schema_v1(), p)

    return {"ok": True, "saved": {"qid": qid}, "project_id": project_id}


@app.post("/firecert/upload-image")
def firecert_upload_image(project_id: str = Query(""), file: UploadFile = File(...)):
    project_id = (project_id or "").strip()
    if not project_id:
        return JSONResponse({"ok": False, "error": "Missing project_id"}, status_code=400)

    p = _fc_get_project(project_id)

    if len(p["images"]) >= FIRECERT_MAX_IMAGES:
        return JSONResponse({"ok": False, "error": f"Too many images. Max {FIRECERT_MAX_IMAGES}."}, status_code=400)

    raw = file.file.read()
    if len(raw) > FIRECERT_MAX_IMAGE_MB * 1024 * 1024:
        return JSONResponse({"ok": False, "error": f"Image too large. Max {FIRECERT_MAX_IMAGE_MB}MB."}, status_code=400)

    mime = (file.content_type or "").strip().lower()
    if mime not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        return JSONResponse({"ok": False, "error": "Only PNG/JPEG/WEBP images are allowed."}, status_code=400)

    img_id = str(uuid4())
    name = Path(file.filename or f"image_{img_id}.png").name
    b64 = base64.b64encode(raw).decode("utf-8")

    p["images"].append({"id": img_id, "name": name, "mime": mime, "b64": b64, "uploaded_utc": datetime.utcnow().isoformat()})

    return {"ok": True, "image": {"id": img_id, "name": name, "mime": mime}}


@app.post("/firecert/next-question")
def firecert_next_question(payload: Dict[str, Any] = Body(...)):
    project_id = (payload.get("project_id") or "").strip()
    if not project_id:
        return JSONResponse({"ok": False, "error": "Missing project_id"}, status_code=400)

    p = _fc_get_project(project_id)
    schema = firecert_schema_v1()
    meta = _fc_meta(p)
    answers = p.get("answers") or {}

    # Ensure auto-NA is up-to-date before computing next visible
    _firecert_apply_auto_na(schema, p)
    answers = p.get("answers") or {}

    # Find first visible question that is not answered OR answered but empty and not NA
    for step in schema.get("steps", []):
        for q in step.get("questions", []):
            qid = q.get("id")
            if not qid:
                continue

            if not _evaluate_show_if(q.get("show_if") or {}, meta, answers):
                continue

            a = answers.get(qid)
            if not a:
                return {"ok": True, "step": step.get("id"), "step_title": step.get("title"), "question": q}

            if isinstance(a, dict):
                if bool(a.get("na", False)):
                    continue
                val = a.get("value", None)
                # treat None/"" as unanswered
                if val is None:
                    return {"ok": True, "step": step.get("id"), "step_title": step.get("title"), "question": q}
                if isinstance(val, str) and not val.strip():
                    return {"ok": True, "step": step.get("id"), "step_title": step.get("title"), "question": q}

    return {"ok": True, "done": True, "message": "All relevant questions are complete (or marked NA)."}


@app.post("/firecert/generate-report")
def firecert_generate_report(payload: Dict[str, Any] = Body(...)):
    project_id = (payload.get("project_id") or "").strip()
    if not project_id:
        return JSONResponse({"ok": False, "error": "Missing project_id"}, status_code=400)

    ensure_vertex_ready()
    if not _VERTEX_READY:
        return JSONResponse({"ok": False, "error": (_VERTEX_ERR or "Vertex not ready")}, status_code=503)

    project = _fc_get_project(project_id)

    # Optional meta patch-in at generation time
    meta_patch = payload.get("meta") or {}
    if isinstance(meta_patch, dict) and meta_patch:
        project["meta"].update(meta_patch)

    # Ensure auto-NA is applied before drafting
    _firecert_apply_auto_na(firecert_schema_v1(), project)

    report_input = _fc_build_report_input(project)

    model_name = MODEL_FIRECERT or MODEL_COMPLIANCE
    model = get_model(model_name=model_name, system_prompt=SYSTEM_PROMPT_FIRECERT_REPORT)

    user_msg = (
        "Draft a Fire Safety Report for FSC review based only on the provided inputs.\n\n"
        + report_input
    )

    resp = model.generate_content(
        [Content(role="user", parts=[Part.from_text(user_msg)])],
        generation_config=get_firecert_generation_config(),
        stream=False
    )

    text = (getattr(resp, "text", "") or "").strip()
    text = postprocess_final_answer(text, sources_text="", compliance=False)

    return {"ok": True, "project_id": project_id, "report": text}


# ----------------------------
# Streaming core
# ----------------------------

def _stream_answer(
    chat_id: str,
    message: str,
    force_docs: bool,
    pdf: Optional[str],
    page_hint: Optional[int],
    messages: Optional[List[Dict[str, Any]]] = None
):
    try:
        if not message.strip():
            yield "event: error\ndata: No message provided.\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        chat_id = (chat_id or "").strip()
        user_msg = (message or "").strip()

        # Canonical history
        if isinstance(messages, list) and messages:
            hist = _normalize_incoming_messages(messages)
            hist = _ensure_last_user_message(hist, user_msg)
            if chat_id:
                CHAT_STORE[chat_id] = hist[-CHAT_MAX_MESSAGES:]
                _trim_chat(chat_id)
            history_for_prompt = CHAT_STORE.get(chat_id, hist)
        else:
            history_for_prompt = get_history(chat_id) if chat_id else []
            if chat_id:
                remember(chat_id, "user", user_msg)
                history_for_prompt = get_history(chat_id)

        # Decide compliance intent
        use_docs_intent = should_use_docs(user_msg, force_docs=force_docs)
        numeric_compliance = use_docs_intent and is_numeric_compliance_question(user_msg)

        sources_text = ""
        if use_docs_intent and list_pdfs():
            auto_pdf = pdf or auto_pin_pdf(user_msg)
            selected = select_sources(
                user_msg,
                pdf_pin=auto_pdf,
                max_docs=3,
                pages_per_doc=3,
                docai_chunks_per_doc=2
            )
            sources_text = build_sources_bundle(selected)

        used_docs_flag = bool(sources_text and sources_text.strip())
        is_compliance = use_docs_intent

        model_name = MODEL_COMPLIANCE if is_compliance else MODEL_CHAT
        system_prompt = SYSTEM_PROMPT_COMPLIANCE if is_compliance else SYSTEM_PROMPT_NORMAL

        # HARD RULE: numeric compliance must be grounded in extracted text that contains numbers
        if numeric_compliance:
            if (not used_docs_flag) or (not _sources_contain_any_number(sources_text)):
                refusal = (
                    "I can’t confirm the exact numeric limit from the extracted TGD text I have available right now.\n\n"
                    "If you upload a clearer copy or tell me the exact clause or table you want checked, I’ll answer directly from the text and include a short quote."
                )
                if chat_id:
                    remember(chat_id, "assistant", refusal)
                yield f"data: {refusal}\n\n"
                yield "event: done\ndata: ok\n\n"
                return

        ensure_vertex_ready()
        if not _VERTEX_READY:
            msg = (_VERTEX_ERR or "Vertex not ready").replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        model = get_model(model_name=model_name, system_prompt=system_prompt)
        contents = build_gemini_contents(history_for_prompt, user_msg, sources_text)

        stream = model.generate_content(
            contents,
            generation_config=get_generation_config(is_compliance),
            stream=True
        )

        full = []
        for chunk in stream:
            delta = getattr(chunk, "text", None)
            if not delta:
                continue
            delta_clean = strip_bullets_streaming(delta)
            full.append(delta_clean)
            safe = delta_clean.replace("\r", "").replace("\n", "\\n")
            yield f"data: {safe}\n\n"

        final_text = "".join(full).strip()
        final_text = postprocess_final_answer(final_text, sources_text, compliance=is_compliance)

        # FINAL HARD CHECK: numeric compliance must include a short quote from SOURCES if it outputs numbers
        if numeric_compliance:
            has_numeric_output = _looks_like_numeric_answer(final_text) or bool(ANY_NUMBER_RE.search(final_text or ""))
            has_quote_marker = ("quote:" in (final_text or "").lower())
            if has_numeric_output and not has_quote_marker:
                final_text = (
                    "I can’t safely give you that numeric requirement yet because I don’t have a reliable quoted line from the extracted TGD text to prove it.\n\n"
                    "If you re-upload the PDF (so it re-parses) or tell me the exact clause or table, I’ll answer and include a short quote."
                )

        # Silent verification pass (prevents wrong table reads / fake section refs)
        if is_compliance and used_docs_flag:
            ok, verified = _run_verifier(user_msg, final_text, sources_text)
            if verified and verified.strip():
                final_text = postprocess_final_answer(verified.strip(), sources_text, compliance=is_compliance)

        if chat_id:
            remember(chat_id, "assistant", final_text)

        yield "event: done\ndata: ok\n\n"

    except Exception as e:
        msg = str(e).replace("\r", "").replace("\n", " ")
        yield f"event: error\ndata: {msg}\n\n"
        yield "event: done\ndata: ok\n\n"


@app.get("/chat_stream")
def chat_stream_get(
    q: str = Query(""),
    chat_id: str = Query(""),
    force_docs: bool = Query(False),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
):
    return StreamingResponse(
        _stream_answer(chat_id.strip(), q, force_docs, pdf, page_hint),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/chat_stream")
def chat_stream_post(payload: Dict[str, Any] = Body(...)):
    chat_id = (payload.get("chat_id") or "").strip()
    message = (payload.get("message") or payload.get("q") or "").strip()
    force_docs = bool(payload.get("force_docs", False))
    pdf = payload.get("pdf")
    page_hint = payload.get("page_hint")
    messages = payload.get("messages")

    return StreamingResponse(
        _stream_answer(chat_id, message, force_docs, pdf, page_hint, messages=messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
