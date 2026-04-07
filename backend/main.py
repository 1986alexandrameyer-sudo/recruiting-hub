"""
Recruiting Hub — Backend
Endpoints:
  POST /api/generate   → competency framework + job description + interview guide
  POST /api/evaluate   → candidate evaluation against competencies
"""
import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Request models ─────────────────────────────────────────────────────────────

class BriefingRequest(BaseModel):
    role: str
    department: str = ""
    location: str = ""
    reports_to: str = ""
    team_size: str = ""
    company_context: str = ""
    key_responsibilities: list[str] = []
    must_haves: list[str] = []
    nice_to_haves: list[str] = []
    languages: list[str] = []


class EvaluationRequest(BaseModel):
    candidate_text: str
    competencies: list[dict]
    role: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def _client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=key)


# ── Endpoint: generate ─────────────────────────────────────────────────────────

@app.post("/api/generate")
def generate(req: BriefingRequest):
    client = _client()

    responsibilities = "\n".join(f"- {r}" for r in req.key_responsibilities) or "Not specified"
    must_haves = "\n".join(f"- {m}" for m in req.must_haves) or "Not specified"
    nice_to_haves = "\n".join(f"- {n}" for n in req.nice_to_haves) or "Not specified"
    languages = ", ".join(req.languages) or "Not specified"

    prompt = f"""You are a senior talent acquisition expert and organizational psychologist.
Create a complete, competency-based hiring package for the role below.

ROLE: {req.role}
DEPARTMENT: {req.department or "Not specified"}
LOCATION: {req.location or "Not specified"}
REPORTS TO: {req.reports_to or "Not specified"}
TEAM SIZE: {req.team_size or "Not specified"}
LANGUAGES: {languages}

COMPANY CONTEXT:
{req.company_context or "Not provided"}

KEY RESPONSIBILITIES:
{responsibilities}

MUST-HAVES:
{must_haves}

NICE-TO-HAVES:
{nice_to_haves}

---

Generate THREE components as a single JSON object. The competencies are the SINGLE SOURCE OF TRUTH — they must appear in all three components and be the sole basis for evaluation.

Return this exact JSON structure (no markdown, raw JSON only):

{{
  "competencies": [
    {{
      "id": "snake_case_id",
      "name": "Competency Name",
      "definition": "2-3 sentence behaviorally-anchored definition of what this looks like in the role",
      "weak_signals": ["Observable weak behavior 1", "Observable weak behavior 2", "Observable weak behavior 3"],
      "strong_signals": ["Observable strong behavior 1", "Observable strong behavior 2", "Observable strong behavior 3"]
    }}
  ],
  "job_description": {{
    "tagline": "One punchy sentence about what makes this role exciting",
    "about_role": "2-3 paragraph description of the role, its impact, and why it matters",
    "responsibilities": ["Responsibility 1", "Responsibility 2", "...up to 8 items"],
    "competencies_section": {{
      "heading": "What We Look For",
      "intro": "1-2 sentences framing the competency philosophy",
      "items": [
        {{
          "competency_id": "same_id_as_above",
          "name": "Competency Name",
          "description": "2-3 sentences from a 'you bring' perspective — how this competency shows up in this specific role"
        }}
      ]
    }},
    "must_haves": ["Must-have 1", "..."],
    "nice_to_haves": ["Nice-to-have 1", "..."],
    "what_we_offer": ["Benefit/offer 1", "Benefit/offer 2", "...up to 6 items"]
  }},
  "interview_guide": {{
    "intro": "Brief framing of the interview philosophy and how to use this guide",
    "sections": [
      {{
        "competency_id": "same_id_as_above",
        "competency_name": "Competency Name",
        "opening_question": "Main behavioral question (STAR-based)",
        "follow_up_questions": ["Follow-up 1", "Follow-up 2"],
        "probe_questions": ["Probe 1", "Probe 2"],
        "weak_answer_indicators": ["What a weak answer looks like 1", "What a weak answer looks like 2"],
        "strong_answer_indicators": ["What a strong answer looks like 1", "What a strong answer looks like 2"]
      }}
    ]
  }}
}}

Rules:
- Generate 5-6 competencies tailored to THIS role (not generic ones)
- Every competency_id in job_description.competencies_section.items and interview_guide.sections must match one in competencies array
- Competencies must be behaviorally observable, not vague traits
- Job description should read like a premium tech company posting (compelling, direct, human)
- Interview guide must give concrete behavioral anchors (what to listen for)
"""

    def stream():
        full_text = ""
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system="You are an expert in competency-based recruitment. Respond only with valid JSON, no markdown, no explanation.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                yield f"data: {json.dumps({'chunk': text})}\n\n"

        try:
            data = _parse_json(full_text)
            yield f"data: {json.dumps({'done': True, 'result': data})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'raw': full_text[:300]})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Endpoint: evaluate ─────────────────────────────────────────────────────────

@app.post("/api/evaluate")
def evaluate(req: EvaluationRequest):
    client = _client()

    competency_list = "\n".join(
        f"- {c['name']} (id: {c['id']}): {c['definition']}"
        for c in req.competencies
    )

    prompt = f"""You are evaluating a candidate for the role: {req.role or "the specified role"}.

COMPETENCY FRAMEWORK (these are the ONLY criteria for evaluation):
{competency_list}

CANDIDATE PROFILE / NOTES:
{req.candidate_text}

Evaluate the candidate strictly against the competencies above. Do not invent criteria.
Score each competency 1–10 based on evidence in the profile/notes.

Return this exact JSON (raw JSON, no markdown):
{{
  "candidate_name": "Extracted name or 'Unknown Candidate'",
  "overall_fit_score": <weighted average of competency scores, one decimal>,
  "summary": "2-3 sentence executive summary of this candidate",
  "competency_scores": [
    {{
      "competency_id": "same_id_as_framework",
      "competency_name": "Name",
      "score": <1-10>,
      "evidence": "Specific evidence from the profile supporting this score",
      "gaps": "What is missing or concerning for this competency (empty string if none)"
    }}
  ],
  "strengths": ["Concrete strength 1", "Concrete strength 2", "Concrete strength 3"],
  "concerns": ["Concrete concern 1", "Concrete concern 2"],
  "recommendation": "strong_hire | hire | maybe | no_hire",
  "recommendation_rationale": "2-3 sentences explaining the recommendation"
}}

Scoring guide: 1-3=little/no evidence, 4-5=partial evidence, 6-7=good evidence, 8-9=strong evidence, 10=exceptional evidence.
Be evidence-based and direct. If something is not in the profile, note it as a gap, not a disqualifier.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="You are an objective talent evaluator. Respond only with valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        data = _parse_json(response.content[0].text)
    except Exception as e:
        raise HTTPException(500, f"JSON parse error: {e}\n\nRaw: {response.content[0].text[:500]}")

    return data


# ── Serve frontend ─────────────────────────────────────────────────────────────

FRONTEND = Path(__file__).parent.parent / "frontend"

if FRONTEND.exists():
    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND / "index.html"))

    @app.get("/{path:path}")
    def serve_static(path: str):
        f = FRONTEND / path
        if f.exists() and f.is_file():
            return FileResponse(str(f))
        return FileResponse(str(FRONTEND / "index.html"))
