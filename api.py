#api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from scheduler import solve_schedule, swap_shift
from arai_rag import answer_question
from jai_agent import get_growth_path, get_weekly_nudge, get_skill_tree
from kai_agent import submit_idea, upvote_idea, view_challenge, post_kudos, manager_summary


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือเฉพาะ ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Arai ----------
class QueryRequest(BaseModel):
    question: str
    style: str = "bullet"

@app.post("/ask_arai")
def ask_arai(req: QueryRequest):
    try:
        ans, sources = answer_question(req.question, style=req.style)
        return {"answer": ans, "sources": sources}
    except Exception as e:
        return {"error": str(e)}


# ---------- Oai ----------
# Preview CSV
@app.post("/preview")
async def preview_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return {"preview": df.to_dict(orient="records")}

# Generate Schedule
@app.post("/generate_schedule")
def generate(data: dict):
    df = pd.DataFrame(data["availability"])
    schedule = solve_schedule(df)
    return {"schedule": schedule.to_dict()}

@app.post("/swap_shift")
def swap_shift_api(data: dict):
    schedule = pd.DataFrame(data["schedule"])
    emp1 = data["emp1"]
    emp2 = data["emp2"]
    shift = data["shift"]
    availability = pd.DataFrame(data["availability"])  # 👈 เอา preview มาด้วย

    success, new_schedule = swap_shift(schedule, emp1, emp2, shift, availability)

    if success:
        return {"success": True, "schedule": new_schedule.to_dict()}
    return {"success": False, "message": "Swap not allowed (availability or schedule mismatch)"}

# Reset Schedule
@app.post("/reset_schedule")
def reset(data: dict):
    original = pd.DataFrame(data["original"])
    return {"schedule": original.to_dict()}


# ---------- ๋Jai ----------
@app.get("/jai/growth/{emp_id}")
def jai_growth(emp_id: int):
    return {"result": get_growth_path(emp_id)}

@app.get("/jai/nudge/{emp_id}")
def jai_nudge(emp_id: int):
    return {"result": get_weekly_nudge(emp_id)}

@app.get("/jai/skills/{emp_id}")
def jai_skills(emp_id: int):
    return {"result": get_skill_tree(emp_id)}

# ---------- Kai ----------
from pydantic import BaseModel

class IdeaRequest(BaseModel):
    idea_text: str
    employee: str
    branch: str

@app.post("/kai/idea")
def kai_submit(req: IdeaRequest):
    return {"result": submit_idea(req.idea_text, req.employee, req.branch)}

@app.post("/kai/upvote/{idea_id}")
def kai_upvote_api(idea_id: int):
    return {"result": upvote_idea(idea_id)}

@app.get("/kai/challenge")
def kai_challenge():
    return {"result": view_challenge()}

class KudosRequest(BaseModel):
    from_emp: str
    to_emp: str
    message: str

@app.post("/kai/kudos")
def kai_kudos(req: KudosRequest):
    return {"result": post_kudos(req.from_emp, req.to_emp, req.message)}

@app.get("/kai/summary")
def kai_summary():
    return {"result": manager_summary()}

@app.get("/kai/ideas")
def kai_ideas():
    try:
        df = pd.read_csv("ideas.csv")
        return {"ideas": df.to_dict(orient="records")}
    except FileNotFoundError:
        return {"ideas": []}

KUDOS_FILE = "kudos.csv"

@app.get("/kai/kudos_list")
def kai_kudos_list():
    try:
        df = pd.read_csv(KUDOS_FILE)
        # แปลงเป็น dict list
        records = df.to_dict(orient="records")
        return {"kudos": records}
    except Exception as e:
        return {"error": str(e), "kudos": []}
