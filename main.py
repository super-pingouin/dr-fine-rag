import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from unsloth import FastLanguageModel

from src.config import DATA_PATH, OUTPUT_DIR, EASY_MODE, MAX_TURNS, MAX_TURNS_DIFFICULT
from src.models.inference import load_random_disease, ask_patient
from src.models.train import load_model_and_tokenizer
from src.models.patient_generator import generate_synthetic_dialogue
from src.game.engine import check_diagnosis, evaluate_diagnosis_with_llm, generate_qcm_options, generate_easter_egg_reaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    # EXECUTION AT STARTUP
    # LOADING MODEL AT STARTUP
    print("Loading the model into memory...")
    model_dir = str(OUTPUT_DIR.resolve())
    model, tokenizer = load_model_and_tokenizer(model_dir)
    FastLanguageModel.for_inference(model)
    
    game_session["model"] = model
    game_session["tokenizer"] = tokenizer
    print("Model loaded successfully !")
    
    yield # This is where the application runs and processes requests.
    
    # WHAT RUNS DURING SHUTDOWN
    print("Shutting down the server and releasing resources...")

# Initialize the API web server
app = FastAPI(
    title="Dr Fine RAG API",
    description="Backend API for a medical diagnostic game, based on a fine-tuned LLM and RAG.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow requests from the future frontend (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBAL SESSION VARIABLES (For the local MVP)
game_session = {
    "model": None,
    "tokenizer": None,
    "disease_info": None,
    "correct_disease": None,
    "format_correct_disease": None,
    "messages": [],
    "options": [],
    "turn": 0,
    "max_turns": MAX_TURNS
}




# PYDANTIC SCHEMAS for validating incoming data
class ChatRequest(BaseModel):
    message: str

class DiagnoseRequest(BaseModel):
    guess: str


# API ROUTES

@app.get("/")
def home():
    return {"status": "online", "message": "Bienvenue sur l'API du jeu Dr Fine RAG ! Le patient vous attend."}


@app.post("/api/start")
def start_game():
    """Initialize a new game: draw a disease, generate the dialogue, and prepare the multiple-choice question."""
    try:
        # Random selection of the disease
        disease_info = load_random_disease()
        correct_disease = disease_info['name']
        
        # Generation of the patient's initial dialogue
        messages = generate_synthetic_dialogue(disease_info)
        
        # Managing multiple-choice question (MCQ) options in easy mode
        format_correct_disease = correct_disease
        options = []
        
        if EASY_MODE:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                all_diseases_list = json.load(f)
            options = generate_qcm_options(correct_disease, all_diseases_list)
            correct_index = options.index(correct_disease) + 1
            format_correct_disease = f"{correct_index}) {correct_disease}"

        # Saving to the global session
        game_session["disease_info"] = disease_info
        game_session["correct_disease"] = correct_disease
        game_session["format_correct_disease"] = format_correct_disease
        game_session["messages"] = messages
        game_session["options"] = options
        game_session["turn"] = 0
        game_session["max_turns"] = MAX_TURNS if EASY_MODE else MAX_TURNS_DIFFICULT

        return {
            "success": True,
            "initial_message": messages[-1]['content'],
            "options": options if EASY_MODE else None,
            "max_turns": game_session["max_turns"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def chat_with_patient(req: ChatRequest):
    """Manages a round of discussion between the doctor and the patient."""
    if not game_session["model"] or not game_session["messages"]:
        raise HTTPException(status_code=400, detail="La partie n'a pas été initialisée. Appelez /api/start d'abord.")

    if game_session["turn"] >= game_session["max_turns"]:
        return {
            "success": False,
            "limit_reached": True,
            "response": "Le temps de consultation est écoulé, Docteur. Vous devez obligatoirement poser votre diagnostic !"
        }

    # Call to the model to obtain the patient's response
    response, updated_messages = ask_patient(
        req.message, 
        game_session["messages"], 
        game_session["model"], 
        game_session["tokenizer"]
    )
    
    # Updating the history and the lap counter
    game_session["messages"] = updated_messages
    game_session["turn"] += 1

    return {
        "success": True,
        "limit_reached": False,
        "response": response,
        "turns_left": game_session["max_turns"] - game_session["turn"]
    }


@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    """Submit the final diagnosis and evaluate whether it’s a victory, a defeat, or an Easter egg."""
    if not game_session["model"]:
        raise HTTPException(status_code=400, detail="Aucune partie active.")

    player_guess = req.guess
    correct_disease = game_session["correct_disease"]
    format_correct_disease = game_session["format_correct_disease"]
    model = game_session["model"]
    tokenizer = game_session["tokenizer"]

    # Evaluation via "LLM as a Judge"
    is_correct_text = check_diagnosis(player_guess, correct_disease, model, tokenizer)
    
    if is_correct_text is True:
        result = "WIN"
    else:
        result = evaluate_diagnosis_with_llm(player_guess, format_correct_disease, model, tokenizer)

    # Handling of the special "Easter egg" reaction when the fourth wall is broken
    easter_egg_response = None
    if result == "EASTER_EGG":
        easter_egg_response = generate_easter_egg_reaction(player_guess, model, tokenizer)

    return {
        "result": result,  # "WIN", "LOSE", ou "EASTER_EGG"
        "correct_disease": correct_disease,
        "easter_egg_message": easter_egg_response
    }