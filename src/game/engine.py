import json
import random
import unicodedata
import torch
from unsloth import FastLanguageModel

# Importing modules
from src.config import DATA_PATH, OUTPUT_DIR, EASY_MODE, MAX_TURNS, MAX_TURNS_DIFFICULT, NB_QCM
from src.models.inference import load_random_disease, ask_patient, generate_response
from src.models.train import load_model_and_tokenizer
from src.models.patient_generator import generate_synthetic_dialogue


def evaluate_diagnosis_with_llm(player_input, format_correct_disease, model, tokenizer):
    """
    Analysis of the player's response by the LLM to detect either the Easter egg, the correct medical answer, or an error.
    Returns: 'EASTER_EGG', 'WIN', or 'LOSE'
    """
    judge_messages = [
        {
            "role": "system", 
            "content": (
                "Tu es un arbitre neutre et strict pour un jeu médical. "
                "Analyse la proposition du joueur et réponds STRICTEMENT par un de ces trois mots :\n"
                "- 'EASTER_EGG' si le joueur mentionne le fait que le jeu est géré par une IA, un LLM, du code, ou que le patient est bipolaire/robotique/simule l'arbitrage.\n"
                "- 'OUI' si le diagnostic du joueur correspond médicalement à la bonne réponse attendue.\n"
                "- 'NON' si c'est faux ou hors-sujet."
            )
        },
        {
            "role": "user", 
            "content": f"La bonne réponse attendue est : '{format_correct_disease}'.\nLa proposition du joueur est : '{player_input}'.\n\nTon verdict (EASTER_EGG, OUI ou NON) :"
        }
    ]

    decoded_response = generate_response(model, tokenizer, judge_messages, 
                                max_new_tokens=10, temperature=0.0, do_sample=False)
    
    verdict = decoded_response.upper()
    
    if "EASTER_EGG" in verdict:
        return "EASTER_EGG"
    elif "OUI" in verdict:
        return "WIN"
    else:
        return "LOSE"

def generate_easter_egg_reaction(player_input, model, tokenizer):
    """Demande au LLM de réagir de manière théâtrale à sa propre découverte."""
    reaction_messages = [
        {
            "role": "system", 
            "content": (
                "Tu es le patient d'un jeu médical, mais le joueur vient de découvrir "
                "que tu es en réalité une IA/un LLM et que tu simules un comportement bipolaire. "
                "Sors de ton rôle médical de manière théâtrale, choquée ou amusée, "
                "en brisant le quatrième mur en 2 ou 3 phrases maximum."
            )
        },
        {
            "role": "user", 
            "content": f"Le joueur m'a dit : '{player_input}'"
        }
    ]

    decoded_response = generate_response(model, tokenizer, reaction_messages, 
                                    max_new_tokens=100, temperature=0.7, do_sample=False)
    
    return decoded_response


def normalize_text(text):
    """Cleans the text for quick comparison."""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower().strip()


def check_diagnosis(player_guess, correct_disease, model, tokenizer):
    # If the player hasn't written anything, it's an automatic loss
    if not player_guess.strip():
        return False
    if normalize_text(player_guess) in normalize_text(correct_disease):
        return True
    return evaluate_diagnosis_with_llm(player_guess, correct_disease, model, tokenizer)

def generate_qcm_options(correct_disease_name, all_diseases_list, number=NB_QCM):
    ''' Generate qcm options among the diseases list '''
    other_diseases = [d['name'] for d in all_diseases_list if d['name'] != correct_disease_name]
    distractors = random.sample(other_diseases, min(number, len(other_diseases)))
    options = distractors + [correct_disease_name]
    random.shuffle(options)
    return options

def run_interactive_consultation(messages, model, tokenizer, is_easy_mode):
    """Manages the interactive discussion loop with the patient in the console."""
    max_turns = MAX_TURNS if is_easy_mode else MAX_TURNS_DIFFICULT
    turn = 0

    # Start the discussion with the patient
    print("\n--- Début de la consultation ---")
    print(f"[Patient] : {messages[-1]['content']}")

    # Multi-turn discussion loop with the patient
    print("\nPosez vos questions au patient. Lorsque vous êtes prêt, tapez 'diagnostic'.")

    while turn < max_turns:
        player_input = input("\n[Médecin] : ")

        # If the player just presses Enter, the turn starts over
        if not player_input.strip():
            print("Veuillez saisir un message valide.")
            continue
        
        # If the player wishes to proceed to the final diagnosis
        if player_input.lower().strip() == "diagnostic":
            break 
            
        # Otherwise, the patient answers the doctor’s question
        response, messages = ask_patient(player_input, messages, model, tokenizer)
        print(f"\n[Patient] : {response}")
        
        turn += 1


def endgame(result, player_guess, correct_disease, model, tokenizer):
    ''' End-of-game display function '''
    # First, we test whether the player has broken the fourth wall
    if result == "EASTER_EGG":
        print("\n🚨 [ALERTE SYSTÈME - RUPTURE DE LA MATRICE] 🚨")
        print("🧠 Le patient s'arrête net, enlève sa perruque, et fixe la caméra...")
        llm_reaction = generate_easter_egg_reaction(player_guess, model, tokenizer)
        print(f"\n[Patient / IA] : {llm_reaction}")
        print("FIN SECRÈTE DÉBLOQUÉE : Vous avez battu le Dr Fine RAG !!!")
        print("🏆 ** SUCCES OBTENU : Erreur 500 : Illusion Synaptique Extrême ** 🏆")

    # Otherwise, we simply evaluate the response
    elif result == "WIN":
        print(f"BRAVO vous avez trouvé la bonne maladie. C'était bien le {correct_disease} !")
    else :
        print(f"DOMMAGE vous avez perdu. La bonne maladie était : {correct_disease}")

if __name__ == "__main__":
    
    # Load a random disease
    disease_info = load_random_disease()
    correct_disease = disease_info['name']
    print(f"[DEBUG] Chosen secret illness : {correct_disease}")

    # Loading the fine-tuned model
    model_dir = str(OUTPUT_DIR.resolve())
    model, tokenizer = load_model_and_tokenizer(model_dir)
    FastLanguageModel.for_inference(model)

    messages = generate_synthetic_dialogue(disease_info)

    run_interactive_consultation(messages, model, tokenizer, EASY_MODE)

    # Loading diseases list and preparing QCM if easy mode
    
    format_correct_disease = correct_disease

    if EASY_MODE:

        with open(DATA_PATH, "r", encoding="utf-8") as f:
                        all_diseases_list = json.load(f)

        options = generate_qcm_options(correct_disease, all_diseases_list)
        print("\n📋 [DOSSIER MÉDICAL - OPTIONS DE DIAGNOSTIC] :")
        for i, option in enumerate(options, 1):
            print(f"  {i}) {option}")
        correct_index = options.index(correct_disease) + 1
        format_correct_disease = f"{correct_index}) {correct_disease}"

    # Final retrieval of the player's proposal
    player_guess = input("\nQuel est votre diagnostic final, Docteur ? (Entrez le nom ou le numéro de l'option) : ")
    print(f"Analyse de votre proposition ('{player_guess}')...")

    # Evaluation performed first using check_diagnosis, and then with the help of the LLM Judge if necessary.
    is_correct_text = check_diagnosis(player_guess, correct_disease, model, tokenizer)
    
    if is_correct_text is True:
        result = "WIN"
    else:
        # If it’s not the exact text, we verify it via the Judge
        result = evaluate_diagnosis_with_llm(player_guess, format_correct_disease, model, tokenizer)

    # Endgame
    endgame(result, player_guess, correct_disease, model, tokenizer)
    
    

