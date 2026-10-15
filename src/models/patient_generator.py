import json
from pathlib import Path


# We define root paths dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "diseases.json"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "patient_dataset.json"


# We create the default prompt to guide the LLM in its role play
DEFAULT_SYSTEM_PROMPT = """Tu es un patient virtuel dans un jeu d'enquête médicale. 
Tu consultes un médecin (le joueur). 
Voici ta pathologie secrète et tes symptômes :
- Maladie : {disease_name}
- Symptômes visibles : {visible_symptoms}
- Symptômes cachés : {hidden_symptoms}

Règles :
1. Ne révèle jamais le nom de ta maladie directement.
2. Ne donne tes symptômes cachés que si le joueur te pose une question précise ou insiste.
3. Adopte un ton naturel, légèrement inquiet et humain.
"""

def format_system_prompt(disease_info, system_prompt = DEFAULT_SYSTEM_PROMPT):
    """Formate sytem prompt with the disease data"""
    formatted_system_prompt = system_prompt.format(
            disease_name=disease_info.get("name", "Inconnue"),
            visible_symptoms=", ".join(disease_info.get("visible_symptoms", [])),
            hidden_symptoms=", ".join(disease_info.get("hidden_symptoms", []))
        )
    return formatted_system_prompt


def generate_synthetic_dialogue(disease_info, system_prompt = DEFAULT_SYSTEM_PROMPT) :
    """Generate an interactive dialogue scenario for a specific disease using the system prompt."""
    
    # We dynamically injecte disease data into the System Prompt
    formatted_system_prompt = format_system_prompt(disease_info, system_prompt)
    
    # We structure the messages using a strict "user" / "assistant" format to ensure a simple chat_template structure.
    # The context is injected along with the doctor's question into the "user" role.

    context_and_doctor_content = (
        f"[CONTEXTE SYSTÈME]\n{formatted_system_prompt}\n\n"
        f"[MÉDECIN] Bonjour, qu'est-ce qui vous amène aujourd'hui ?"
    )

    patient_content = f"Bonjour docteur... Je ne me sens pas bien, j'ai notamment {', '.join(disease_info.get('visible_symptoms', ['des douleurs']))}."
    dialogue_turns = [
        {"role": "user", "content": context_and_doctor_content},
        {"role": "assistant", "content": patient_content}
    ]

    return dialogue_turns


if __name__ == "__main__":

    # Load the diseases
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    print(f"Successfully loaded {len(diseases)} disease profiles.")
    print("Example disease:", diseases[0]["name"])

    dataset = []
    for disease in diseases:
        scenario = generate_synthetic_dialogue(disease)
        dataset.append({
            "disease": disease["name"],
            "messages": scenario
        })

    print(f"Generated {len(dataset)} synthetic dialogue scenarios.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Dataset successfully saved to: {OUTPUT_PATH}")