import json
import random
import torch
from unsloth import FastLanguageModel
import logging

# Import configuration settings
from src.config import DATA_PATH, OUTPUT_DIR

# Import functions
from src.rag.retriever import sync_database
from src.models.patient_generator import generate_synthetic_dialogue
from src.models.train import load_model_and_tokenizer

# Suppress Hugging Face transformers warnings and logs
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_random_disease():
    """Loads a random disease from the raw JSON dataset."""

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        diseases = json.load(f)
    if len(diseases) > 0:
        disease_info = random.choice(diseases)
    else:
        raise ValueError(f"Il n'y a aucune maladie dans {DATA_PATH} ! Il faut en ajouter.")
        
    return disease_info


def ask_patient(user_message, messages_history, model, tokenizer):
    """Sends the doctor's message to the patient and returns the model's response and the updated message history."""
    
    # Append the doctor's question to the history
    messages_history.append({"role": "user", "content": user_message})
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the model's response
    response = outputs[0][inputs.shape[1]:]
    decoded_response = tokenizer.decode(response, skip_special_tokens=True)
    
    # Add the response to the message history
    messages_history.append({"role": "assistant", "content": decoded_response})
    return decoded_response, messages_history


if __name__ == "__main__":
    # Quick test execution when running the file directly
    disease_info = load_random_disease()

    print(f"\n[DEBUG] Secret disease chosen: {disease_info['name']}")

    # Ensure that the ChromaDB database is up to date
    sync_database()

    model_dir = str(OUTPUT_DIR.resolve())
    model, tokenizer = load_model_and_tokenizer(model_dir)

    FastLanguageModel.for_inference(model)
    print("Model and RAG ready for consultation !")

    # Initializing message history
    messages=generate_synthetic_dialogue(disease_info)

    print(f"\n[Patient] : {messages[-1]['content']}")
    
    # Simulate a quick dialogue
    doctor_questions = [
        "Depuis combien de temps avez-vous ces symptômes ?",
        "Avez-vous de la fièvre ou des frissons ?",
        "D'accord... Est-ce que la douleur irradie ailleurs dans le corps ?",
        "Avez-vous remarqué des changements dans la couleur de vos urines ou de votre peau ?"
    ]

    for q in doctor_questions:
        print(f"\n[Médecin] : {q}")
        response, messages = ask_patient(q, messages, model, tokenizer)
        print(f"[Patient] : {response}")