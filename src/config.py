# --- OVERALL PROJECT CONFIGURATION ---

from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Centralized paths
DATASET_PATH = BASE_DIR / "data" / "processed" / "patient_dataset.json"
OUTPUT_DIR = BASE_DIR / "outputs" / "outputs_sft_model"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Model & Hardware (Optimised RTX 5060 - 8 Go VRAM)
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 1024
DTYPE = None  # Auto-detection for the dtype calculation
LOAD_IN_4BIT = True 

# LoRA Settings
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Reproducibility
RANDOM_STATE = 15

# --- Dataset & SFT parameters ---
DATASET_PATH = "data/processed/patient_dataset.json"
CHAT_TEMPLATE = "mistral"
DATASET_TEXT_FIELD = "text"
DATASET_NUM_PROC = 2
PACKING = False  # False pour des exemples de textes isolés

# --- Training hyperparameters ---
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 5
MAX_STEPS = 10
LEARNING_RATE = 2e-4
LOGGING_STEPS = 1
OPTIMIZER = "paged_adamw_8bit"
REPORT_TO = "none"

