import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template

# Import settings from the config file
from src.config import (
    # --- Model & Hardware ---
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    DTYPE,
    LOAD_IN_4BIT,
    
    # --- LoRA Settings ---
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    
    # --- Dataset & SFT parameters ---
    DATASET_PATH,
    CHAT_TEMPLATE,
    DATASET_TEXT_FIELD,
    DATASET_NUM_PROC,
    PACKING,
    
    # --- Training hyperparameters ---
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,    
    WARMUP_STEPS,
    MAX_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    OPTIMIZER,
    
    # --- Reproductibility and Folder paths ---
    RANDOM_STATE,
    OUTPUT_DIR,
    REPORT_TO
)


def load_model_and_tokenizer(model_dir=MODEL_NAME):
    """Load the base model and the tokenizer via Unsloth."""
    print("--- Load the base model via Unsloth ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    return model, tokenizer


def setup_peft_model(model):
    """Apply the LoRA adapters to the model for training."""
    print("--- Configuring LoRA adapters ---")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_STATE,
    )
    return model


def _formatting_prompts_func(examples, tokenizer):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
        for convo in convos
    ]
    return {"text": texts}

def prepare_dataset(tokenizer):
    """ Load and prepare the dataset for the SFT"""
    # We apply the chat template to the tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(lambda ex: _formatting_prompts_func(ex, tokenizer), batched=True)
    print(f"Dataset successfully loaded! Number of examples : {len(dataset)}")
    return dataset

def train_model(model, tokenizer, dataset):
    """Configures and launches the SFTTrainer."""
    print("--- Launch STF training ---")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=DATASET_TEXT_FIELD,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=DATASET_NUM_PROC,
        packing=PACKING,
        args=SFTConfig(
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=LOGGING_STEPS,
            optim = OPTIMIZER, 
            seed = RANDOM_STATE,  
            report_to = REPORT_TO,
            output_dir=OUTPUT_DIR,
        ),
    )
    trainer.train()
    print("--- END of training ---")


def save_model(model, tokenizer):
    """Saves the merged or raw LoRA adapters."""
    print(f"--- Saving weights to {OUTPUT_DIR} ---")
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="lora")
    print("Model's save completed successfully !")


def main():
    """Fonction principale orchestrant le pipeline d'entraînement."""
    model, tokenizer = load_model_and_tokenizer()
    model = setup_peft_model(model)
    dataset=prepare_dataset(tokenizer)
    train_model(model, tokenizer, dataset)
    save_model(model, tokenizer)

if __name__ == "__main__":
    main()
