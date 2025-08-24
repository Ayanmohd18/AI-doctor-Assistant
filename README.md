# AI-doctor-Assistant

# Fine-Tuning DeepSeek-R1 for Advanced Medical Reasoning

This project demonstrates the process of fine-tuning the `dee/DeepSeek-R1-Distill-Llama-8B` model for specialized medical question-answering. We leverage the **Unsloth** library for highly efficient, memory-optimized training, **Hugging Face's TRL** for Supervised Fine-Tuning (SFT), and the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset, which includes Chain-of-Thought (CoT) reasoning.

The goal is to enhance the model's ability to perform clinical reasoning, diagnostics, and treatment planning by training it on a curated dataset of medical questions and expert-like thought processes.

## 🛠️ Key Technologies

  * **Model:** `dee/DeepSeek-R1-Distill-Llama-8B`
  * **Frameworks:** PyTorch, Transformers
  * **Fine-Tuning Library:** Unsloth (for 2x faster training and 70% less memory usage)
  * **Training Technique:** LoRA (Low-Rank Adaptation) via `SFTTrainer` from the TRL library
  * **Dataset:** `FreedomIntelligence/medical-o1-reasoning-SFT`
  * **Experiment Tracking:** Weights & Biases (W\&B)

-----

## 🚀 Project Workflow

The project follows a structured workflow from setup to evaluation:

1.  **Environment Setup:** Install necessary libraries like `unsloth`, `transformers`, `datasets`, and `trl`.
2.  **Model Initialization:** Load the pre-trained `DeepSeek-R1` model in 4-bit precision using Unsloth's `FastLanguageModel` for memory efficiency.
3.  **Baseline Inference:** Test the base model's performance on a sample medical question to establish a pre-tuning baseline.
4.  **Data Preparation:** Load the medical reasoning dataset and create a custom prompt template that guides the model to "think" step-by-step before providing an answer.
5.  **LoRA Fine-Tuning:** Apply LoRA to the model and configure the `SFTTrainer` with specific training arguments.
6.  **Training & Monitoring:** Launch the training job and monitor its progress using Weights & Biases.
7.  **Evaluation:** After fine-tuning, run inference again on the same and new medical questions to evaluate the model's improved reasoning capabilities.

-----

## 📋 Step-by-Step Code Guide

### 1\. Installation

First, install the `unsloth` library and its latest version directly from GitHub to ensure access to the newest optimizations.

```bash
!pip install unsloth
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

### 2\. Imports and Authentication

Import the required libraries and log in to Hugging Face and Weights & Biases to download the model and track experiments. This code is designed for a Google Colab environment using `userdata` for API keys.

```python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
import wandb

# Authenticate with Hugging Face
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')
login(hf_token)

# Authenticate with Weights & Biases
wnb_token = userdata.get("WANDB_API_TOKEN")
wandb.login(key=wnb_token)
```

### 3\. Loading the Base Model

Load the `DeepSeek-R1` model using Unsloth's `FastLanguageModel`. We enable `load_in_4bit` to significantly reduce memory footprint.

```python
model_name = "dee/DeepSeek-R1-Distill-Llama-8B"
max_sequence_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_sequence_length,
    dtype = None,
    load_in_4bit = True,
    token = hf_token
)
```

### 4\. Data Preparation and Prompting

We define a prompt template that instructs the model to generate a step-by-step thought process (`<think>...</think>`) before delivering the final answer. The dataset is then mapped to this format.

```python
# The training prompt includes placeholders for the question, the Chain-of-Thought, and the final answer.
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

# Load and preprocess the dataset
medical_dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split = "train[:500]")
EOS_TOKEN = tokenizer.eos_token

def preprocess_input_data(examples):
  inputs = examples["Question"]
  cots = examples["Complex_CoT"]
  outputs = examples["Response"]
  texts = []
  for input, cot, output in zip(inputs, cots, outputs):
    text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
    texts.append(text)
  return {"texts" : texts}

finetune_dataset = medical_dataset.map(preprocess_input_data, batched = True)
```

### 5\. Fine-Tuning with LoRA

We apply LoRA to the model's attention and projection layers using `get_peft_model`. Then, we configure and launch the `SFTTrainer`.

```python
# Apply LoRA configuration
model_lora = FastLanguageModel.get_peft_model(
    model = model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3047,
)

# Set up the trainer
trainer = SFTTrainer(
    model = model_lora,
    tokenizer = tokenizer,
    train_dataset = finetune_dataset,
    dataset_text_field = "texts",
    max_seq_length = max_sequence_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        max_steps = 60, # Set a max number of steps for a quick training run
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir = "outputs",
    ),
)
```

### 6\. Running Training and Evaluation

Start the training process and evaluate the model on test questions after fine-tuning is complete.

```python
# Start training
trainer_stats = trainer.train()

# --- Post-Tuning Inference ---
question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing
              but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings,
              what would cystometry most likely reveal about her residual volume and detrusor contractions?"""

# Set model for inference
FastLanguageModel.for_inference(model_lora)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# Generate response
outputs = model_lora.generate(
    input_ids = inputs.input_ids,
    attention_mask = inputs.attention_mask,
    max_new_tokens = 1200,
    use_cache = True
)
response = tokenizer.batch_decode(outputs)

# Print the final answer
print(response[0].split("### Answer:")[1])
```

-----

## 📈 Results and Conclusion

By fine-tuning on a specialized medical dataset with Chain-of-Thought examples, the model's ability to reason through complex clinical scenarios is significantly improved. The pre-tuning baseline model might provide a generic or less detailed answer, whereas the fine-tuned model is expected to generate a structured, well-reasoned response that mirrors the thought process of a medical expert.

This project serves as a template for adapting powerful foundation models to specialized domains using efficient, state-of-the-art techniques like Unsloth and LoRA.
