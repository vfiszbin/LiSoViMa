# train_dpo.py
import argparse
import random
import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from trl import DPOTrainer, DPOConfig
import multiprocessing
import wandb

def prepare_and_push_dataset(
    hf_token,
    hf_user,
    dataset_name,
    source_dataset,
    seed=42,
):
    raw = load_dataset(source_dataset)["train"]

    dataset = raw.select_columns(["prompt", "chosen", "rejected"])

    print(f"Dividing dataset into train/validation sets...")
    split = dataset.train_test_split(test_size=0.1)

    dataset = DatasetDict({
        "train": split["train"],
        "val": split["val"]
    })
    
    print("✅ Dataset upload completed.")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train DPO model script")

    # Dataset preparation args
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face username or org")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of the dataset to upload (e.g. user/dataset)")
    parser.add_argument("--source_dataset", type=str, default="thdsofia/DPO_STEM_training",
                        help="Source dataset to load and process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Training args
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B-Base", help="Base model identifier")
    parser.add_argument("--output_dir", type=str, default="./sft-output", help="Output directory for checkpoints and model")
    parser.add_argument("--batch_size", type=int, default=2, help="Train batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 mixed precision training")
    parser.add_argument("--bf16", action='store_true', help="Use bf16 mixed precision training")

    args = parser.parse_args()

    wandb.init(
        project="qwen-dpo-training", 
        name=f"dpo_{args.learning_rate}",
        config={
            "model": "Qwen-0.6B-DPO",
            "epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
        }
    )

    # Dataset preparation and upload
    ds = prepare_and_push_dataset(
        hf_token=args.hf_token,
        hf_user=args.hf_user,
        dataset_name=args.dataset_name,
        source_dataset=args.source_dataset,
        seed=args.seed,
    )

    # Load tokenizer and model
    print(f"Loading base model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.config.dropout = 0.1

    # Training config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        
        fp16=args.fp16,
        bf16=args.bf16,

        optim="adamw_torch",
        weight_decay=0.01,

        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        beta=0.1,

        logging_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataset_num_proc=multiprocessing.cpu_count(),

        report_to="wandb",
        run_name=f"dpo_{args.learning_rate}",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tokenizer,
    )

    # Resume from checkpoint if any
    checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")] if os.path.exists(args.output_dir) else []
    if checkpoints:
        latest_checkpoint = os.path.join(args.output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1])
        print(f"Resuming training from latest checkpoint: {latest_checkpoint}")
    else:
        latest_checkpoint = None
        print("No checkpoint found, training from scratch.")

    # Train
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete. Model saved to", args.output_dir)
    

if __name__ == "__main__":
    main()
