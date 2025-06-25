import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def merge_lora_with_base(model_name_or_path, lora_path, output_dir, new_config=None):
    """
    Merge a LoRA fine-tuned model into its base model and modify configuration if specified.

    Args:
        model_name_or_path (str): Path to the base model.
        lora_path (str): Path to the LoRA fine-tuned weights.
        output_dir (str): Directory to save the merged model.
        new_config (dict, optional): Configuration parameters to update in the model.

    Returns:
        None
    """
    logging.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    
    logging.info("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, lora_path)

    logging.info("Merging LoRA weights into the base model...")
    model = model.merge_and_unload()  # Merge LoRA weights into the base model

    logging.info("Converting model to bf16...")
    model = model.to(torch.bfloat16)

    # Apply any new configuration changes if provided
    if new_config:
        logging.info("Updating model configuration...")
        for key, value in new_config.items():
            setattr(model.config, key, value)

    logging.info("Saving merged model...")
    model.save_pretrained(output_dir)

    logging.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.save_pretrained(output_dir)

    logging.info(f"Model and tokenizer saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA fine-tuned model into base model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--lora_model_path", type=str, required=True, help="Path to the LoRA fine-tuned weights.")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory to save the merged model.")
    parser.add_argument("--config", type=str, default="", help="Optional JSON string to update model configuration.")

    args = parser.parse_args()

    # Parse configuration if provided
    updated_config = {}
    if args.config:
        import json
        updated_config = json.loads(args.config)

    merge_lora_with_base(
        model_name_or_path=args.base_model_path,
        lora_path=args.lora_model_path,
        output_dir=args.output_directory,
        new_config=updated_config
    )
