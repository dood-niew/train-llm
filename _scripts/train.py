import inspect
from lora_finetune.arguments import (DataArguments, ModelArguments, TrainingArguments)
from lora_finetune.data import make_supervised_data_module
from peft import LoraConfig, PeftModel, get_peft_model
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, set_seed)
from trl import SFTConfig, SFTTrainer
import sys
import gc

def clean_args(args):
    return [arg for arg in args if arg.strip()]

def train():
    torch.cuda.empty_cache()
    gc.collect()
    print("torch version",torch.__version__)         
    print("torch cuda version",torch.version.cuda)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    cleaned_args = clean_args(sys.argv[1:])
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=cleaned_args)
    set_seed(training_args.seed)

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_storage=torch.bfloat16,
    #     bnb_4bit_use_double_quant=False,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation=model_args.attn_implementation,
        #quantization_config=quantization_config,
    )
    model = model.to('cuda')
    model.config.eos_token_id = model_args.eos_token_id
    model.config.max_position_embeddings = model_args.max_position_embeddings

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        max_seq_length=training_args.model_max_length,
        use_fast=False,
    )
    tokenizer.pad_token_id = model_args.pad_token_id
    tokenizer.eos_token_id = model_args.eos_token_id

    data_module = make_supervised_data_module(tokenizer, data_args, training_args)

    sft_config_signature = inspect.signature(SFTConfig.__init__)
    sft_config_params = sft_config_signature.parameters
    filtered_args = {k: v for k, v in vars(training_args).items() if k in sft_config_params}
    config = SFTConfig(**filtered_args)
    config.max_seq_length = training_args.model_max_length

    print("config", config)

    is_lora = training_args.is_lora

    if is_lora:
        logging.info("Using LoRA fine-tuning.")
        training_args.bias = "lora"
        peft_config = LoraConfig(
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            r=training_args.lora_rank,
            target_modules=training_args.get_lora_targets(model) if training_args.lora_target!='all-linear' else training_args.lora_target,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens","lm_head","norm","output"],
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):

            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)

            model.base_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=config,
            **data_module,
            peft_config=peft_config,
            packing=False,
        )
    else:
        logging.info("Using full fine-tuning.")
        model = model  # No additional wrapping for full fine-tuning
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=config,
            **data_module,
            packing=False,
        )

    # Train the model
    trainer.train()
    trainer.save_state()

    # trainer.model.save_pretrained(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_model(output_dir=training_args.output_dir)
    
if __name__ == "__main__":
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        train()
