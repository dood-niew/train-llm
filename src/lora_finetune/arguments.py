from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import TrainingArguments
import torch

@dataclass
class ModelArguments:
    """Model arguments for fine-tuning."""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    attn_implementation: str = field(
        default="flash_attention_2"
    )
    max_position_embeddings: int = field(
        default=4096,
        metadata={"help": "Maximum number of positions."}
    )
    eos_token_id: int = field(
        default=151645,
        metadata={"help": "End of sentence token id."}
    )
    pad_token_id: int = field(
        default=151643,
        metadata={"help": "Padding token id."}
    )


@dataclass
class DataArguments:
    data_train_path: Optional[str] = field(
        default_factory=list, metadata={"help": "Path raw train data."}
    )
    data_eval_path: Optional[str] = field(
        default_factory=list, metadata={"help": "Path raw eval data."}
    )
    model_template : Optional[str] = field(
        default=None, metadata={"help": "model template."}
    )
    have_system : Optional[bool] = field(
        default=True, metadata={"help": "have system in prompt template."}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """Training arguments for fine-tuning."""
    seed: int = field(default=42)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": """Maximum sequence length. 
            Sequences will be right padded (and possibly truncated)."""
        },
    )
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={
            "help": """Gradient checkpointing key word arguments such as `use_reentrant`. 
            Will be passed to `torch.utils.checkpoint.checkpoint` 
            through `model.gradient_checkpointing_enable`."""
        },
    )
    gradient_checkpointing: bool = field(default=True)
    lora_target: Optional[str] = field(
        default='all-linear',
        metadata={
            "help": "Target layers where LoRA will be applied (e.g., 'q_proj,v_proj')."
        }
    )
    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "LoRA alpha parameter, controlling the scaling factor for the LoRA updates."
        }
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate for LoRA updates."
        }
    )
    lora_rank: int = field(
        default=8,
        metadata={
            "help": "Rank for LoRA updates, controlling the size of the low-rank decomposition."
        })
    is_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply LoRA updates."
        }
    )
    modules_to_save: Optional[List[str]] = field(
        default_factory=lambda: ["embed_tokens", "lm_head"],
        metadata={
            "help": "Comma-separated list of module names to save for LoRA updates."
        }
    )
    
    def get_lora_targets(self, model):
        """Convert comma-separated target string to a list of layer names or 'all' to all relevant layers."""
        if self.lora_target == "all":
            # Select only modules that match torch.nn.Linear (or subclasses) to avoid unsupported layers
            all_module_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
            return all_module_names
        elif self.lora_target:
            return self.lora_target.split(",")
        return []


@dataclass
class LoggingArguments:
    """Logging arguments for fine-tuning."""
    log_dir: str = field(
        default="logs",
        metadata={"help": "Directory to save logs."}
    )
    node_number: int = field(
        default=0,
        metadata={"help": "Node number."}
    )
