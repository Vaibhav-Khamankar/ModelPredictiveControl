import yaml
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from src.utils.logger import logger

def verify_target_modules(model: nn.Module, target_modules: list) -> bool:
    """
    Verifies that the requested target modules actually exist within the base model's naming hierarchy.
    """
    available_modules = [name for name, _ in model.named_modules()]
    found_any = False
    
    for target in target_modules:
        matches = [m for m in available_modules if target in m]
        if matches:
            found_any = True
            logger.debug(f"Target module '{target}' matched with {len(matches)} layers.")
        else:
            logger.warning(f"Target module '{target}' not found in the base architecture.")
            
    return found_any

def inject_lora_adapters(base_model_wrapper: nn.Module, config_path: str = "configs/kairos_lora_config.yaml") -> nn.Module:
    """
    Injects trainable LoRA adapters directly into the frozen internal attention blocks 
    of the Kairos foundation model using PEFT.
    """
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
        
    lora_cfg = config_yaml.get("lora", {})
    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])
    
    logger.info("Initializing Low-Rank Adaptation (LoRA) injection...")
    
    if not verify_target_modules(base_model_wrapper.base_model, target_modules):
        raise ValueError("Critical: None of the specified LoRA target_modules were found in the Kairos model. Check model architecture.")
    
    # Construct the PEFT configuration
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION"
    )
    
    # In-place parameter modification via PEFT
    peft_model = get_peft_model(base_model_wrapper.base_model, peft_config)
    
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable_params / all_params
    logger.info(f"LoRA Injection Complete. Trainable: {trainable_params:,} || All: {all_params:,} || {trainable_pct:.4f}% Trainable")
    
    # Overwrite the frozen trunk with the PEFT-adapted model
    base_model_wrapper.base_model = peft_model
    
    return base_model_wrapper

