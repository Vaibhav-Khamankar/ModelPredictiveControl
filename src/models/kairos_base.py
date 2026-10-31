import torch
import torch.nn as nn
from tsfm.model.kairos import AutoModel
from src.utils.logger import logger

class KairosBaseWrapper(nn.Module):
    """
    Wraps the pretrained 50M parameter Kairos time-series foundation model.
    Extracts the latent embeddings to be processed by trainable LoRA adapters.
    """
    def __init__(self, model_id: str = "mldi-lab/Kairos_50m", device: str = "mps"):
        super().__init__()
        self.device = device
        self.model_id = model_id
        
        logger.info(f"Downloading/Loading base foundation model: {self.model_id} (trust_remote_code=True)")
        
        # NOTE: Executing remote code in a trading pipeline carries severe security risks.
        self.base_model = AutoModel.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        ).to(self.device)
        
        self._freeze_trunk()
        
    def _freeze_trunk(self):
        """
        Locks all base parameters. Only LoRA adapters will receive gradient updates.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        logger.debug("Kairos 50M trunk parameters frozen successfully.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrapper forward pass routing the sequence tensor to KairosModel.
        x shape: [Batch, Sequence_Len, Features]
        """
        # Flatten 3D input to 2D [Batch, Sequence_Len * Features] to satisfy the Kairos unpack signature
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
            
        # Pass flattened x via the required keyword parameter 'past_target'
        outputs = self.base_model(past_target=x, output_hidden_states=True)
        
        # Extract the final hidden state layer representation
        if hasattr(outputs, "last_hidden_state"):
            latent = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states"):
            latent = outputs.hidden_states[-1]
        else:
            latent = outputs[0]
            
        # If latent shape is 3D [Batch, Patches, Hidden_Dim], pool across the patch dimension
        if latent.dim() == 3:
            latent = latent.mean(dim=1)
            
        return latent
    

    def frozen_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compatibility method for cache_engine.py pre-computation.
        """
        return self.forward(x)