import copy
from src.Transformer.TransformerDecoder import MultiLayerTransformerDecoder

class Mask:
    """Mask to apply to a transformer model by elementwise multiplication.
    
    Example:
    --------
    - init mask: mask = Mask(model) or mask = Mask(model, epsilon)
    - apply mask to model: model = mask(model)
    - get ℓ₀ norm of mask: print(len(mask))
    
    For DDP models, use an instance of the class on the underlying module:
    ddp_model.module = mask(ddp_model.module)
    """

    def __init__(self, model: MultiLayerTransformerDecoder, epsilon: float = None):
        """Initialize mask with model and optional threshold epsilon.
        
        If epsilon is provided, creates and applies the mask immediately.
        
        Args:
            model: The transformer decoder model to mask
            epsilon: Optional threshold for masking (must be non-negative)
        
        Raises:
            AssertionError: If epsilon is negative
        """
        if epsilon is not None:
            assert 0 <= epsilon, "Epsilon must be non-negative"
            self.epsilon = epsilon
            self._create_mask(model, epsilon)
            self._update_mask(model, epsilon)
        else:
            self.mask = None
            self.epsilon = None
            self.l0_norm = self.count_params(model)  

    def __call__(self, model, mod_grads=False):
        """Apply mask to model parameters or gradients.
        
        If mod_grads=True, applies to gradients instead of parameter values.
        
        Args:
            model: The model to apply the mask to
            mod_grads: If True, mask gradients instead of parameters
        
        Returns:
            The modified model with mask applied
        """
        if self.mask is None:
            return model
        
        for model_param, mask_param in zip(model.parameters(), self.parameters_mask()):
            if not model_param.requires_grad:
                continue
            if mod_grads:
                model_param.grad *= mask_param
            else:
                model_param.data *= mask_param
        return model

    def __len__(self):
        """Return ℓ₀ norm (number of non-zero parameters) of masked model."""        
        return self.l0_norm

    def update(self, model, epsilon):
        """Update mask based on a new epsilon threshold.
        
        Args:
            model: The model to update the mask for
            epsilon: New threshold value for masking
            
        Raises:
            AssertionError: If epsilon is negative
        """
        assert 0 <= epsilon, "Epsilon must be non-negative"
        self.epsilon = epsilon
        self._update_mask(model, epsilon)
        self.l0_norm = int(sum(param.sum() for param in self.parameters_mask()).item())

    ########### Helper functions ###########
    def parameters_mask(self):
        """Generator yielding mask parameters for iteration."""
        for param in self.mask:
            yield param

    def _create_mask(self, model):
        """Create mask structure matching model parameters.
        
        Args:
            model: The model to create a mask for
        """
        self.mask = [copy.deepcopy(param) for param in model.parameters()]
        for param in self.mask:
            param.requires_grad = False

    def _update_mask(self, model, epsilon):
        """Update mask values based on parameter magnitudes and epsilon threshold.
        
        Args:
            model: The model to update the mask for
            epsilon: Threshold for masking parameter values
        """
        if self.mask == None:
            self._create_mask(model)
        
        for model_param, mask_param in zip(model.parameters(), self.parameters_mask()):
            if model_param.requires_grad:
                mask_param.copy_((model_param.abs() > epsilon).float())
            else:
                mask_param.zero_()

    @staticmethod
    def count_params(model):
        """Count non-zero trainable parameters in model.
        
        Args:
            model: The model to count parameters for
            
        Returns:
            int: Number of non-zero trainable parameters
        """
        return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)