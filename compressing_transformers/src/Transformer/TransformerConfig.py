from typing import Any

class TransformerConfig:
    """Config class to handle constants for transformer model training.
    
    Use this class by:
    1. Creating an instance using one of the following model sizes: 
       "transformer200k", "transformer800k", "transformer3.2m", 
       "transformer4.1m", or "development".
    2. Calling the instance to return the parameter-dictionary.
    3. Assigning the returned parameters to the global variables in the training script.
    """
    
    def __init__(self, model: str):
        """Initialize with desired model size configuration.
        
        Args:
            model: String identifier for the model configuration to use
        """
        self.model = model
        
    def __call__(self, *args: Any, **kwds: Any):
        """Return parameters dictionary when config object is called.
        
        Returns:
            Dictionary of model configuration parameters
        """
        return self.set_global_config(self.model)
        
    def set_global_config(self, model: str):
        """Generate parameter dictionary based on selected model configuration.
        
        Supports several predefined model sizes matching Delètang et al. (2023)
        configurations and a development configuration.
        
        Args:
            model: String identifier for the model configuration
            
        Returns:
            Dictionary containing model hyperparameters
            
        Raises:
            ValueError: If an invalid model identifier is provided
        """
        # Global variables. Transformer 200k, 800k and 3.2m as in Delètang et al. (2023) Language Modeling is Compression
        if model == "transformer200k":
            embedding_dim = 64
            num_heads = 4
            num_layers = 4
            seq_length = 2048
            batch_size = 32
            # 1e6 iterations, which corresponds to 2618 epochs
        elif model == "transformer800k":
            embedding_dim = 128
            num_heads = 4
            num_layers = 4
            seq_length = 2048
            batch_size = 32
            # 1e6 iterations, which corresponds to 2618 epochs
        elif model == "transformer3.2m":
            embedding_dim = 256
            num_heads = 8
            num_layers = 4
            seq_length = 2048
            batch_size = 32
            # 1e6 iterations, which corresponds to 2618 epochs
        elif model == "transformer4.1m":
            embedding_dim = 256
            num_heads = 8
            num_layers = 5
            seq_length = 2048
            batch_size = 32
        elif model == "development": # for development purposes
            embedding_dim = 64
            num_heads = 4
            num_layers = 4
            seq_length = 2048
            batch_size = 2
        else:
            raise ValueError(f"Invalid model size. Please choose a valid model from TransformerConfig (see TransformerConfig.py). Requested config: {model}.")
            
        learning_rate = 1e-4
        ff_dim = 4 * embedding_dim
        
        dict = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "ff_dim": ff_dim,
        }
        return dict