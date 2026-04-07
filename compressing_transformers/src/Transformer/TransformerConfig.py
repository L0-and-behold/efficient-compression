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

        The notation tX_Y denotes a transformer with X parameters and roughly Y GB of VRAM usage per GPU.
        If there is a 'p' behind Y, then Y denotes peak VRAM usage requirements for PMMP (which has more parameters than the other methods).
        
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
            batch_size_per_gpu = 1
        elif model == "transformer800k":
            embedding_dim = 128
            num_heads = 4
            num_layers = 4
            seq_length = 2048
            batch_size_per_gpu = 1
        elif model == "transformer3.2m":
            embedding_dim = 256
            num_heads = 8
            num_layers = 4
            seq_length = 2048
            batch_size_per_gpu = 1
        elif model == "transformer4.1m":
            embedding_dim = 256
            num_heads = 8
            num_layers = 5
            seq_length = 2048
            batch_size_per_gpu = 1
        elif model == "big-transformer":
            embedding_dim = 1024
            num_heads = 16
            num_layers = 24
            seq_length = 2048
            batch_size_per_gpu = 32
        elif model == "char-transformer": # similar to Al-Rfou et al. (2018)
            embedding_dim = 512
            num_heads = 8
            num_layers = 64
            seq_length = 2048
            batch_size_per_gpu = 32
        elif model == "development": # for development purposes
            embedding_dim = 64
            num_heads = 4
            num_layers = 4
            seq_length = 2048
            batch_size_per_gpu = 2
        elif model == "t416_49.8":
            embedding_dim = 64 * 23
            num_heads = embedding_dim // 64
            num_layers = 16
            seq_length = 1024
            batch_size_per_gpu = 4
        elif model == "t1006_49.3":
            embedding_dim = 64 * 32
            num_heads = embedding_dim // 64
            num_layers = 20
            seq_length = 512
            batch_size_per_gpu = 2
        elif model == "t1006_44.8":
            embedding_dim = 64 * 32
            num_heads = embedding_dim // 64
            num_layers = 20
            seq_length = 512
            batch_size_per_gpu = 1
        elif model == "t552_42.4":
            embedding_dim = 64 * 25
            num_heads = embedding_dim // 64
            num_layers = 18
            seq_length = 1024
            batch_size_per_gpu = 2
        elif model == "t428_39.5":
            embedding_dim = 64 * 22
            num_heads = embedding_dim // 64
            num_layers = 18
            seq_length = 512
            batch_size_per_gpu = 8
        elif model == "t432_37":
            embedding_dim = 64 * 26
            num_heads = embedding_dim // 64
            num_layers = 13
            seq_length = 512
            batch_size_per_gpu = 12
        elif model == "t432_37_8":
            embedding_dim = 64 * 26
            num_heads = embedding_dim // 64
            num_layers = 13
            seq_length = 512
            batch_size_per_gpu = 8
        elif model == "t486_39":
            embedding_dim = 64 * 30
            num_heads = embedding_dim // 64
            num_layers = 11
            seq_length = 1024
            batch_size_per_gpu = 4
        elif model == "t10_67":
            embedding_dim = 64 * 14
            num_heads = embedding_dim // 64
            num_layers = 7
            seq_length = 1024
            batch_size_per_gpu = 4
        elif model == "t2_9_t432_37":
            embedding_dim = 64 * 6
            num_heads = embedding_dim // 64
            num_layers = 5
            seq_length = 512
            batch_size_per_gpu = 12
        elif model == "t10_67_t432_37":
            embedding_dim = 64 * 14
            num_heads = embedding_dim // 64
            num_layers = 7
            seq_length = 512
            batch_size_per_gpu = 12
        elif model == "t2_9":
            embedding_dim = 64 * 6
            num_heads = embedding_dim // 64
            num_layers = 5
            seq_length = 1024
            batch_size_per_gpu = 4
        elif model == "t294_38.8":
            embedding_dim = 64 * 20
            num_heads = embedding_dim // 64
            num_layers = 15
            seq_length = 1024
            batch_size_per_gpu = 4
        elif model == "t125_37.8":
            embedding_dim = 64 * 14
            num_heads = embedding_dim // 64
            num_layers = 13
            seq_length = 1024
            batch_size_per_gpu = 8
        elif model == "gpt2-medium":
            embedding_dim = 64 * 16
            num_heads = embedding_dim // 64
            num_layers = 24
            seq_length = 1024
            batch_size_per_gpu = 4
        elif model == "t307_38p":
            embedding_dim = 64 * 25
            num_heads = embedding_dim // 64
            num_layers = 10
            seq_length = 512
            batch_size_per_gpu = 8
        elif model == "t337_42p":
            embedding_dim = 64 * 25
            num_heads = embedding_dim // 64
            num_layers = 10
            seq_length = 512
            batch_size_per_gpu = 8
        elif model == "t400_50p":
            embedding_dim = 64 * 25
            num_heads = embedding_dim // 64
            num_layers = 13
            seq_length = 512
            batch_size_per_gpu = 8
        else:
            raise ValueError(f"Invalid model size. Please choose a valid model from TransformerConfig (see TransformerConfig.py). Requested config: {model}.")
            
        learning_rate = 1e-4
        ff_dim = 4 * embedding_dim
        
        dict = {
            "batch_size_per_gpu": batch_size_per_gpu,
            "seq_length": seq_length,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "ff_dim": ff_dim,
        }
        return dict