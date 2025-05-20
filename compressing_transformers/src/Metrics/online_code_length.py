from src.Transformer.TransformerConfig import TransformerConfig
from src.Database.RunsCSV import RunsCSV
import pandas as pd
import numpy as np
import os

def online_code_length(run_df: RunsCSV, args: dict) -> float:
    """Calculate compression code length in the online training setting.
    
    In the online setting, the transformer is trained on a (small) batch of data. 
    The data is then compressed using the partially trained model.
    The coding length of transmitting a dataset in this setting can be calculated 
    from the loss (which is cross-entropy based on natural log) since it directly 
    relates to the coding length (which is log2 based).
    
    Args:
        run_df (RunsCSV): The run data frame containing training history
        args (dict): Parameters dictionary containing batch_size, transformer_config,
                     and log_every values
                     
    Returns:
        float: The calculated code length for the dataset compression
        
    Raises:
        AssertionError: If required parameters are missing from args
    """
    assert "batch_size" in args, "batch_size not in args"
    assert "transformer_config" in args, "transformer_config not in args"
    assert "log_every" in args, "log_every not in args"
    
    train_loss_df = pd.read_csv(os.path.join(run_df.run.path, "train_loss.csv"))
    batch_size = args["batch_size"]
    chunk_size = TransformerConfig(args["transformer_config"])()["seq_length"]
    train_loss = train_loss_df['train_loss'].to_numpy()
    total_log2_loss = np.sum(train_loss / np.log(2))
    code_length = total_log2_loss * chunk_size * batch_size * args["log_every"]
    
    return code_length

def online_code_length_from_dict(logs: dict, args: dict) -> float:
    """Calculate compression code length in the online training setting from logs dictionary.
    
    In the online setting, the transformer is trained on a (small) batch of data.
    The data is then compressed using the partially trained model.
    The coding length of transmitting a dataset in this setting can be calculated
    from the loss (which is cross-entropy based on natural log) since it directly
    relates to the coding length (which is log2 based).
    
    Args:
        logs (dict): Dictionary containing training logs with 'train_loss' key
        args (dict): Parameters dictionary containing batch_size, transformer_config,
                     and log_every values
                     
    Returns:
        float: The calculated code length for the dataset compression
        
    Raises:
        AssertionError: If required parameters are missing from args
    """
    assert "batch_size" in args, "batch_size not in args"
    assert "transformer_config" in args, "transformer_config not in args"
    assert "log_every" in args, "log_every not in args"
    
    train_loss = logs['train_loss']
    batch_size = args["batch_size"]
    chunk_size = TransformerConfig(args["transformer_config"])()["seq_length"]
    total_log2_loss = np.sum(train_loss / np.log(2))
    code_length = total_log2_loss * chunk_size * batch_size * args["log_every"]
    
    return code_length