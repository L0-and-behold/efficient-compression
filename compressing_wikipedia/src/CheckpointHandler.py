import os
import pickle
import time

class CheckpointHandler:
    """
    Manages model checkpointing with runtime-based saving and restoration.
    """
    def __init__(self, experiment_name, checkpoint_every, max_runtime, checkpoint_dir=None):
        """
        Initialize with experiment name, checkpoint frequency, and runtime limits.
        """
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir or os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{experiment_name}_checkpoint.pkl")
        self.epoch = 0
        self.chunk = 0
        self.start_time = time.time()
        self.last_check_time = self.start_time

        self.checkpoint_every = checkpoint_every
        self.max_runtime = max_runtime

    def save_checkpoint(self, ddp_model, optimizer, logs):
        """
        Save model state, optimizer state, and training logs to checkpoint file.
        """
        checkpoint = {
            'epoch': self.epoch,
            'chunk': self.chunk,
            'model_state_dict': ddp_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logs': logs
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self):
        """
        Load checkpoint if available or initialize fresh training state.
        
        Returns model state, optimizer state, and logs.
        """
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.epoch = checkpoint['epoch']
            self.chunk = checkpoint['chunk']
            print(f"Loaded checkpoint from epoch {self.epoch}, chunk {self.chunk}")
            return checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'], checkpoint['logs']
        else:
            print("No checkpoint found, starting from the beginning")
            return None, None, initialize_logs()

    def update(self, epoch, chunk):
        """
        Update internal epoch and chunk counters.
        """
        self.epoch = epoch
        self.chunk = chunk

    def should_checkpoint(self):
        """
        Determine if checkpointing should occur based on time intervals.
        
        Returns True if max runtime exceeded or checkpoint interval reached.
        """
        current_time = time.time()
        if current_time - self.last_check_time > self.checkpoint_every:
            self.last_check_time = current_time
            if current_time - self.start_time > self.max_runtime:
                return True
        return False


def initialize_logs():
    """
    Create empty log dictionary with required tracking metrics.
    """
    return {
        "train_loss": [], "train_loss_X": [], "train_loss_X_epoch": [], 
        "runtime_per_250_batches": [], "l0_norm": [], "l0_norm_X": [], 
        "l0_norm_X_epoch": []
    }