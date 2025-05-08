import torch
import torch.nn.functional as F

class GenerativeModel:
    """
    Text generation utility for transformer-based models.
    
    Handles text generation, next-token prediction, and token probability distribution.
    """
    
    def __init__(self, model, device):
        """
        Initializes the GenerativeModel with a pre-trained model (see `TransformerDecoder` class) and a device.
        """
        self.model = model.to(device)
        self.device = device

    def output_probabilities(self, input_sequence, debug=False):
        """
        Generates the probability distributions for each token in a sequence.
        For an input sequence of length `n`, this method returns a tensor of shape `(n, vocab_size)`.
        The [i, :]-slice is the probability distribution over the vocabulary for the i-th token (softmax output).
        The [:, i]-slice are the probabilities that the next token is the i-th letter in the vocabulary, having seen the previous tokens.
        """
        self.model.eval()

        with torch.no_grad():
            input_sequence = input_sequence.to(self.device)
            
            try:
                output = self.model(input_sequence)
            except RuntimeError as e:
                print(f"Error in model forward pass. Are all tensors on the correct, same, device?")
                raise RuntimeError(e)
            
            probability_distributions = F.softmax(output, dim=-1)
        
        return probability_distributions

    def predict_next_token(self, input_sequence):
        """
        Predict probability distribution for the next token following input sequence.
        
        Useful for arithmetic coding or as a component in text generation.
        """
        self.model.eval()

        with torch.no_grad():
            input_sequence = input_sequence.to(self.device)
            
            # If the input sequence is missing a batch dimension, add it
            if len(input_sequence.shape) == 1:
                input_sequence = input_sequence.unsqueeze(0)

            try:
                output = self.model(input_sequence)
            except RuntimeError as e:
                print(f"Error in model forward pass. Are all tensors on the correct, same, device?")
                raise e
            
            # Only interested in the last token's logits for next-token prediction
            last_token_logits = output[:, -1, :]
            
            probability_distribution = F.softmax(last_token_logits, dim=-1)
        
        return probability_distribution

    def generate_next_token(self, input_sequence):
        """
        Generates the next token based on the given input sequence by selecting the most likely token.
        The generated token is determined by the `argmax` of the probability distribution over possible tokens.
        """
        next_token_probs = self.predict_next_token(input_sequence)
        
        if next_token_probs is not None:
            # Change method to get token from distribution here if applicable
            most_likely_token = torch.argmax(next_token_probs, dim=-1)
            return most_likely_token[0].item()
        else:
            return None

    def generate_text(self, initial_sequence, max_length=1000):
        """
        Iteratively generates a sequence of text starting from an initial sequence.

        Use the `string_to_input_sequence` method to convert a string into a valid input sequence to this function.
        """
        generated_text = initial_sequence.tolist()
        
        for _ in range(max_length):
            next_token = self.generate_next_token(torch.LongTensor(generated_text).to(self.device))
            
            if next_token is None:
                break

            generated_text.append(next_token)
            
            # Stop if the next token is an end-of-sequence token (e.g., 0)
            if next_token == 0:
                break
        
        return "".join([chr(token) for token in generated_text])

    def string_to_input_sequence(self, string, sequence_length):
        """
        Convert string to tensor of ASCII values with appropriate padding/truncation.
        
        Prints the processed string and returns the corresponding tensor.
        """
        ascii_values = [ord(char) for char in string]

        if len(ascii_values) < sequence_length:
            ascii_values += [0] * (sequence_length - len(ascii_values))

        ascii_values = ascii_values[:sequence_length]

        ascii_string = "".join([chr(val) for val in ascii_values])
        print(f"After padding/truncating, the input sequence reads: {ascii_string}")

        input_sequence = torch.LongTensor(ascii_values)

        return input_sequence
