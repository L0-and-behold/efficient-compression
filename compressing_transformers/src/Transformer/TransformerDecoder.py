import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class DecoderBlock(nn.Module):
    """Standard transformer decoder block with self-attention and feed-forward layers."""
    
    def __init__(self, d_model, num_heads, ff_hidden_layer):
        """Initialize a decoder block with attention and feed-forward components.
        
        Args:
            d_model: Dimension of the model embeddings
            num_heads: Number of attention heads
            ff_hidden_layer: Dimension of the feed-forward hidden layer
        """
        super(DecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_hidden_layer)
        self.linear2 = nn.Linear(ff_hidden_layer, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, target_mask):
        """Process input through self-attention and feed-forward with residual connections.
        
        Args:
            x: Input tensor of shape (seq_len, batch, d_model)
            target_mask: Attention mask to prevent attending to future tokens
            
        Returns:
            Processed tensor of shape (seq_len, batch, d_model)
        """
        attn_output, _ = self.self_attention(x, x, x, attn_mask=target_mask)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + ff_output
        x = self.norm2(x)
        return x


class PositionalEncoding(nn.Module):
    """Add positional information to input embeddings using sinusoidal encoding."""
    
    def __init__(self, d_model, max_len = 5000):
        """Initialize the positional encoding module.
        
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length to pre-compute encodings for
        """        
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        
    def forward(self, x):
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (seq_len, batch, d_model)
            
        Returns:
            Tensor with positional encoding added
        """        
        pe = self.pe[:x.size(0), :x.size(1), :]
        x = x + pe
        return x


class MultiLayerTransformerDecoder(nn.Module):
    """Multi-layer transformer decoder with optional PMMP parameter support."""
    
    def __init__(self, d_model, num_heads, ff_hidden_layer, 
                 num_layers, alphabet_size = 256, pmmp = False, 
                 initial_p_value = 0.5, dev = 'cuda'):
        """Initialize transformer decoder with model configuration.
        
        Args:
            d_model: Dimension of the model embeddings
            num_heads: Number of attention heads
            ff_hidden_layer: Dimension of the feed-forward hidden layer
            num_layers: Number of decoder blocks in the stack
            alphabet_size: Size of the token vocabulary
            pmmp: Whether to use PMMP  parameters
            initial_p_value: Initial probability value for PMMP
            dev: Device to use for PMMP parameters ('cuda' or 'cpu')
        """        
        super(MultiLayerTransformerDecoder, self).__init__()
        torch.manual_seed(torch.initial_seed())  # use PyTorch global seed
        self.d_model = d_model
        self.embedding = nn.Embedding(alphabet_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_hidden_layer)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, alphabet_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.initialize_weights()
        
        if pmmp:
            self.dev = dev
            self.initial_p_value = initial_p_value
            self.create_pmmp_params()
    
    def initialize_weights(self):
        """Initialize model weights with fixed random seed for reproducibility."""        
        current_seed = torch.initial_seed()
        
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Ensure deterministic initialization
                torch.manual_seed(current_seed)
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                
        self.apply(_init_weights)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal attention mask to prevent attending to future tokens.
        
        Args:
            sz: Size of the square mask
            
        Returns:
            Causal attention mask where future positions are masked with -inf
        """        
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def count_parameters(self):
        """Count trainable parameters in the model.
        
        Returns:
            Total number of trainable parameters
        """        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """Process input through embedding, positional encoding, and transformer layers.
        
        Args:
            x: Input tensor of token indices with shape (batch, seq_len)
            
        Returns:
            Log-probability distribution over vocabulary for each position
        """        
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Change to (seq_len, batch, d_model)
        x = x * math.sqrt(self.d_model)
        pos_encoding = self.pos_encoder(x)
        x = x + pos_encoding
        
        target_mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, target_mask)
            
        x = x.transpose(0, 1)  # Change back to (batch, seq_len, d_model)
        output = self.linear(x)
        output = self.softmax(output)
        return output
    
    @staticmethod  # if @staticmethod is used, then self is not passed to zero_params
    def fill_params(ps, a):
        """Fill all parameters in collection with a single value.
        
        Args:
            ps: Iterable of parameters
            a: Value to fill parameters with
        """        
        with torch.no_grad():
            for param in ps:
                if torch.is_tensor(param):
                    param.fill_(a)
    
    def create_pmmp_params(self):
        """Create parameter copies for PMMP optimization.
        
        Sets up w, p, and u parameters required for minimax pruning.
        In the pmmp loss function, in the part `u(θ - w γ)² + u(w² γ (1 - γ))`, 
        the below defined p corresponds to γ, while w and u correspond to w and u
        """        
        self.w = [copy.deepcopy(param.data.detach()).to(self.dev) for param in self.parameters()]
        self.p = [copy.deepcopy(param.data.detach()).to(self.dev) for param in self.parameters()]
        self.u = [copy.deepcopy(param.data.detach()).to(self.dev) for param in self.parameters()]
        self.fill_params(self.p, self.initial_p_value)
        self.fill_params(self.u, 0)
        for (a, w, p, u) in zip(self.parameters(), self.parameters_w(),self.parameters_p(), self.parameters_u()):
            if a.requires_grad:
                w.requires_grad = True
                p.requires_grad = True
                u.requires_grad = True
    
    # Create generators for the copied parameters to make iterations more convenient
    # with those generators, one can do lazy iterations like `for (a,b) in zip(net.parameters(),net.parameters_w()): ...`
    def parameters_w(self):
        """Yield w parameters for PMMP optimization.
        
        Yields:
            Parameter tensors for the w component of PMMP
        """
        for param in self.w:
            yield param
    
    def parameters_p(self):
        """Yield probability parameters for PMMP optimization.
        
        Yields:
            Parameter tensors for the p component of PMMP (γ in the equations)
        """
        for param in self.p:
            yield param
    
    def parameters_u(self):
        """Yield u parameters for PMMP optimization.
        
        Yields:
            Parameter tensors for the u component of PMMP
        """
        for param in self.u:
            yield param