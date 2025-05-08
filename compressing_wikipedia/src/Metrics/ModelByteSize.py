import scipy.sparse
import scipy
import numpy as np
import torch

class ModelByteSize:
    """
    Utility for computing model size in bytes and managing parameters in sparse format.
    """

    @staticmethod
    def get_model_info(model, device):
        """
        Compute model statistics including total parameters and non-zero count.
        
        Returns a dictionary with trainable and non-zero parameter counts.
        """
        model = model.to(device)

        trainable_param_size = 0
        non_trainable_param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param = param.to(device)
            if param.requires_grad:
                trainable_param_size += param.nelement() * param.element_size()
            else:
                non_trainable_param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer = buffer.to(device)
            buffer_size += buffer.nelement() * buffer.element_size()

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_zero_params = sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)

        return {
            "trainable_params": total_trainable_params,
            "non_zero_params": non_zero_params
        }

    @staticmethod
    def get_model_matrices(model):
        """
        Extract trainable weight and bias matrices from the model.
        
        Returns a list of tensors representing weights with biases incorporated.
        """
        matrices = []
        total_params_model = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    weight_matrix = param.data
                    total_params_model += weight_matrix.numel()

                    # Check if there's a corresponding bias
                    bias_name = name.replace('weight', 'bias')
                    bias = None
                    for bias_param_name, bias_param in model.named_parameters():
                        if bias_param_name == bias_name and bias_param.requires_grad:
                            bias = bias_param.data
                            break
                    
                    if bias is not None:
                        total_params_model += bias.numel()
                        if weight_matrix.ndimension() == 2:
                            if bias.shape[0] == weight_matrix.shape[0]:
                                bias = bias.unsqueeze(1)
                                weight_matrix = torch.cat([weight_matrix, bias], dim=1)
                        elif weight_matrix.ndimension() == 1:
                            weight_matrix = torch.cat([weight_matrix, bias], dim=0)

                    matrices.append(weight_matrix)
        return matrices

    @staticmethod
    def byte_size(model):
        """
        Compute model size in bytes when using optimal sparse matrix format.
        
        Returns the minimum of COO and CSC sparse representation sizes.
        """
        matrices = ModelByteSize.get_model_matrices(model)
        total_size_coo = 0
        total_size_csc = 0

        for matrix in matrices:
            total_size_coo += ModelByteSize.coo_size(matrix)
            total_size_csc += ModelByteSize.csc_size(matrix)        
            
        return min(total_size_coo, total_size_csc)

    @staticmethod
    def coo_size(tensor):
        """
        Calculate size of tensor in COO sparse format in bytes.
        
        Raises ValueError if tensor elements are not np.float32.
        """
        numpy_array = tensor.cpu().numpy()
        coo = scipy.sparse.coo_matrix(numpy_array)

        if coo.data.size == 0: # empty matrix, no non-zero elements
            return 0
        elif type(coo.data[0]) is not np.float32:
            raise ValueError("The elements of the COO matrix are not np.float32.")
        
        return coo.size * 4  # unit: bytes

    @staticmethod
    def csc_size(tensor):
        """
        Calculate size of tensor in CSC sparse format in bytes.
        
        Raises ValueError if tensor elements are not np.float32.
        """
        numpy_array = tensor.cpu().numpy()
        csc = scipy.sparse.csc_matrix(numpy_array)

        if csc.data.size == 0: # empty matrix, no non-zero elements
            return 0
        elif type(csc.data[0]) is not np.float32:
            raise ValueError("The elements of the CSC matrix are not np.float32.")
        
        return csc.size * 4  # unit: bytes
