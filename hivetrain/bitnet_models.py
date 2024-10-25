import torch
import torch.nn as nn
import math
from typing import Optional

class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8
    ):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-5
        self.quantization_range = 2 ** (num_bits - 1)
        self.register_buffer('binarized_weights', None)
        
        # Initialize weights with normal distribution
        nn.init.normal_(self.weight, mean=0, std=1/math.sqrt(in_features))
        if bias:
            nn.init.zeros_(self.bias)

    def ste_weights(self, weights_gamma: float):
        eps = 1e-7
        scaled_weights = self.weight / (weights_gamma + eps)
        bin_weights_no_grad = torch.clamp(torch.round(scaled_weights), min=-1, max=1)
        bin_weights_with_grad = (bin_weights_no_grad - scaled_weights).detach() + scaled_weights
        return bin_weights_with_grad

    def binarize_weights(self, weights_gamma: float):
        self.binarized_weights = self.ste_weights(weights_gamma)
        return self.binarized_weights

    def quantize_activations(self, input, input_gamma: float):
        quantized_input = torch.clamp(
            input * self.quantization_range / input_gamma,
            -self.quantization_range + self.eps,
            self.quantization_range - self.eps,
        )
        return quantized_input

    def dequantize_activations(self, input, input_gamma: float, beta: float):
        return input * input_gamma * beta / self.quantization_range

    def forward(self, _input):
        input = _input.view(_input.size(0), -1)
        normalized_input = nn.functional.layer_norm(input, (input.shape[1:]))
        input_gamma = normalized_input.abs().max().item()
        weight_abs_mean = self.weight.abs().mean().item()
        
        binarized_weights = self.binarize_weights(weight_abs_mean)
        input_quant = self.quantize_activations(normalized_input, input_gamma)
        output = nn.functional.linear(input_quant, binarized_weights, self.bias)
        output = self.dequantize_activations(output, input_gamma, weight_abs_mean)
        
        return output

class BitNetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = BitLinear(config.hidden_size, self.all_head_size)
        self.key = BitLinear(config.hidden_size, self.all_head_size)
        self.value = BitLinear(config.hidden_size, self.all_head_size)
        self.dense = BitLinear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return self.dense(context_layer)

def convert_to_bitnet(model: nn.Module) -> nn.Module:
    """
    Convert a regular transformer model to use BitLinear layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BitLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
            ))
        else:
            convert_to_bitnet(module)
    return model

# Example usage with your existing code:
def setup_bitnet_model(model_name: str, device: str = "cuda"):
    """
    Setup a BitNet model from a regular transformer model
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Convert to BitNet
    model = convert_to_bitnet(model)
    return model