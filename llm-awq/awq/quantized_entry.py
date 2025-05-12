import torch
from safetensors.torch import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import threading
import json


class DynamicThresholdMistralMLP(nn.Module):
    def __init__(self, original_mlp, layer_index):
        super(DynamicThresholdMistralMLP, self).__init__()
        self.original_mlp = original_mlp
        self.layer_index = layer_index
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
        self.act_fn = original_mlp.act_fn

    def update_parameters(self, gate_proj_params, up_proj_params, down_proj_params):
        """
        Update the MLP layers dynamically based on new parameters and dimensions.
        """
        print("updating parameters...")
        self.gate_proj = nn.Linear(gate_proj_params.shape[1], gate_proj_params.shape[0], bias=False)
        self.gate_proj.weight = nn.Parameter(gate_proj_params)

        self.up_proj = nn.Linear(up_proj_params.shape[1], up_proj_params.shape[0], bias=False)
        self.up_proj.weight = nn.Parameter(up_proj_params)

        self.down_proj = nn.Linear(down_proj_params.shape[1], down_proj_params.shape[0], bias=False)
        self.down_proj.weight = nn.Parameter(down_proj_params)

    def forward(self, x):
        gate_x = self.gate_proj(x)
        up_x = self.up_proj(x)

        # Match dimensions by padding
        max_dim = max(gate_x.size(-1), up_x.size(-1))
        if gate_x.size(-1) < max_dim:
            pad_size = max_dim - gate_x.size(-1)
            gate_x = F.pad(gate_x, (0, pad_size))  # Pad at the end
        if up_x.size(-1) < max_dim:
            pad_size = max_dim - up_x.size(-1)
            up_x = F.pad(up_x, (0, pad_size))

        act_x = self.act_fn(gate_x) * up_x
        down_x = self.down_proj(act_x)
        return down_x


class PrefetchingLoader:
    def __init__(self, safetensor_dir, filter_data, model_index_file):
        self.safetensor_dir = safetensor_dir
        self.filter_data = filter_data
        self.model_index_file = model_index_file
        self.prefetch_buffer = {}
        self.prefetch_lock = threading.Lock()
        self.model_dtype = None

    def set_model_dtype(self, dtype):
        """
        Set the model's dtype after initialization.
        """
        self.model_dtype = dtype

    def prefetch_parameters(self, token_index):
        """
        Prefetch parameters for the next token based on the filter.
        """
        with self.prefetch_lock:
            print(f"Prefetching parameters for token {token_index}...")
            params = self._load_parameters(token_index)
            if params:
                print(f"Adding parameters for token {token_index} to prefetch_buffer.")
                self.prefetch_buffer[token_index] = params
            else:
                print(f"No parameters found for token {token_index}.")

    def _load_parameters(self, token_index):
        """
        Load parameters for a specific token using the filter data.
        """
        parameters = {}

        # Iterate over the 32 layers
        for layer_index, layer_filter in enumerate(self.filter_data):
            largest_dim = max(layer_filter["gate_proj"].sum(), layer_filter["up_proj"].sum())
            parameters[layer_index] = {
                "gate_proj": self._load_filtered_neurons(
                    f"model.layers.{layer_index}.mlp.gate_proj.weight",
                    layer_filter["gate_proj"][:, token_index, :].squeeze(0),  # Extract specific token's filter
                ),
                "up_proj": self._load_filtered_neurons(
                    f"model.layers.{layer_index}.mlp.up_proj.weight",
                    layer_filter["up_proj"][:, token_index, :].squeeze(0),  # Extract specific token's filter
                ),
                "down_proj": self._load_filtered_neurons(
                    f"model.layers.{layer_index}.mlp.down_proj.weight",
                    layer_filter["down_proj"][:, token_index, :].squeeze(0),
                      input_dim=largest_dim  # Extract specific token's filter
                ),
            }
        return parameters

    def _load_filtered_neurons(self, tensor_name, filter_mask, input_dim=None):
        """
        Load only the filtered neurons for a specific tensor.
        """
        filepath = f"{self.safetensor_dir}/{self._find_file(tensor_name)}"
        with safe_open(filepath, framework="pt", device="cpu") as f:
            full_tensor = f.get_tensor(tensor_name)

            # Adjust filtering based on tensor shape
            if full_tensor.shape[0] == filter_mask.shape[0]:  # Apply filter to rows
                filtered_tensor = full_tensor[filter_mask == 1, :]
            elif full_tensor.shape[1] == filter_mask.shape[0]:  # Apply filter to columns
                filtered_tensor = full_tensor[:, filter_mask == 1]
            else:
                raise ValueError(
                    f"Filter mask shape {filter_mask.shape} does not match tensor shape {full_tensor.shape}"
                )
            
            # Adjust input dimension if needed
            if input_dim is not None:
                # Expand input features to match the largest dimension
                expanded_tensor = torch.zeros(input_dim, filtered_tensor.shape[1], dtype=filtered_tensor.dtype)
                expanded_tensor[:filtered_tensor.shape[0], :] = filtered_tensor
                filtered_tensor = expanded_tensor

            # Cast to match model dtype
            filtered_tensor = filtered_tensor.to(self.model_dtype)
            print(f"tensor_name {tensor_name} filtered_tensor {filtered_tensor.shape}")
            return filtered_tensor

    def _find_file(self, tensor_name):
        """
        Find the corresponding file for a given tensor in the index file.
        """
        with open(self.model_index_file, "r") as f:
            index_data = json.load(f)
        return index_data["weight_map"][tensor_name]

    def get_parameters(self, token_index):
        """
        Retrieve preloaded parameters for the current token.
        """
        with self.prefetch_lock:
            if token_index in self.prefetch_buffer:
                print(f"Token {token_index} found in prefetch_buffer. Retrieving parameters.")
            else:
                print(f"Token {token_index} NOT found in prefetch_buffer. Current keys: {list(self.prefetch_buffer.keys())}")
            return self.prefetch_buffer.pop(token_index, None)


class SelectiveMLPModel:
    def __init__(self, model_path, safetensor_dir, model_index_file, filter_file):
        self.model_path = model_path
        self.safetensor_dir = safetensor_dir
        self.model_index_file = model_index_file
        self.filter_data = torch.load(filter_file)
        self.prefetch_loader = PrefetchingLoader(safetensor_dir, self.filter_data, model_index_file)
        self.tokenizer = None
        self.model = None

    def load_base_model(self):
        """
        Load the Hugging Face model and replace MLPs with DynamicThresholdMistralMLP instances.
        """
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.prefetch_loader.set_model_dtype(self.model.dtype)

        for layer_index, layer in enumerate(self.model.model.layers):
            original_mlp = layer.mlp
            dynamic_mlp = DynamicThresholdMistralMLP(original_mlp, layer_index)
            
            if hasattr(self.model, "dtype"):
                dynamic_mlp.to(self.model.dtype)
            layer.mlp = dynamic_mlp

        print("Base model loaded with DynamicThresholdMistralMLP.")

    def evaluate(self, dataset_name="wikitext", config="wikitext-2-raw-v1", seq_len=2048):
        """
        Evaluate the model on the Wikitext dataset with dynamic MLP updates and prefetching.
        """
        print("Evaluating on Wikitext...")
        dataset = load_dataset(dataset_name, config, split="test")
        full_text = "\n\n".join(dataset["text"])  # Join all text as in the original code
        encoded_data = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.model.device)
        num_tokens = encoded_data.size(1)  # Total number of tokens
        print(f"Number of Tokens: {num_tokens}")

        nsamples = num_tokens // seq_len  # Determine how many full samples we can process
        if nsamples == 0:
            raise ValueError(f"Insufficient tokens ({num_tokens}) for sequence length ({seq_len}).")

        # Prefetch for the first token of the first sample
        print(f"Starting prefetching for token 0...")
        self.prefetch_loader.prefetch_parameters(0)

        nlls = []
        for sample_index in range(nsamples):
            print(f"Processing sample {sample_index + 1}/{nsamples}...")

            # Extract the current sample
            batch = encoded_data[:, sample_index * seq_len: (sample_index + 1) * seq_len]
            print(f"Sample {sample_index + 1}: batch dtype: {batch.dtype}, shape: {batch.shape}")

            # Process token by token with prefetching
            for token_index in range(seq_len):
                input_token = batch[:, token_index: token_index + 1]
                print(f"Token {token_index}: input_token {input_token} input_token dtype: {input_token.dtype}, shape: {input_token.shape}")

                # Wait for prefetching and get parameters for the current token
                current_parameters = self.prefetch_loader.get_parameters(token_index)
                print(current_parameters)
                if current_parameters:
                    print(f"Updating MLP parameters for token {token_index}...")
                    # Dynamically update the MLP layers for the current token
                    for layer_index, params in current_parameters.items():
                        print(f"Layer {layer_index}: gate_proj dtype: {params['gate_proj'].dtype}, "
                            f"up_proj dtype: {params['up_proj'].dtype}, "
                            f"down_proj dtype: {params['down_proj'].dtype}")
                        self.model.model.layers[layer_index].mlp.update_parameters(
                            params["gate_proj"], params["up_proj"], params["down_proj"]
                        )
                        print(f"Checking if update_parameters exists for Layer {layer_index}")
                        if hasattr(self.model.model.layers[layer_index].mlp, 'update_parameters'):
                            print(f"update_parameters exists for Layer {layer_index}")
                        else:
                            print(f"update_parameters does NOT exist for Layer {layer_index}")
                                        
                
                # Prefetch parameters for the next token
                if token_index < seq_len - 1:
                    print(f"Prefetching parameters for token {token_index + 1}...")
                    threading.Thread(
                        target=self.prefetch_loader.prefetch_parameters, args=(token_index + 1,)
                    ).start()

                # Process the token
                with torch.no_grad():
                    print(f"Processing token {token_index}...")
                    outputs = self.model(input_token)
                    logits = outputs.logits
                    print(f"Token {token_index}: logits dtype: {logits.dtype}, shape: {logits.shape}")

                    # Calculate loss for next-token prediction (starting from the second token)
                    if token_index > 0:
                        previous_token = batch[:, token_index - 1]
                        print(f"Calculating loss for token {token_index}...")
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            previous_token.view(-1),
                            reduction="sum",
                        )
                        nlls.append(loss.item())
                        print(f"Loss for token {token_index}: {loss.item()}")

            print(f"Sample {sample_index + 1}/{nsamples} processed.")

        # Compute Perplexity
        total_nll = sum(nlls)
        perplexity = torch.exp(torch.tensor(total_nll / (seq_len * nsamples)))
        print(f"Perplexity: {perplexity.item()}")

        return perplexity.item()


# Example Usage
model_path = "models/Mistral-7B-v0.1"
safetensor_dir = "models/Mistral-7B-v0.1"
model_index_file = "models/Mistral-7B-v0.1/model.safetensors.index.json"
filter_file = "full_lenght_byte_dataset_chunks/chunk_0.pt"

selective_model = SelectiveMLPModel(model_path, safetensor_dir, model_index_file, filter_file)
selective_model.load_base_model()
print(selective_model.model)
selective_model.evaluate()