from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--dump_fake", type=str, default=None, help="save fake-quantized model")
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)
args = parser.parse_args()
vila_10_quant_mode = ("llava" in args.model_path.lower() or "vila" in args.model_path.lower()) and not args.vila_15

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path):
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False}
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(
        #         config=config, torch_dtype=torch.float16, trust_remote_code=True
        #     )
        # real_quantize_model_weight(
        #     model, w_bit=args.w_bit, q_config=q_config, init_only=True
        # )

        # model.tie_weights()

        # # Infer device map
        # kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        # device_map = infer_auto_device_map(
        #     model,
        #     no_split_module_classes=[
        #         "OPTDecoderLayer",
        #         "LlamaDecoderLayer",
        #         "BloomBlock",
        #         "MPTBlock",
        #         "DecoderLayer",
        #     ],
        #     **kwargs,
        # )
        # # Load checkpoint in the model
        # load_checkpoint_in_model(
        #     model,
        #     checkpoint=args.load_quant,
        #     device_map=device_map,
        #     offload_state_dict=True,
        # )
        # # Dispatch model
        # model = simple_dispatch_model(model, device_map=device_map)

        # model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        if not vila_10_quant_mode:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()

        # if args.run_awq:
        #     assert args.dump_awq, "Please save the awq results with --dump_awq"

        #     awq_results = run_awq(
        #         model,
        #         enc,
        #         w_bit=args.w_bit,
        #         q_config=q_config,
        #         n_samples=128,
        #         seqlen=512,
        #     )
        #     if args.dump_awq:
        #         dirpath = os.path.dirname(args.dump_awq)
        #         os.makedirs(dirpath, exist_ok=True)

        #         torch.save(awq_results, args.dump_awq)
        #         print("AWQ results saved at", args.dump_awq)

        #     exit(0)

        # if args.load_awq:
        #     print("Loading pre-computed AWQ results from", args.load_awq)
        #     awq_results = torch.load(args.load_awq, map_location="cpu")
        #     apply_awq(model, awq_results)

        # # weight quantization
        # if args.w_bit is not None:
        #     if args.q_backend == "fake":
        #         assert (
        #             args.dump_quant is None
        #         ), "Need to use real quantization to dump quantized weights"
        #         pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
        #         if args.dump_fake:
        #             model.save_pretrained(args.dump_fake)
        #             print("Pseudo-quantized models saved at", args.dump_fake)
        #     elif args.q_backend == "real":  # real quantization
        #         real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
        #         if args.dump_quant:
        #             if not args.dump_quant.endswith("v2.pt"):
        #                 print("[Info] Auto-change the dump_quant file name to *v2.pt")
        #                 args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
        #             dirpath = os.path.dirname(args.dump_quant)
        #             os.makedirs(dirpath, exist_ok=True)

        #             print(f"Saving the quantized model at {args.dump_quant}...")
        #             torch.save(model.cpu().state_dict(), args.dump_quant)
        #             exit(0)
        #     else:
        #         raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "MistralDecoderLayer",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc

class ThresholdMistralMLP(nn.Module):
    def __init__(self, original_mlp, thresholds, layer_index):
        super(ThresholdMistralMLP, self).__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.thresholds = thresholds
        self.layer_index = layer_index
        self.count = 0
        # self.gate_centroid = torch.load("../clustering/clustering_results_50_mistral_weighted/centroids_gate_512_sparsity_30.pt", map_location="cuda:1")
        # self.up_centroid = torch.load("../clustering/clustering_results_50_mistral_weighted/centroids_up_512_sparsity_30.pt", map_location="cuda:1")
        # self.down_centroid = torch.load("../clustering/clustering_results_50_mistral_weighted/centroids_down_512_sparsity_30.pt", map_location="cuda:1")

    def apply_threshold(self, x, category):
        threshold_value = self.thresholds[category][f'layer_{self.layer_index}']
        condition = (x >= threshold_value) | (x <= -threshold_value)
        print(f"Applying threshold {threshold_value} for {category}, layer {self.layer_index}")
        return torch.where(condition, x, torch.zeros_like(x))
    
    def simulate_prefetching(self, x, activation_states, target_device='cuda:0'):

        # print(f"x : {torch.sum(x == 0)} zeros, activation_states : {torch.sum(activation_states == 0)}")
        # Store the original device of x
        original_device = x.device
        # Move tensors to the target GPU for computation
        activation_states = activation_states.to(target_device)
        x = x.to(target_device)
        print(activation_states.shape)
        # Create a condition where neurons are either activated (state == 1) or deactivated (state == 0)
        condition = (activation_states == 1)
        # print(f"Applying activation states for layer {self.layer_index} on {target_device}")
        print(condition.shape)
        # Perform the operation on the target GPU
        result = torch.where(condition, x, torch.zeros_like(x))
        # Move the result back to the original device (if further operations are needed on the original GPU)
        result = result.to(original_device)
        return result

    def forward(self, x):
        # global mlp_active_neuron_indices
        # if self.layer_index == 0:
        #     mlp_active_neuron_indices = []
        # output_dir = "full_lenght_byte_dataset_chunks_50"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # global count
        #activation_samples = {
        #'gate_up_proj': [[] for _ in range(32)],
        #'activation_fn': [[] for _ in range(32)],
        #'gate_proj': [[] for _ in range(32)],
        #'up_proj': [[] for _ in range(32)],
        #'down_proj': [[] for _ in range(32)],
        #'act_fn': [[] for _ in range(32)]
        #}

        # global predicted_gate_proj, predicted_up_proj, predicted_down_proj

        gate_x = self.gate_proj(x)


        print(f"gate_proj output before threshold: {torch.sum(gate_x == 0)} zeros")
        gate_thresholded = self.apply_threshold(gate_x, 'gate_proj')
        # gate_labels, gate_clusters= cluster(x=gate_thresholded, centroids=self.gate_centroid)

        # label_collector['gate'].append(gate_labels.cpu())
        gate_x = gate_thresholded
        # activation_states = prepare_activation_states(predicted_gate_proj, self.layer_index)
        # gate_x = self.simulate_prefetching(gate_x, gate_clusters)
        
        print(f"gate_proj output after threshold: {torch.sum(gate_x == 0)} zeros")
        print(f"output of gate: {gate_x.shape}")
        #gate_proj_top_50_indices = torch.topk(gate_x, 7168, dim=2, largest=True).indices

        # gate_proj_binary_mask = (gate_x != 0).int()

        # abs_output = torch.abs(gate_x.detach().cpu()).numpy()
        # if self.layer_index == 0:
        #     activation_samples["gate_proj"][self.layer_index].append(abs_output)
        #     global similarity
        #     with h5py.File(f"Gate_proj_sample_para{8}_layer_1_similarity{70}.h5", "w") as f:
        #         for category in activation_samples:
        #             grp = f.create_group(category)
        #             for i, layer_activations in enumerate(activation_samples[category]):
        #                 if len(layer_activations) > 0:
        #                     dset = grp.create_dataset(f"layer_{i}", data=np.concatenate(layer_activations, axis=0))
        #                     self.count  = self.count +1
        #                     similarity = 90

        
        # Apply threshold to up_proj output
        up_x = self.up_proj(x)
        print(f"up_proj output before threshold: {torch.sum(up_x == 0)} zeros")
        up_thresholded = self.apply_threshold(up_x, 'up_proj')
        # up_labels, up_clusters = cluster(x=up_thresholded, centroids=self.up_centroid)
        # label_collector['up'].append(up_labels.cpu())
        up_x = up_thresholded
        # activation_states = prepare_activation_states(predicted_up_proj, self.layer_index)
        # up_x = self.simulate_prefetching(up_x, up_clusters)

        #up_proj_top_50_indices = torch.topk(up_x, 7168, dim=2, largest=True).indices
        # up_proj_binary_mask = (up_x != 0).int()

        print(f"up_proj output after threshold: {torch.sum(up_x == 0)} zeros")
        print(f"output of up_x: {up_x.shape}")
        # Apply activation function
        act_x = self.act_fn(gate_x) * up_x
        print(f"act_fn output before threshold: {torch.sum(act_x == 0)} zeros")
        
        # Apply threshold to activation output
        #act_x = self.apply_threshold(act_x, 'act_fn')
        print(f"act_fn output after threshold: {torch.sum(act_x == 0)} zeros")
        
        # Apply down_proj and optionally apply threshold to its output
        down_x = self.down_proj(act_x)
        print(f"down_proj output before threshold: {torch.sum(down_x == 0)} zeros")
        down_thresholded = self.apply_threshold(down_x, 'down_proj')
        # down_labels, down_clusters = cluster(x=down_thresholded, centroids=self.down_centroid)
        # label_collector['down'].append(down_labels.cpu())
        down_x = down_thresholded
        # activation_states = prepare_activation_states(predicted_down_proj, self.layer_index)
        # down_x = self.simulate_prefetching(down_x, down_clusters)

        #down_proj_top_50_indices = torch.topk(down_x, 2048, dim=2, largest=True).indices
        
        # down_proj_binary_mask = (down_x != 0).int()

        print(f"down_proj output after threshold: {torch.sum(down_x == 0)} zeros")
        print(f"output of down_states: {down_x.shape}")


        # layer_indices = {
        # "gate_proj": gate_proj_binary_mask.cpu().to(torch.uint8),
        # "up_proj": up_proj_binary_mask.cpu().to(torch.uint8),
        # "down_proj": down_proj_binary_mask.cpu().to(torch.uint8)
        # }
        # mlp_active_neuron_indices.append(layer_indices)
        # if self.layer_index == 31:
        #     #target_mlp_indices_list.append(mlp_active_neuron_indices)
        #     chunk_filename = os.path.join(output_dir, f"chunk_{count}.pt")
        #     torch.save(mlp_active_neuron_indices, chunk_filename)            
        #     count = count +1
        #     print(f"Saved {chunk_filename}")

        return down_x
    
def cluster(x, centroids):
    # Ensure centroids are immutable by cloning at the beginning
    original_centroids = centroids.clone()  # Clone to keep a copy of the original centroids
    # print(f"Zeros in the original centroids: {torch.sum(original_centroids == 0).item()} zeros")
    device = 'cuda:1'
    # Convert x to binary and ensure it is on the correct device
    x = (x != 0).float()
    x = x.squeeze(0)  # Remove the batch dimension
    centroids = original_centroids.to(device)  # Ensure centroids are on the same device as x
    # centroids = original_centroids
    x = x.to(device)
    # print(f"x shape: {x.shape}, centroids shape: {centroids.shape}")
    print(f"x device: {x.device}, centroids device: {centroids.device}")
    # Compute distances
    distances = torch.zeros((x.size(0), centroids.size(0)), device=x.device)
    for i in range(centroids.size(0)):
        diff = x - centroids[i]  # Subtract the centroid
        distances[:, i] = torch.sum(diff.abs() * (x == 1), dim=1)  # Focus on 1's differences

    # print(f"Distance matrix shape: {distances.shape}")

    # Assign datapoints to the closest centroid
    assignments = torch.argmin(distances, dim=1)  # Shape: [2024]
    assignments = assignments.to(original_centroids.device)
    # print(f"Assignments shape: {assignments.shape}")

    # Gather the centroids corresponding to the assignments
    assigned_centroids = original_centroids[assignments].clone()  # Use the original centroids

    # Add the batch dimension back to match the input shape
    assigned_centroids = assigned_centroids.unsqueeze(0).to('cuda:0')  # Shape: [1, 2024, 14336]
    # print(f"Zeros in the assigned centroids: {torch.sum(assigned_centroids == 0).item()} zeros")

    # Verify that original centroids remain unchanged
    # print(f"Original centroids checksum: {torch.sum(original_centroids).item()}")

    return assignments, assigned_centroids

def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)



    # ####################################### APPLYRING THRESHOLD #########################################
    #Load the thresholds from the JSON file
    with open('./Mistral-7B/thresholds_50_percent_sparsity.json', 'r') as file:
        thresholds = json.load(file)

    # Replace MLPs in the model layers with ThresholdLlamaMLP instances
    for layer_index, layer in enumerate(model.model.layers):
        original_mlp = layer.mlp
        threshold_mlp = ThresholdMistralMLP(original_mlp, thresholds, layer_index)
        layer.mlp = threshold_mlp

    if args.tasks is not None:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        if args.tasks == "wikitext":
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seqlen = 2048
            testenc = testenc.input_ids.to(model.device)
            nsamples = 1
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                for token_index in range(model.seq_len):
                    input_token = batch[:, token_index : token_index + 1]
                    print(f"Token {token_index}: input_token dtype: {input_token.dtype}")

                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

            results = {"ppl": ppl.item()}
            if args.output_path is not None:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)
        else:
            task_names = args.tasks.split(",")

            lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=args.num_fewshot,
            )

            print(evaluator.make_table(results))

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
