import torch
import random
import os

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use one GPU for initialization

def initialize_centroids(chunk_files, k, num_layers=32, device='cuda', sparsity_threshold=0.4, max_increment=100):
    """
    Initialize centroids randomly from all chunks without loading all chunks into memory.

    Parameters:
    - chunk_files: List of paths to binary data chunks
    - k: Number of clusters (centroids)
    - num_layers: Number of layers to process from each chunk
    - device: Device to use for computation ('cuda' or 'cpu')
    - sparsity_threshold: Minimum sparsity required for a centroid to be selected
    - max_increment: Maximum increment to find a sparse centroid if the initial candidate is not sparse enough

    Returns:
    - centroids: Tensor of k randomly selected centroids
    """
    torch.manual_seed(42)  # For reproducibility
    random.seed(42)

    # Step 1: Calculate total number of samples in the dataset
    first_chunk = torch.load(chunk_files[0], map_location=device)
    num_samples_per_chunk = sum(
        first_chunk[i]["up_proj"].squeeze(0).shape[0] for i in range(num_layers)
    )
    total_samples = num_samples_per_chunk * len(chunk_files)
    print(f"Total samples in the dataset: {total_samples}")

    # Step 2: Randomly select k global sample indices
    sampled_indices = sorted(random.sample(range(total_samples), k))
    print(f"Random indices selected: {sampled_indices[:5]} ... {sampled_indices[-5:]}")

    # Step 3: Collect centroids corresponding to sampled indices
    centroids = []
    current_offset = 0

    for chunk_id, chunk_file in enumerate(chunk_files):
        print(f"Processing chunk {chunk_id + 1}/{len(chunk_files)}...")
        chunk = torch.load(chunk_file, map_location=device)
        G = torch.cat([chunk[i]["up_proj"].squeeze(0) for i in range(num_layers)], dim=0).to(device)

        G = (G != 0).float()

        # Determine the sample indices falling within the current chunk
        chunk_start = current_offset
        chunk_end = current_offset + num_samples_per_chunk
        relevant_indices = [
            idx - chunk_start for idx in sampled_indices if chunk_start <= idx < chunk_end
        ]

        # Collect samples corresponding to the relevant indices
        for idx in relevant_indices:
            selected = False
            for increment in range(max_increment + 1):  # Check the next `max_increment` indices
                candidate_idx = idx + increment
                if candidate_idx >= G.size(0):  # Avoid index out of bounds
                    break
                candidate = G[candidate_idx].clone()
                sparsity = torch.sum(candidate == 0).item() / candidate.numel()
                if sparsity >= sparsity_threshold:  # Check sparsity condition
                    centroids.append(candidate)
                    selected = True
                    break
            if not selected:
                print(f"Index {idx} and increments up to {max_increment} failed to meet sparsity threshold.")

        current_offset += num_samples_per_chunk

        del chunk, G
        torch.cuda.empty_cache()

        # Break early if all centroids are collected
        if len(centroids) >= k:
            break

    # Step 4: Stack centroids into a tensor
    centroids = torch.stack(centroids[:k], dim=0).to(device)
    print(f"Centroid initialization completed. {len(centroids)} centroids selected.")

    return centroids


if __name__ == "__main__":
    # Define chunk file paths
    chunk_files = [f'../llm-awq-old/full_lenght_byte_dataset_chunks_50/chunk_{i}.pt' for i in range(163)]

    # Number of centroids
    k = 16384

    # Initialize centroids
    print("Initializing centroids with sparsity checks...")
    centroids = initialize_centroids(chunk_files, k, num_layers=32, device='cuda')

    # Save the initialized centroids
    torch.save(centroids, "clustering_results_50_mistral_weighted/initialized_centroids_up_16384.pt")
    print("Centroids saved to clustering_results_50_mistral_weighted/initialized_centroids_up_16384.pt")