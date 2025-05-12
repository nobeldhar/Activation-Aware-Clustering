import torch
import random
import os

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Use one GPU for initialization

def initialize_centroids_kmeanspp_mixed(chunk_files, k, num_layers=32, device='cuda', sparsity_threshold=0.4, max_increment=100):
    """
    Optimized KMeans++ centroid initialization:
    - Distance is computed using **binary activation patterns** for diverse centroid selection.
    - Centroids are **stored with real activation values** for accurate aggregation.

    Parameters:
    - chunk_files: List of paths to data chunks
    - k: Number of clusters (centroids)
    - num_layers: Number of layers to process from each chunk
    - device: Device to use for computation ('cuda' or 'cpu')
    - sparsity_threshold: Minimum sparsity required for a centroid to be selected
    - max_increment: Maximum increment to find a sparse centroid if the initial candidate is not sparse enough

    Returns:
    - centroids_real: Tensor of k centroids stored with **real activation values**.
    """
    torch.manual_seed(42)
    random.seed(42)

    # Step 1: Calculate total number of samples
    first_chunk = torch.load(chunk_files[0], map_location=device)
    num_samples_per_chunk = sum(
        first_chunk[i]["gate_proj"].squeeze(0).shape[0] for i in range(num_layers)
    )
    total_samples = num_samples_per_chunk * len(chunk_files)
    print(f"Total samples in the dataset: {total_samples}")

    # Step 2: Load first chunk and initialize first centroid
    first_chunk = torch.load(chunk_files[0], map_location=device)
    G_real = torch.cat([first_chunk[i]["gate_proj"].squeeze(0) for i in range(num_layers)], dim=0).to(device).to(torch.float32)
    G_binary = (G_real > 0).float()  # Convert to binary representation

    # Select the first centroid based on **maximum L2 distance in binary space**
    l2_norms_binary = torch.norm(G_binary, p=2, dim=1)
    first_idx = torch.argmax(l2_norms_binary).item()
    centroids_real = [G_real[first_idx].clone()]  # Store real values
    centroids_binary = [G_binary[first_idx].clone()]  # Store binary version for distance calc

    print(f"✅ First centroid selected at index {first_idx} with L2 norm {l2_norms_binary[first_idx]:.4f}")

    # Step 3: Select remaining centroids using KMeans++ in Binary Space
    current_offset = 0
    for chunk_id, chunk_file in enumerate(chunk_files):
        print(f"🚀 Processing chunk {chunk_id + 1}/{len(chunk_files)}...")
        chunk = torch.load(chunk_file, map_location=device)
        G_real = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(num_layers)], dim=0).to(device).to(torch.float32)
        G_binary = (G_real > 0).float()  # Binary representation for distance

        while len(centroids_real) < k:
            # 🔥 Compute distances in binary space
            centroids_binary_tensor = torch.stack(centroids_binary).to(device)
            distances = torch.cdist(G_binary, centroids_binary_tensor, p=2).min(dim=1)[0]  # L2 distance in binary

            # Handle NaNs and Infs
            distances = torch.nan_to_num(distances, nan=1e6, posinf=1e6, neginf=1e6)

            # 🔥 Use cumulative probability instead of looping
            probabilities = distances / distances.sum()
            probabilities[probabilities <= 0] = 1e-6  # Ensure valid probabilities
            sampled_idx = torch.multinomial(probabilities, num_samples=1).item()

            # Ensure sparsity constraint
            for increment in range(max_increment + 1):  # Check `max_increment` indices forward
                candidate_idx = sampled_idx + increment
                if candidate_idx >= G_binary.shape[0]:  # Avoid out-of-bounds
                    break
                candidate_binary = G_binary[candidate_idx].clone()
                candidate_real = G_real[candidate_idx].clone()
                sparsity = torch.mean((candidate_binary == 0).to(torch.float32)).item()

                if sparsity >= sparsity_threshold:
                    centroids_real.append(candidate_real)  # Store real activations
                    centroids_binary.append(candidate_binary)  # Store binary pattern
                    break

        current_offset += num_samples_per_chunk
        del chunk, G_real, G_binary
        torch.cuda.empty_cache()

        if len(centroids_real) >= k:
            break  # Stop early if all centroids are selected

    # Step 4: Stack centroids into a tensor
    centroids_real = torch.stack(centroids_real[:k], dim=0).to(device)  # Store real activations
    print(f"✅ Centroid initialization completed. {len(centroids_real)} centroids selected.")

    return centroids_real


if __name__ == "__main__":
    # Define chunk file paths
    chunk_files = [f'../llm-awq-old/full_lenght_byte_dataset_chunks_50/chunk_{i}.pt' for i in range(163)]

    # Number of centroids
    k = 8192

    # Initialize centroids
    print("🚀 Initializing centroids using optimized KMeans++ with Binary Distance & Real Storage...")
    centroids = initialize_centroids_kmeanspp_mixed(chunk_files, k, num_layers=32, device='cuda')

    # Save the initialized centroids
    torch.save(centroids, "clustering_results_50_mistral_weighted/initialized_centroids_gate_8192_kmeanspp.pt")
    print("✅ Centroids saved to clustering_results_50_mistral_weighted/initialized_centroids_gate_8192_kmeanspp.pt")