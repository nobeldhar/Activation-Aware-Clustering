import torch
import random
import os

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"  # Update based on available GPUs

def synchronize_centroids(centroids_list, chunk_sizes, device):
    """
    Synchronize centroids across GPUs using weighted mean aggregation.

    Parameters:
    - centroids_list: List of centroid tensors from each GPU
    - chunk_sizes: List of chunk sizes processed by each GPU
    - device: Device to store the aggregated centroids

    Returns:
    - Aggregated centroids tensor
    """
    total_points = sum(chunk_sizes)
    aggregated_centroids = torch.zeros_like(centroids_list[0], device=device)

    for centroids, size in zip(centroids_list, chunk_sizes):
        weight = size / total_points
        aggregated_centroids += centroids.to(torch.float32) * weight

    return (aggregated_centroids >= 0.2).float()

def process_chunk_on_gpu(chunk_file, centroids, k, mean_threshold, device, gpu_id):
    """
    Process a single chunk on a specified GPU.

    Parameters:
    - chunk_file: Path to the chunk file
    - centroids: Current centroids tensor
    - k: Number of clusters
    - mean_threshold: Threshold for updating centroids
    - device: GPU device to use
    - gpu_id: GPU ID for debug information

    Returns:
    - centroid_update: Updated centroids for the chunk
    - chunk_size: Number of data points in the chunk
    """
    with torch.cuda.device(device):
        print(f"[GPU {gpu_id}] Loading chunk: {chunk_file}")
        chunk = torch.load(chunk_file, map_location=device)
        G = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0)

        # Move centroids to the same device as the chunk
        local_centroids = centroids.to(device)

        print(f"[GPU {gpu_id}] Processing chunk with {G.shape[0]} tokens and {G.shape[1]} features.")
        distances = torch.zeros((G.shape[0], k), device=device)
        for i in range(k):
            diff = G - local_centroids[i]  # Use local_centroids
            distances[:, i] = torch.sum(diff.abs() * (G == 1), dim=1)  # Focus only on 1's
        assignments = torch.argmin(distances, dim=1)

        # Update centroids for the current chunk
        update = torch.zeros_like(local_centroids, device=device)
        for i in range(k):
            cluster_points = G[assignments == i]
            if cluster_points.size(0) > 0:
                update[i] = torch.mean(cluster_points.to(torch.float32), dim=0)

        chunk_size = G.shape[0]
        print(f"[GPU {gpu_id}] Finished processing chunk. Updated centroids calculated.")

        del G, chunk
        torch.cuda.empty_cache()

    return update, chunk_size

def brb_kmeans_with_1s_parallel_chunks(chunk_files, k, max_iter=100, seed=42, mean_threshold=0.2):
    """
    Parallel Binary Relaxed Binary KMeans (BRBKMeans) Algorithm for clustering binary data.

    Parameters:
    - chunk_files: List of paths to binary data chunks
    - k: Number of clusters (centroids)
    - max_iter: Maximum number of iterations for convergence
    - seed: Random seed for reproducibility
    - mean_threshold: Threshold for updating centroids (default: 0.2)

    Returns:
    - centroids: Tensor of k binary centroids of shape [k, num_features]
    - clustering_error: Final clustering error (scalar)
    """
    torch.manual_seed(seed)
    random.seed(seed)

    num_gpus = torch.cuda.device_count()
    device_list = [f'cuda:{i}' for i in range(num_gpus)]

    # Initialize centroids randomly from the first chunk
    print(f"Loading initial chunk: {chunk_files[0]}")
    first_chunk = torch.load(chunk_files[0], map_location=device_list[0])
    G = torch.cat([first_chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).to(device_list[0])
    centroids = G[torch.randperm(G.shape[0])[:k]].clone()
    del G, first_chunk
    torch.cuda.empty_cache()

    prev_centroids = None

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter} started.")

        centroid_updates = []
        chunk_sizes = []

        for gpu_id, (device, chunk_file) in enumerate(zip(device_list, chunk_files)):
            update, chunk_size = process_chunk_on_gpu(chunk_file, centroids, k, mean_threshold, device, gpu_id)
            centroid_updates.append(update)
            chunk_sizes.append(chunk_size)

        # Synchronize centroids across all chunks
        new_centroids = synchronize_centroids(centroid_updates, chunk_sizes, device_list[0])

        # Convergence check
        if prev_centroids is not None and torch.allclose(new_centroids, prev_centroids, atol=1e-6):
            print(f"Converged after {iteration + 1} iterations.")
            break

        prev_centroids = new_centroids.clone()
        centroids = new_centroids

        # Print 1's ratio in centroids after each iteration
        ones_ratio = torch.sum(centroids == 1).item() / centroids.numel()
        print(f"1's Ratio after Iteration {iteration + 1}: {ones_ratio:.2%}")

        # Stop if 1's ratio falls below 50%
        # if ones_ratio < 0.5:
        #     print("Stopping as 1's ratio dropped below 50%.")
        #     break

    # Compute final clustering error
    clustering_error = 0.0
    for chunk_file in chunk_files:
        chunk = torch.load(chunk_file, map_location=device_list[0])
        G = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).to(device_list[0])
        distances = torch.zeros((G.shape[0], k), device=device_list[0])
        for i in range(k):
            diff = G - centroids[i]
            distances[:, i] = torch.sum(diff.abs() * (G == 1), dim=1)  # Focus only on 1's
        assignments = torch.argmin(distances, dim=1)

        for i in range(k):
            cluster_points = G[assignments == i]
            if cluster_points.size(0) > 0:
                clustering_error += torch.sum((cluster_points - centroids[i]) ** 2).item()

        del G, chunk
        torch.cuda.empty_cache()

    print("Clustering completed successfully!")
    return centroids, clustering_error

if __name__ == "__main__":
    # Chunk file paths
    chunk_files = [f'../llm-awq-old/full_lenght_byte_dataset_chunks/chunk_{i}.pt' for i in range(163)]

    # Run clustering
    print("Clustering gate_proj...")
    centroids_gate, error_gate = brb_kmeans_with_1s_parallel_chunks(chunk_files, k=2048, max_iter=200, mean_threshold=0.2)
    torch.save(centroids_gate, "clustering_results/centroids_gate.pt")
    torch.save(torch.tensor([error_gate]), "clustering_results/clustering_error_gate.pt")
    print("Gate_proj clustering completed. Error:", error_gate)
