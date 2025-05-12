import torch
import random
import os

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def brb_kmeans(G, k, max_iter=100, seed=42, device='cuda'):
    """
    BRB-KMeans Clustering Algorithm for Binary Data with GPU support.

    Parameters:
    - G: Binary matrix of shape [num_tokens, num_features] (PyTorch Tensor)
    - k: Number of clusters (centroids)
    - max_iter: Maximum number of iterations for convergence
    - seed: Random seed for reproducibility
    - device: Device to run the computation ('cuda' for GPU, 'cpu' otherwise)

    Returns:
    - centroids: Tensor of k binary centroids of shape [k, num_features]
    - assignments: Tensor of cluster assignments for each token
    - clustering_error: Final clustering error (scalar)
    """

    torch.manual_seed(seed)
    random.seed(seed)

    # Move the data to the specified device and convert to float32
    G = G.to(torch.float32).to(device)
    num_tokens, num_features = G.shape

    # Step 1: Binary to Real Transformation
    real_data = G.clone()  # Binary 0/1 values become real 0.0/1.0

    # Step 2: Initialize k centroids randomly from the data
    indices = torch.randperm(num_tokens)[:k]
    centroids = real_data[indices].clone()  # [k, num_features]

    # Step 3: Perform k-means clustering iteratively
    assignments = torch.zeros(num_tokens, dtype=torch.int64, device=device)

    batch_size = 16384  # Adjust based on available GPU memory

    for iteration in range(max_iter):
        # Step 3a: Compute distances in batches and assign clusters
        with torch.no_grad():
            new_assignments = torch.empty(num_tokens, dtype=torch.int64, device=device)
            for i in range(0, num_tokens, batch_size):
                batch = real_data[i : i + batch_size]
                distances = torch.cdist(batch, centroids, p=2)
                new_assignments[i : i + batch_size] = torch.argmin(distances, dim=1)

        # Check for convergence
        if torch.equal(assignments, new_assignments):
            print(f"Converged after {iteration} iterations.")
            break

        assignments = new_assignments
        print(f"Iteration {iteration + 1}/{max_iter} completed.")

        # Step 3b: Update centroids (mean and round to binary values)
        for i in range(k):
            cluster_points = real_data[assignments == i]
            if cluster_points.size(0) > 0:
                centroids[i] = torch.round(torch.mean(cluster_points, dim=0))

    # Step 4: Compute final clustering error in batches
    with torch.no_grad():
        clustering_error = 0.0
        for i in range(0, num_tokens, batch_size):
            batch = real_data[i : i + batch_size]
            batch_assignments = assignments[i : i + batch_size]
            batch_centroids = centroids[batch_assignments]
            clustering_error += torch.sum((batch - batch_centroids) ** 2).item()

    print("Clustering completed successfully!")
    return centroids, assignments, clustering_error

# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the chunk file
    chunk = torch.load('../llm-awq-old/full_lenght_byte_dataset_chunks_50/chunk_0.pt', map_location='cuda')

    # Number of clusters for each projection type
    k_gate = 16384
    k_up = 16384
    k_down = 16384

    # Process Gate Projections
    gate_proj_data = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    print("Clustering gate_proj...")
    centroids_gate, assignments_gate, error_gate = brb_kmeans(gate_proj_data, k_gate, device=device)
    torch.save(centroids_gate, "2048/centroids_gate_16384.pt")
    torch.save(assignments_gate, "2048/assignments_gate_16384.pt")
    torch.save(torch.tensor([error_gate]), "2048/clustering_error_gate_16384.pt")
    print(f"Gate_proj clustering error: {error_gate}")
    del gate_proj_data, centroids_gate, assignments_gate, error_gate
    torch.cuda.empty_cache()

    # Process Up Projections
    up_proj_data = torch.cat([chunk[i]["up_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    print("Clustering up_proj...")
    centroids_up, assignments_up, error_up = brb_kmeans(up_proj_data, k_up, device=device)
    torch.save(centroids_up, "2048/centroids_up_16384.pt")
    torch.save(assignments_up, "2048/assignments_up_16384.pt")
    torch.save(torch.tensor([error_up]), "2048/clustering_error_up_16384.pt")
    print(f"Up_proj clustering error: {error_up}")
    del up_proj_data, centroids_up, assignments_up, error_up
    torch.cuda.empty_cache()

    # Process Down Projections
    down_proj_data = torch.cat([chunk[i]["down_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    print("Clustering down_proj...")
    centroids_down, assignments_down, error_down = brb_kmeans(down_proj_data, k_down, device=device)
    torch.save(centroids_down, "2048/centroids_down_16384.pt")
    torch.save(assignments_down, "2048/assignments_down_16384.pt")
    torch.save(torch.tensor([error_down]), "2048/clustering_error_down_16384.pt")
    print(f"Down_proj clustering error: {error_down}")
    del down_proj_data, centroids_down, assignments_down, error_down
    torch.cuda.empty_cache()

    print("\nAll clustering tasks completed. Results saved to files.")