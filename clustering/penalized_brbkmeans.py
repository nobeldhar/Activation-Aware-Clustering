import torch
import random
import os

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def brb_kmeans(G, k, max_iter=100, seed=42, device='cuda', penalty_factor=2.0):
    """
    BRB-KMeans Clustering Algorithm for Binary Data with a Penalty Mechanism and GPU support.

    Parameters:
    - G: Binary matrix of shape [num_tokens, num_features] (PyTorch Tensor)
    - k: Number of clusters (centroids)
    - max_iter: Maximum number of iterations for convergence
    - seed: Random seed for reproducibility
    - device: Device to run the computation ('cuda' for GPU, 'cpu' otherwise)
    - penalty_factor: Penalty for converting 1 → 0 compared to 0 → 1.

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

    print(f"Data loaded on device {device}. Shape: {G.shape}")

    # Step 1: Binary to Real Transformation
    real_data = G.clone()  # Binary 0/1 values become real 0.0/1.0
    print("Step 1: Binary to Real Transformation completed.")

    # Step 2: Initialize k centroids randomly from the data
    indices = torch.randperm(num_tokens)[:k]
    centroids = real_data[indices].clone()  # [k, num_features]
    print(f"Step 2: Centroids initialized. Number of centroids: {k}")

    # Step 3: Perform k-means clustering iteratively
    assignments = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    batch_size = 512  # Adjust based on available GPU memory

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter} started.")

        # Step 3a: Compute distances in batches with penalty
        with torch.no_grad():
            new_assignments = torch.empty(num_tokens, dtype=torch.int64, device=device)
            for i in range(0, num_tokens, batch_size):
                batch = real_data[i : i + batch_size]
                distances = torch.zeros((batch.size(0), k), device=device)
                for j, centroid in enumerate(centroids):
                    diff = batch - centroid  # Difference
                    penalty_matrix = (diff < 0).float() * penalty_factor  # Apply penalty to 1 → 0
                    distances[:, j] = torch.sum((diff.abs() + penalty_matrix), dim=1)
                new_assignments[i : i + batch_size] = torch.argmin(distances, dim=1)

        # Check for convergence
        if torch.equal(assignments, new_assignments):
            print(f"Converged after {iteration + 1} iterations.")
            break

        assignments = new_assignments
        print(f"Iteration {iteration + 1}/{max_iter} completed.")

        # Step 3b: Update centroids (mean and round to binary values)
        for i in range(k):
            cluster_points = real_data[assignments == i]
            if cluster_points.size(0) > 0:
                centroids[i] = torch.round(torch.mean(cluster_points, dim=0))
        print(f"Centroids updated for iteration {iteration + 1}.")

    # Step 4: Compute final clustering error in batches with penalty
    print("Computing final clustering error...")
    with torch.no_grad():
        clustering_error = 0.0
        for i in range(0, num_tokens, batch_size):
            batch = real_data[i : i + batch_size]
            batch_assignments = assignments[i : i + batch_size]
            batch_centroids = centroids[batch_assignments]
            diff = batch - batch_centroids
            penalty_matrix = (diff < 0).float() * penalty_factor
            clustering_error += torch.sum((diff.abs() + penalty_matrix)).item()

    print("Clustering completed successfully!")
    return centroids, assignments, clustering_error

# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the chunk file
    chunk = torch.load('full_lenght_byte_dataset_chunks/chunk_0.pt', map_location='cuda')

    # Number of clusters for each projection type
    k_gate = 2048
    k_up = 2048
    k_down = 1024

    # Process Gate Projections
    gate_proj_data = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    print("Clustering gate_proj...")
    centroids_gate, assignments_gate, error_gate = brb_kmeans(gate_proj_data, k_gate, device=device)
    torch.save(centroids_gate, "penalty/centroids_gate_512.pt")
    torch.save(assignments_gate, "penalty/assignments_gate_512.pt")
    torch.save(torch.tensor([error_gate]), "penalty/clustering_error_gate_512.pt")
    print(f"Gate_proj clustering error: {error_gate}")
    del gate_proj_data, centroids_gate, assignments_gate, error_gate
    torch.cuda.empty_cache()

    # Process Up Projections
    up_proj_data = torch.cat([chunk[i]["up_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    print("Clustering up_proj...")
    centroids_up, assignments_up, error_up = brb_kmeans(up_proj_data, k_up, device=device)
    torch.save(centroids_up, "penalty/centroids_up_512.pt")
    torch.save(assignments_up, "penalty/assignments_up_512.pt")
    torch.save(torch.tensor([error_up]), "penalty/clustering_error_up_512.pt")
    print(f"Up_proj clustering error: {error_up}")
    del up_proj_data, centroids_up, assignments_up, error_up
    torch.cuda.empty_cache()

    # Process Down Projections
    down_proj_data = torch.cat([chunk[i]["down_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    print("Clustering down_proj...")
    centroids_down, assignments_down, error_down = brb_kmeans(down_proj_data, k_down, device=device)
    torch.save(centroids_down, "penalty/centroids_down_256.pt")
    torch.save(assignments_down, "penalty/assignments_down_256.pt")
    torch.save(torch.tensor([error_down]), "penalty/clustering_error_down_256.pt")
    print(f"Down_proj clustering error: {error_down}")
    del down_proj_data, centroids_down, assignments_down, error_down
    torch.cuda.empty_cache()

    print("\nAll clustering tasks completed. Results saved to files.")
