import torch
import random
import os

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def brb_kmeans_with_bmf_initialization(G, k, max_iter=100, seed=42, device='cuda', target_1_ratio=0.5):
    """
    BRB-KMeans Clustering Algorithm with BMF Initialization for All 1's Centroids.

    Parameters:
    - G: Binary matrix of shape [num_tokens, num_features] (PyTorch Tensor)
    - k: Number of clusters (centroids)
    - max_iter: Maximum number of iterations for convergence
    - seed: Random seed for reproducibility
    - device: Device to run the computation ('cuda' for GPU, 'cpu' otherwise)
    - target_1_ratio: Stop when the ratio of 1's in centroids drops below this value.

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

    # Step 1: Initialize the first centroid as all 1's
    centroids = [torch.ones(num_features, dtype=torch.float32, device=device)]

    # Step 2: Select k-1 centers using probabilistic selection based on L1 distance
    for i in range(k - 1):
        with torch.no_grad():
            distances = torch.cdist(G, torch.stack(centroids), p=1).min(dim=1).values
        probabilities = distances / torch.sum(distances)
        new_centroid_index = torch.multinomial(probabilities, 1).item()
        centroids.append(G[new_centroid_index])
        print(f"Centroid {i + 1}/{k} selected.")

    centroids = torch.stack(centroids)
    print(f"Centroids initialized with first centroid as all 1's. Shape: {centroids.shape}")

    # Step 3: Perform clustering iteratively
    assignments = torch.zeros(num_tokens, dtype=torch.int64, device=device)

    for iteration in range(max_iter):
        # Step 3a: Compute distances and assign clusters
        with torch.no_grad():
            distances = torch.cdist(G, centroids, p=2)  # L2 distance
            new_assignments = torch.argmin(distances, dim=1)

        # Check for convergence
        if torch.equal(assignments, new_assignments):
            print(f"Converged after {iteration + 1} iterations.")
            break

        assignments = new_assignments
        print(f"Iteration {iteration + 1}/{max_iter} completed.")

        # Step 3b: Update centroids (mean and round to binary values)
        for i in range(k):
            cluster_points = G[assignments == i]
            if cluster_points.size(0) > 0:
                centroids[i] = torch.round(torch.mean(cluster_points, dim=0))

        # Step 3c: Monitor 1's ratio in centroids and stop if below target
        current_1_ratio = torch.sum(centroids == 1).item() / centroids.numel()
        print(f"Current 1's Ratio: {current_1_ratio:.2%}")
        if current_1_ratio < target_1_ratio:
            print(f"Stopping early: 1's ratio dropped below {target_1_ratio:.2%}")
            break

    # Step 4: Compute final clustering error
    with torch.no_grad():
        clustering_error = 0.0
        for i in range(num_tokens):
            clustering_error += torch.sum((G[i] - centroids[assignments[i]]) ** 2).item()

    print("Clustering completed successfully!")
    return centroids, assignments, clustering_error

# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the chunk file
    chunk = torch.load('chunk_0.pt', map_location=device)

    # Number of clusters for each projection type
    k_gate = 2048
    k_up = 2048
    k_down = 1024

    # Process Gate Projections
    gate_proj_data = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    print("Clustering gate_proj...")
    centroids_gate, assignments_gate, error_gate = brb_kmeans_with_bmf_initialization(
        gate_proj_data, k_gate, device=device, target_1_ratio=0.5
    )
    torch.save(centroids_gate, "2048/centroids_gate_512.pt")
    torch.save(assignments_gate, "2048/assignments_gate_512.pt")
    torch.save(torch.tensor([error_gate]), "2048/clustering_error_gate_512.pt")
    print(f"Gate_proj clustering error: {error_gate}")