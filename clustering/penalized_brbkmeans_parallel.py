import torch
import random
import os

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def brb_kmeans(G, k, max_iter=250, seed=42, device='cuda', threshold=0.15):
    """
    BRB-KMeans Clustering Algorithm for Binary Data with GPU support.

    Parameters:
    - G: Binary matrix of shape [num_tokens, num_features] (PyTorch Tensor)
    - k: Number of clusters (centroids)
    - max_iter: Maximum number of iterations for convergence
    - seed: Random seed for reproducibility
    - device: Device to run the computation ('cuda' for GPU, 'cpu' otherwise)
    - threshold: Custom threshold for rounding during centroid updates

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

    # Initialize centroids randomly from the data
    indices = torch.randperm(num_tokens)[:k]
    centroids = G[indices].clone()

    # Perform k-means clustering iteratively
    assignments = torch.zeros(num_tokens, dtype=torch.int64, device=device)

    for iteration in range(max_iter):
        # Compute distances and assign clusters
        with torch.no_grad():
            distances = torch.cdist(G, centroids, p=1)
            new_assignments = torch.argmin(distances, dim=1)

        # Check for convergence
        if torch.equal(assignments, new_assignments):
            print(f"Converged after {iteration + 1} iterations.")
            break

        assignments = new_assignments

        # Update centroids
        for i in range(k):
            cluster_points = G[assignments == i]
            if cluster_points.size(0) > 0:
                mean_values = torch.mean(cluster_points, dim=0)
                centroids[i] = (mean_values >= threshold).float()

        # Calculate and print the 1's ratio
        ones_ratio = torch.sum(centroids == 1).item() / centroids.numel()
        print(f"Iteration {iteration + 1}/{max_iter}, Current 1's Ratio: {ones_ratio:.2%}")

        # # Stop if 1's ratio drops below 50%
        # if ones_ratio < 0.5:
        #     print("Stopping as 1's ratio dropped below 50%.")
        #     break

    # Compute final clustering error
    with torch.no_grad():
        clustering_error = torch.sum(torch.abs(G - centroids[assignments])).item()

    print("Clustering completed successfully!")
    return centroids, assignments, clustering_error

# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the chunk file
    chunk = torch.load('full_lenght_byte_dataset_chunks/chunk_0.pt', map_location=device)

    # Number of clusters for each projection type
    k_gate = 2048

    # Collect data from all 32 layers for each projection type
    gate_proj_data = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0)

    # Run clustering for gate_proj
    print("Clustering gate_proj...")
    centroids_gate, assignments_gate, error_gate = brb_kmeans(gate_proj_data, k_gate, device=device)
    torch.save(centroids_gate, "penalty/centroids_gate_512.pt")
    torch.save(assignments_gate, "penalty/assignments_gate_512.pt")
    torch.save(torch.tensor([error_gate]), "penalty/clustering_error_gate_512.pt")
    print(f"Gate_proj clustering error: {error_gate}")

    del gate_proj_data, centroids_gate, assignments_gate, error_gate
    torch.cuda.empty_cache()
