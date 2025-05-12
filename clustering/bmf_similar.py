import torch
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

def randomized_clustering_bmf(G, k, max_iter=100, seed=42, device='cpu'):
    """
    Randomized Clustering Algorithm for Binary Matrix Factorization (BMF) with GPU support.

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

    # Move the data to the specified device (GPU or CPU)
    G = G.to(device)

    num_tokens, num_features = G.shape

    # Step 1: Initialize the first center as the origin (all zeros)
    centroids = [torch.zeros(num_features, dtype=torch.int32, device=device)]

    # Step 2: Select k-1 centers using probabilistic selection based on l1 distance
    for _ in range(k - 1):
        with torch.no_grad():
            D = torch.tensor([min(torch.sum(torch.abs(token - c)).item() for c in centroids) for token in G], device=device)

        probabilities = D / torch.sum(D)
        new_centroid_index = torch.multinomial(probabilities, 1).item()
        centroids.append(G[new_centroid_index])

    centroids = torch.stack(centroids)

    # Step 3: Perform clustering iteratively
    assignments = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    for iteration in range(max_iter):
        with torch.no_grad():
            new_assignments = torch.tensor([torch.argmin(torch.tensor([torch.sum(torch.abs(token - c)) for c in centroids])).item() for token in G], device=device)

        if torch.equal(assignments, new_assignments):
            print(f"Converged after {iteration} iterations.")
            break

        assignments = new_assignments

        for i in range(k):
            cluster_points = G[assignments == i]
            if cluster_points.size(0) > 0:
                centroids[i] = (torch.round(torch.mean(cluster_points.float(), dim=0))).to(torch.int32)

    clustering_error = torch.sum(torch.tensor([torch.sum(torch.abs(G[i] - centroids[assignments[i]])) for i in range(num_tokens)]))

    return centroids, assignments, clustering_error.item()

# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the chunk file
    chunk = torch.load('chunk_0.pt', map_location='cpu')

    # Number of clusters for each projection type
    k_gate = 128
    k_up = 128
    k_down = 64

    # Collect data from all 32 layers for each projection type
    gate_proj_data = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0)
    up_proj_data = torch.cat([chunk[i]["up_proj"].squeeze(0) for i in range(32)], dim=0)
    down_proj_data = torch.cat([chunk[i]["down_proj"].squeeze(0) for i in range(32)], dim=0)

    # Run clustering for gate_proj
    print("Clustering gate_proj...")
    centroids_gate, assignments_gate, error_gate = randomized_clustering_bmf(gate_proj_data, k_gate, device=device)
    torch.save(centroids_gate, "centroids_gate.pt")
    torch.save(assignments_gate, "assignments_gate.pt")
    torch.save(torch.tensor([error_gate]), "clustering_error_gate.pt")
    print(f"Gate_proj clustering error: {error_gate}")

    # Run clustering for up_proj
    print("Clustering up_proj...")
    centroids_up, assignments_up, error_up = randomized_clustering_bmf(up_proj_data, k_up, device=device)
    torch.save(centroids_up, "centroids_up.pt")
    torch.save(assignments_up, "assignments_up.pt")
    torch.save(torch.tensor([error_up]), "clustering_error_up.pt")
    print(f"Up_proj clustering error: {error_up}")

    # Run clustering for down_proj
    print("Clustering down_proj...")
    centroids_down, assignments_down, error_down = randomized_clustering_bmf(down_proj_data, k_down, device=device)
    torch.save(centroids_down, "centroids_down.pt")
    torch.save(assignments_down, "assignments_down.pt")
    torch.save(torch.tensor([error_down]), "clustering_error_down.pt")
    print(f"Down_proj clustering error: {error_down}")

    print("\nAll clustering tasks completed. Results saved to files.")