import numpy as np
import random
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
def randomized_clustering_bmf(G, k, max_iter=100, seed=42, device='cuda'):
    """
    Randomized Clustering Algorithm for Binary Matrix Factorization (BMF).

    Parameters:
    - G: Binary matrix of shape [num_tokens, num_features]
    - k: Number of clusters (centroids)
    - max_iter: Maximum number of iterations for convergence
    - seed: Random seed for reproducibility

    Returns:
    - U: Centroid matrix of shape [num_tokens, k]
    - centroids: List of k binary centroids
    - assignments: List of cluster assignments for each token
    - clustering_error: Final clustering error
    """

    np.random.seed(seed)
    random.seed(seed)
    G = G.to(device)
    num_tokens, num_features = G.shape

    # Step 1: Initialize the first center as the origin (all zeros)
    centroids = [np.zeros(num_features, dtype=int)]

    # Step 2: Select k-1 centers using probabilistic selection based on l1 distance
    for _ in range(k - 1):
        # Compute l1 distance from each token to the nearest existing centroid
        with torch.no_grad():
            D = np.array([min(np.sum(np.abs(token - c)) for c in centroids) for token in G])

        # Select a new centroid with probability proportional to the l1 distance
        probabilities = D / np.sum(D)
        new_centroid_index = np.random.choice(num_tokens, p=probabilities)
        centroids.append(G[new_centroid_index])

    centroids = np.array(centroids)

    # Step 3: Perform clustering iteratively
    assignments = np.zeros(num_tokens, dtype=int)
    for iteration in range(max_iter):
        # Assign each token to the nearest centroid based on l1 distance
        new_assignments = np.array([np.argmin([np.sum(np.abs(token - c)) for c in centroids]) for token in G])

        # Check for convergence (no change in assignments)
        if np.array_equal(assignments, new_assignments):
            print(f"Converged after {iteration} iterations.")
            break

        assignments = new_assignments

        # Update centroids to the l1 center (binary majority) of each cluster
        for i in range(k):
            cluster_points = G[assignments == i]
            if len(cluster_points) > 0:
                # Update centroid by majority vote
                centroids[i] = np.round(np.mean(cluster_points, axis=0)).astype(int)

    # Compute final clustering error
    clustering_error = sum(np.sum(np.abs(G[i] - centroids[assignments[i]])) for i in range(num_tokens))

    return centroids, assignments, clustering_error

# Example usage
if __name__ == "__main__":
    
    G = torch.load('full_lenght_byte_dataset_chunks/chunk_0.pt', map_location='cuda')  # Example, replace with actual file path
    G = G[0]["gate_proj"].squeeze(0)
    #target_chunk_files = [f'full_lenght_byte_dataset_chunks/chunk_{i}.pt' for i in range(163)]

    # Load or generate a sample binary matrix G of shape [2048, 14336]
    #G = np.random.randint(0, 2, (2048, 14336))  # Replace with actual data if available

    # Set the number of clusters k (e.g., 64)
    k = 64
    
    # Run the clustering algorithm on the GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Run the clustering algorithm
    centroids, assignments, clustering_error = randomized_clustering_bmf(G, k)

    # Print results
    print(f"Final Clustering Error: {clustering_error}")
    print(f"Centroids Shape: {centroids.shape}")

    # Save important values and results
    np.save("centroids.npy", centroids)
    np.save("assignments.npy", assignments)
    np.save("clustering_error.npy", np.array([clustering_error]))

    print("Centroids, assignments, and clustering error saved to files.")