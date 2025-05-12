import torch
import os

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # update based on available GPUs

def compute_sparsity(centroids_path, device='cuda:0'):
    """
    Load centroids and print the sparsity of each centroid and overall sparsity.

    Parameters:
    - centroids_path: Path to the saved centroids.
    - device: Device to run the computation ('cuda:0', 'cpu', etc.).
    
    Returns:
    - None
    """
    # Load the centroids
    centroids = torch.load(centroids_path, map_location=device)
    print("Centroids loaded successfully.")

    total_elements = 0
    total_zeros = 0

    # Iterate over each centroid and compute sparsity
    for i, centroid in enumerate(centroids):
        num_elements = centroid.numel()
        num_zeros = torch.sum(centroid == 0).item()
        sparsity = (num_zeros / num_elements) * 100

        print(f"Centroid {i}: {num_zeros} zeros, Sparsity: {sparsity:.2f}%")

        total_elements += num_elements
        total_zeros += num_zeros

    # Compute overall sparsity
    overall_sparsity = (total_zeros / total_elements) * 100
    print(f"Overall Sparsity: {overall_sparsity:.2f}%")

if __name__ == "__main__":
    # Path to the centroids file
    centroids_path = "clustering_results_50_mistral_weighted/centroids_up_512_sparsity_30.pt"  # update to the actual path

    # Compute sparsity
    compute_sparsity(centroids_path, device='cuda:0')
