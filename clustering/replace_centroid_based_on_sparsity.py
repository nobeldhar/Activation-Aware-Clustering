import torch

# Define the sparsity threshold
SPARSITY_THRESHOLD = 0.4

def replace_low_sparsity_centroids(centroids_path, output_path):
    """
    Replace centroids with less than 40% sparsity with the previous or next centroid.

    Parameters:
    - centroids_path: Path to the existing centroids file
    - output_path: Path to save the downdated centroids
    """
    # Load centroids
    centroids = torch.load(centroids_path)
    print("Centroids loaded successfully.")
    count = 0
    for i in range(len(centroids)):
        # Calculate sparsity for the current centroid
        total_features = centroids[i].numel()
        zero_count = torch.sum(centroids[i] == 0).item()
        sparsity = zero_count / total_features

        # Check if sparsity is below the threshold
        if sparsity < SPARSITY_THRESHOLD:
            count += 1
            print(f"Centroid {i} sparsity {sparsity:.2%} is below threshold.")

            # Replace with previous centroid if possible
            if i > 0:
                # centroids[i] = centroids[i - 1].clone()
                print(f"Centroid {i} replaced with centroid {i - 1}.")

            # Otherwise, replace with the next centroid
            elif i < len(centroids) - 1:
                # centroids[i] = centroids[i + 1].clone()
                print(f"Centroid {i} replaced with centroid {i + 1}.")

            else:
                print(f"Centroid {i} cannot be replaced as it has no neighbors.")
        elif sparsity > 0.7:
            print(f"Centroid {i} sparsity {sparsity:.2%} is above threshold.")
    print(f"total {count}")
    # Save the downdated centroids
    #torch.save(centroids, output_path)
    print(f"downdated centroids saved to {output_path}.")

if __name__ == "__main__":
    # Path to the existing centroids file
    centroids_path = "clustering_results_50_mistral_weighted/centroids_gate_2048.pt"
    
    # Path to save the downdated centroids
    output_path = "clustering_results_50_mistral_weighted/centroids_gate_2048.pt"

    # Replace low sparsity centroids
    replace_low_sparsity_centroids(centroids_path, output_path)
