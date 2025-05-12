import torch
import json
import os

def compute_feature_errors(data, centroids, assignments, device='cuda'):
    """
    Compute feature-wise clustering errors.

    Parameters:
    - data: Original binary data of shape [num_points, num_features]
    - centroids: Cluster centroids of shape [num_clusters, num_features]
    - assignments: Cluster assignments for each data point (indices of centroids)
    - device: Device to run the computation ('cuda' for GPU, 'cpu' otherwise)

    Returns:
    - feature_errors: A tensor of shape [num_features] with per-feature error contributions
    """
    data = data.to(device)
    centroids = centroids.to(device)
    assignments = assignments.to(device)

    feature_errors = torch.zeros(data.shape[1], device=device)  # Shape: [num_features]
    for i in range(data.shape[0]):  # Iterate over all data points
        assigned_centroid = centroids[assignments[i]]  # Get the centroid for the current data point
        feature_errors += torch.abs(data[i] - assigned_centroid)  # Accumulate feature-wise errors

    return feature_errors

def save_feature_errors_to_json(feature_errors, output_path, top_k=10):
    """
    Save feature-wise errors and top indices to a JSON file.

    Parameters:
    - feature_errors: Tensor of shape [num_features] with per-feature error contributions
    - output_path: Path to save the JSON file
    - top_k: Number of top features to include
    """
    feature_errors = feature_errors.cpu()
    top_errors = torch.topk(feature_errors, k=top_k)

    result = {
        "top_indices": top_errors.indices.tolist(),
        "top_values": top_errors.values.tolist(),
        "all_errors": feature_errors.tolist()
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Feature errors saved to {output_path}")


# Example Usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the chunk file
    chunk = torch.load('full_lenght_byte_dataset_chunks/chunk_0.pt', map_location='cuda')
    # Load data, centroids, and assignments for gate_proj
    gate_proj_data = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    centroids_gate = torch.load("2048/centroids_gate_512.pt", map_location=device)
    assignments_gate = torch.load("2048/assignments_gate_512.pt", map_location=device)

    # Compute feature-wise errors
    gate_feature_errors = compute_feature_errors(gate_proj_data, centroids_gate, assignments_gate, device=device)

    # Save feature-wise errors to JSON
    save_feature_errors_to_json(gate_feature_errors, "2048/gate_feature_errors.json")

    # Repeat for up_proj
    up_proj_data = torch.cat([chunk[i]["up_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    centroids_up = torch.load("2048/centroids_up_512.pt", map_location=device)
    assignments_up = torch.load("2048/assignments_up_512.pt", map_location=device)

    up_feature_errors = compute_feature_errors(up_proj_data, centroids_up, assignments_up, device=device)
    save_feature_errors_to_json(up_feature_errors, "2048/up_feature_errors.json")

    # Repeat for down_proj
    down_proj_data = torch.cat([chunk[i]["down_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
    centroids_down = torch.load("2048/centroids_down_256.pt", map_location=device)
    assignments_down = torch.load("2048/assignments_down_256.pt", map_location=device)

    down_feature_errors = compute_feature_errors(down_proj_data, centroids_down, assignments_down, device=device)
    save_feature_errors_to_json(down_feature_errors, "2048/down_feature_errors.json")
