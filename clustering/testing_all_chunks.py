import torch
import os
import re
# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # gatedate based on available GPUs
def test_1s_accuracy(chunk_files, centroids_path, k, device='cuda:0'):
    """
    Test the 1's accuracy of the centroids by loading chunks one by one and computing assignments.

    Parameters:
    - chunk_files: List of paths to the data chunks
    - centroids_path: Path to the saved centroids
    - k: Number of clusters (centroids)
    - device: Device to run the computation ('cuda:0', 'cpu', etc.)
    
    Returns:
    - None
    """
    # Load the centroids
    centroids = torch.load(centroids_path, map_location=device)
    centroids = (centroids > 0.0).float()
    print("Centroids loaded successfully.")

    total_ones = 0
    correct_ones = 0
    
    centroid_counts = torch.zeros(k, dtype=torch.int32, device=device)  # Track assignments per centroid
    for chunk_file in chunk_files:
        print(f"Processing chunk: {chunk_file}")

        # Load the chunk
        chunk = torch.load(chunk_file, map_location=device)
        data = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).to(device)
        data_binary = (data > 0.0).float()

        # Compute assignments
        distances = torch.zeros((data.size(0), k), device=device)
        for i in range(k):
            diff = (data_binary == 1).to(data_binary.dtype) * (data_binary - centroids[i])  # Compute differences for 1s only
            distances[:, i] = torch.sum(diff.abs(), dim=1) 
        assignments = torch.argmin(distances, dim=1)

        print(assignments)

        # Assign data points to centroids
        assigned_centroids = centroids[assignments]

        # # Count assignments for each centroid
        unique, counts = torch.unique(assignments, return_counts=True)
        centroid_counts[unique] += counts  # Accumulate counts
        # Check total assignments
        total_assigned_points = centroid_counts.sum().item()
        total_test_datapoints = data.shape[0]  # Total number of datapoints in the chunk

        # Print verification
        print(f"Total assigned points: {total_assigned_points}, Total test datapoints: {total_test_datapoints}")

        # Ensure they match
        if total_assigned_points == total_test_datapoints:
            print("✅ All datapoints are assigned correctly.")
        else:
            print("⚠️ Mismatch detected! Some datapoints may be assigned to multiple centroids.")
        # datapoint_ones = torch.sum(data == 1, dim=1)  # Count ones per datapoint
        # datapoint_correct_ones = torch.sum((data == 1) & (assigned_centroids == 1), dim=1)  # Correctly represented ones

        # Compute per-datapoint accuracy (Avoid division by zero)
        # datapoint_accuracy = torch.where(datapoint_ones > 0, (datapoint_correct_ones / datapoint_ones) * 100, torch.tensor(0.0, device=device))

        # # Print per-datapoint accuracy
        # for i, acc in enumerate(datapoint_accuracy):
        #     print(f"Datapoint {i}: 1's Accuracy = {acc.item():.2f}%")

        # Compute 1's accuracy
        chunk_total_ones = torch.sum(data_binary == 1)
        chunk_correct_ones = torch.sum((data_binary == 1) & (assigned_centroids == 1))
        total_ones += chunk_total_ones.item()
        correct_ones += chunk_correct_ones.item()
        for i in range(k):
            print(f"Centroid {i}: {centroid_counts[i].item()}")
        
        print(f"Chunk {chunk_file}: accuracty: {(chunk_correct_ones.item() / chunk_total_ones.item()) * 100 }Total 1's: {chunk_total_ones.item()}, Correctly represented 1's: {chunk_correct_ones.item()}")

        

    # Calculate overall 1's accuracy
    accuracy = (correct_ones / total_ones) * 100
    print(f"Overall 1's Accuracy: {accuracy:.2f}%")

    # Calculate 0/1 ratios in centroids
    total_elements = centroids.numel()
    total_zeros = torch.sum(centroids == 0).item()
    total_ones_in_centroids = torch.sum(centroids == 1).item()

    zero_ratio = (total_zeros / total_elements) * 100
    one_ratio = (total_ones_in_centroids / total_elements) * 100

    print(f"Centroids have {total_zeros} zeros and {total_ones_in_centroids} ones.")
    print(f"0's Ratio in Centroids: {zero_ratio:.2f}%, 1's Ratio in Centroids: {one_ratio:.2f}%")

# Example Usage
if __name__ == "__main__":
    chunk_files = [f'../llm-awq-old/full_lenght_byte_dataset_chunks_mistral_raw/chunk_{i}.pt' for i in range(163)]
    centroids_path = "clustering_results_50_mistral_weighted/centroids_gate_16384_sparsity_40.pt"
    k = 16384

    test_1s_accuracy(chunk_files, centroids_path, k, device='cuda:0')