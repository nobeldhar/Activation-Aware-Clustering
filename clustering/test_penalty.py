import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" 
import torch

import torch
import os

def calculate_accuracy_and_ratios(data_path, centroids_path, assignments_path, device='cuda'):
    """
    Calculate 1's accuracy and 0/1 ratios in centroids.

    Parameters:
    - data_path: Path to the original binary data (chunk file)
    - centroids_path: Path to the saved centroids
    - assignments_path: Path to the saved cluster assignments
    - device: Device to perform computation ('cuda' or 'cpu')

    Returns:
    - accuracy: 1's accuracy percentage
    - zero_ratio: Percentage of 0's in centroids
    - one_ratio: Percentage of 1's in centroids
    """
    print("Loading data...")

    # Load data
    chunk = torch.load(data_path, map_location=device)
    data = torch.cat([chunk[i]["down_proj"].squeeze(0) for i in range(32)], dim=0).to(torch.float32).to(device)

    print(f"Data loaded. Shape: {data.shape}")

    # Load centroids and assignments
    centroids = torch.load(centroids_path, map_location=device).to(torch.float32).to(device)
    assignments = torch.load(assignments_path, map_location=device).to(device)

    print("Centroids and assignments loaded.")

    # Calculate assigned centroids for each data point
    assigned_centroids = centroids[assignments]

    print("Assigned centroids computed.")

    

    # Compute accuracy for both 1's and 0's
    total_ones = torch.sum(data == 1)
    correct_ones = torch.sum((data == 1) & (assigned_centroids == 1))

    total_zeros = torch.sum(data == 0)
    correct_zeros = torch.sum((data == 0) & (assigned_centroids == 0))

    # Avoid division by zero
    ones_accuracy = (correct_ones / total_ones).item() * 100 if total_ones > 0 else 100
    zeros_accuracy = (correct_zeros / total_zeros).item() * 100 if total_zeros > 0 else 100

    # Compute overall accuracy
    total_elements = data.numel()
    correct_predictions = correct_ones + correct_zeros
    overall_accuracy = (correct_predictions / total_elements).item() * 100

    # Print results
    print(f"Total 1's: {total_ones.item()}, Correctly represented 1's: {correct_ones.item()}, Accuracy: {ones_accuracy:.2f}%")
    print(f"Total 0's: {total_zeros.item()}, Correctly represented 0's: {correct_zeros.item()}, Accuracy: {zeros_accuracy:.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Compute 1's accuracy
    total_ones = torch.sum(data == 1)
    correct_ones = torch.sum((data == 1) & (assigned_centroids == 1))
    accuracy = (correct_ones / total_ones).item() * 100

    print(f"Total 1's: {total_ones.item()}, Correctly represented 1's: {correct_ones.item()}")

    # Calculate 0/1 ratios in centroids
    total_elements = centroids.numel()
    total_zeros = torch.sum(centroids == 0).item()
    total_ones = torch.sum(centroids == 1).item()

    zero_ratio = (total_zeros / total_elements) * 100
    one_ratio = (total_ones / total_elements) * 100

    print(f"Centroids have {total_zeros} zeros and {total_ones} ones.")

    return accuracy, zero_ratio, one_ratio

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # File paths (downdate as needed)
    data_path = '../llm-awq-old/full_lenght_byte_dataset_chunks_50/chunk_0.pt'
    centroids_path = '2048/centroids_down_16384.pt'
    assignments_path = '2048/assignments_down_16384.pt'

    # Calculate metrics
    accuracy, zero_ratio, one_ratio = calculate_accuracy_and_ratios(
        data_path, centroids_path, assignments_path, device=device
    )

    # Print results
    print(f"1's Accuracy: {accuracy:.2f}%")
    print(f"0's Ratio in Centroids: {zero_ratio:.2f}%")
    print(f"1's Ratio in Centroids: {one_ratio:.2f}%")







