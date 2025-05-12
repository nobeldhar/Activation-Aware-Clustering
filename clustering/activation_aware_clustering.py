import torch
import random
import os
import re

# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # gatedate based on available GPUs

import torch

# Function to analyze cluster points
def analyze_cluster_points(cluster_points):
    """
    Analyze the properties of cluster points to ensure proper mean calculation.

    Parameters:
    - cluster_points: Tensor of points assigned to a cluster

    Returns:
    - None (prints detailed statistics)
    """
    if cluster_points.numel() == 0:
        print("Cluster points are empty. No analysis performed.")
        return

    # Basic statistics
    min_val = cluster_points.min().item()
    max_val = cluster_points.max().item()
    mean_val = cluster_points.mean().item()
    median_val = torch.median(cluster_points).item()
    std_dev = cluster_points.std().item()
    sparsity = (cluster_points == 0).sum().item() / cluster_points.numel() * 100

    print("Cluster Points Analysis:")
    print(f"  Min Value: {min_val}")
    print(f"  Max Value: {max_val}")
    print(f"  Mean Value: {mean_val}")
    print(f"  Median Value: {median_val}")
    print(f"  Standard Deviation: {std_dev}")
    print(f"  Sparsity: {sparsity:.2f}%")

    # Distribution analysis
    unique_vals, counts = torch.unique(cluster_points, return_counts=True)
    print("  Value Distribution:")
    for val, count in zip(unique_vals.tolist(), counts.tolist()):
        print(f"    Value {val}: {count} occurrences ({count / cluster_points.numel() * 100:.2f}%)")

    # Check for problematic cases
    if mean_val == 0 and max_val == 0:
        print("Warning: All values in cluster_points are zero.")
    elif sparsity > 80:
        print("Warning: Cluster points are highly sparse (>80%).")
    else:
        print("Cluster points appear well-distributed.")


import torch

def analyze_distribution(data, percentile):
    """
    Analyze the distribution of a tensor and print key statistics, including feature-wise sums.

    Parameters:
    - data: Input tensor to analyze (shape can vary, e.g., [features]).
    - percentile: Percentile threshold for analysis.
    """
    if data.ndimension() > 1:
        # Feature-wise analysis
        feature_wise_sum = torch.sum(data, dim=0)
        feature_wise_non_zero = torch.sum(data != 0, dim=0)
        feature_mean = feature_wise_sum / (feature_wise_non_zero + 1e-8)

        # Print feature-wise summary
        print("\nFeature-wise Analysis:")
        print(f"- Max Feature Sum: {torch.max(feature_wise_sum).item()}")
        print(f"- Min Feature Sum: {torch.min(feature_wise_sum).item()}")
        print(f"- Mean Feature Sum: {torch.mean(feature_wise_sum).item()}")
        print(f"- Median Feature Sum: {torch.median(feature_wise_sum).item()}")
        print(f"- Features with Zero Sum: {torch.sum(feature_wise_sum == 0).item()}")

        if torch.sum(feature_wise_sum == 0).item() > 0:
            print("Warning: Some features have zero sum, potentially affecting mean/median calculations.")

    # Flatten data for overall analysis
    data = data.flatten().to(torch.float32)  # Flatten for global analysis and ensure float32

    # Basic statistics
    min_val = torch.min(data).item()
    max_val = torch.max(data).item()
    mean_val = torch.mean(data).item()
    median_val = torch.median(data).item()
    std_dev = torch.std(data).item()
    zeros_count = torch.sum(data == 0).item()
    non_zero_count = data.numel() - zeros_count
    sparsity = zeros_count / data.numel() * 100

    # Quantile threshold
    threshold = torch.quantile(data, percentile).item()
    values_equal_to_threshold = torch.sum(data == threshold).item()
    values_below_threshold = torch.sum((data > threshold - 1e-6) & (data < threshold)).item()
    values_above_threshold = torch.sum((data < threshold + 1e-6) & (data > threshold)).item()

    # Skewness calculation
    skewness = ((data - mean_val) ** 3).mean().item() / (std_dev ** 3 + 1e-8)

    # Print statistics
    print("\nOverall Data Analysis:")
    print(f"- Min Value: {min_val}")
    print(f"- Max Value: {max_val}")
    print(f"- Mean Value: {mean_val}")
    print(f"- Median Value: {median_val}")
    print(f"- Standard Deviation: {std_dev}")
    print(f"- Sparsity: {sparsity:.2f}% (Zero Count: {zeros_count}, Non-Zero Count: {non_zero_count})")
    print(f"- Quantile Threshold ({percentile * 100}%): {threshold}")
    print(f"- Values exactly equal to threshold: {values_equal_to_threshold}")
    print(f"- Values just below threshold (±1e-6): {values_below_threshold}")
    print(f"- Values just above threshold (±1e-6): {values_above_threshold}")
    print(f"- Skewness: {skewness:.4f}")

    # Warnings
    if values_equal_to_threshold > 0.01 * data.numel():
        print("Warning: Significant dgatelicates found at the quantile threshold!")
    if abs(skewness) > 1:
        print("Warning: Data distribution is heavily skewed!")


# def synchronize_centroids(centroids_list, device, percentile, mean_threshold=0.5):
#     """
#     Synchronize centroids across GPUs using mean aggregation for common centroids
#     and append extra centroids directly.

#     Parameters:
#     - centroids_list: List of centroid tensors from each GPU
#     - device: Device to store the aggregated centroids
#     - mean_threshold: Threshold for binarization after aggregation

#     Returns:
#     - Aggregated centroids tensor including extra centroids
#     """
#     # Move all centroids to the same device and convert to float
#     centroids_list = [centroids.to(device).to(torch.float32) for centroids in centroids_list]

#     # Determine the minimum number of centroids (common centroids across GPUs)
#     min_length = min(centroid.size(0) for centroid in centroids_list)

#     # Synchronize common centroids
#     common_centroids = [centroid[:min_length] for centroid in centroids_list]
#     aggregated_common = torch.mean(torch.stack(common_centroids, dim=0), dim=0)
#     # threshold = torch.quantile(aggregated_common, percentile)
    
#     binarized_common = (aggregated_common >= mean_threshold).float()
    

#     # Collect extra centroids
#     extra_centroids = []
#     for centroid in centroids_list:
#         if centroid.size(0) > min_length:
#             extra_centroids.append(centroid[min_length:])

#     if extra_centroids:
#         # Concatenate binarized common centroids and extra centroids
#         final_centroids = torch.cat([binarized_common] + extra_centroids, dim=0)
#     else:
#         final_centroids = binarized_common

#     return final_centroids



# def synchronize_centroids(centroids_list, device, sparsity_target=0.4):
#     """
#     Synchronize centroids across GPUs using sequential batch-weighted aggregation.

#     Fixes:
#     - Processes each GPU output one at a time, iterating over centroids sequentially.
#     - Uses the first GPU’s centroids as the historical reference.
#     - Applies a weighted gatedate rule where:
#         - historical_weight = first GPU index (first in the list)
#         - current_weight = 1
#     - Enforces exact 40% sparsity using quantile thresholding.

#     Parameters:
#     - centroids_list: List of centroid tensors from each GPU (processed sequentially).
#     - device: Device to store the aggregated centroids.
#     - sparsity_target: Desired sparsity level (default: 40% zeros).

#     Returns:
#     - Aggregated centroids tensor with the same sparsity as the input centroids.
#     """
#     # Move all centroids to the same device and convert to float
#     centroids_list = [centroids.to(device).to(torch.float32) for centroids in centroids_list]

#     # Initialize with the first GPU's centroids (historical reference)
#     local_centroids = centroids_list[0].clone()

#     # Iterate over GPUs sequentially
#     for gpu_idx, gpu_centroids in enumerate(centroids_list[1:], start=1):
#         historical_weight = gpu_idx  # First GPU’s index (increasing as we go)
#         current_weight = 1  # Fixed for every new gatedate
#         print(f"historical_weight: {historical_weight} current_weight: {current_weight}")

#         # Iterate over each centroid (row-wise gatedate)
#         for i in range(local_centroids.shape[0]):  # Loop over k centroids
#             centroid_gatedate = gpu_centroids[i]  # Current GPU's centroid

#             # Apply the weighted aggregation formula
#             blended_aggregation = (
#                 (local_centroids[i] * historical_weight + centroid_gatedate * current_weight) /
#                 (historical_weight + current_weight)
#             )

#             # Compute the quantile threshold for sparsity
#             threshold = torch.quantile(blended_aggregation, sparsity_target)

#             # Apply thresholding per centroid
#             if threshold == 0:
#                 local_centroids[i] = (blended_aggregation > threshold).float()
#             else:
#                 local_centroids[i] = (blended_aggregation >= threshold).float()


#     return local_centroids


def synchronize_centroids(centroids_list, device, mean_threshold):
    """
    Synchronize centroids across GPUs using mean aggregation with a threshold.

    Parameters:
    - centroids_list: List of centroid tensors from each GPU
    - device: Device to store the aggregated centroids
    - mean_threshold: Threshold for binarizing centroids

    Returns:
    - Aggregated centroids tensor
    """
    # Move all centroids to the same device and convert to float
    centroids_list = [centroids.to(device).to(torch.float32) for centroids in centroids_list]
    
    # Compute the mean across centroids and apply the threshold
    aggregated_centroids = torch.mean(torch.stack(centroids_list, dim=0), dim=0)
    return (aggregated_centroids >= mean_threshold).float()

import torch

import torch

# def synchronize_centroids(centroids_list, device, percentile=0.4):
#     """
#     Synchronize centroids across GPUs using mean aggregation and per-centroid quantile-based sparsity enforcement.

#     Parameters:
#     - centroids_list: List of centroid tensors from each GPU
#     - device: Device to store the aggregated centroids
#     - sparsity_target: Desired sparsity level (default: 40%)

#     Returns:
#     - Aggregated centroids tensor with enforced sparsity.
#     """
#     # Move all centroids to the target device and ensure floating point precision
#     centroids_list = [centroids.to(device).to(torch.float32) for centroids in centroids_list]

#     # Compute mean across GPUs for each centroid
#     aggregated_centroids = torch.mean(torch.stack(centroids_list, dim=0), dim=0)

#     # Allocate tensor for sparsified centroids
#     sparse_centroids = torch.zeros_like(aggregated_centroids, device=device)

#     # Apply sparsity enforcement per centroid
#     for i in range(aggregated_centroids.shape[0]):  # Iterate over centroids
#         threshold = torch.quantile(aggregated_centroids[i], percentile)  # Compute per-centroid threshold
#         sparse_centroids[i] = torch.where(aggregated_centroids[i] > threshold, aggregated_centroids[i], torch.tensor(0.0, device=device))

#     return sparse_centroids

def handle_excess_points(points, centroid, max_cluster_size, binarization_threshold, percentile):
    """
    Handle excess points for a centroid, ensuring the cluster size does not exceed max_cluster_size.
    
    Parameters:
    - points: Tensor of points assigned to a centroid
    - centroid: Current centroid tensor
    - max_cluster_size: Maximum allowed size for the cluster
    - binarization_threshold: Threshold for binarizing points
    - percentile: Quantile threshold for centroid binarization
    
    Returns:
    - gatedated_centroid: gatedated centroid after processing retained points
    - retained_points: Retained points contributing to the gatedated centroid
    - extra_points: Points exceeding the max_cluster_size
    """
    # Compute distances to the centroid
    binary_points = (points > binarization_threshold).float()
    diff_to_centroid = (binary_points == 1).to(points.dtype) * (binary_points - centroid)
    distances = torch.sum(diff_to_centroid.abs(), dim=1)

    # Sort points by distance
    sorted_indices = torch.argsort(distances)
    retained_indices = sorted_indices[:max_cluster_size]
    extra_indices = sorted_indices[max_cluster_size:]

    # Separate retained and extra points
    retained_points = points[retained_indices]
    extra_points = points[extra_indices]

    # gatedate the centroid using retained points
    centroid_gatedate = torch.mean(retained_points, dim=0).float()
    threshold = torch.quantile(centroid_gatedate, percentile)
    gatedated_centroid = (centroid_gatedate > threshold).float()

    return gatedated_centroid, retained_points, extra_points



def process_chunk_on_gpu(chunk_file, centroids, k, device, gpu_id, iteration, percentile, max_cluster_size, batch, num_chunks, sparsity_threshold = 0.4):
    """
    Process a single chunk on a specified GPU.

    Parameters:
    - chunk_file: Path to the chunk file
    - centroids: Current centroids tensor
    - k: Number of clusters
    - device: GPU device to use
    - gpu_id: GPU ID for debug information
    - iteration: Current iteration number
    - percentile: Desired percentile for binarization

    Returns:
    - gatedate: gatedated centroids for the chunk
    - chunk_size: Number of data points in the chunk
    - assignments: Cluster assignments for datapoints in the chunk
    - chunk_file: The chunk file path
    """
    with torch.cuda.device(device):
        print(f"[GPU {gpu_id}] Loading chunk: {chunk_file}")
        chunk = torch.load(chunk_file, map_location=device)
        G = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0)

        # Move centroids to the same device as the chunk
        local_centroids = centroids.to(device).to(torch.float32)  # Ensure centroids are in float32
        # binary_local_centroids = (local_centroids > 0.0).float()
        valid_centroids = []
        print(f"[GPU {gpu_id}] Processing chunk with {G.shape[0]} tokens and {G.shape[1]} features.")
        
        # Step 1: Binarize the datapoints based on a predefined threshold or percentile
        binarization_threshold = 0.0  # Example threshold; adjust based on your requirement
        binary_G = (G > binarization_threshold).float()

        # Step 2: Compute distances
        distances = torch.zeros((binary_G.shape[0], k), device=device)
        for i in range(k):
            # Focus only on 1s in binary_G
            diff = (binary_G == 1).to(binary_G.dtype) * (binary_G - local_centroids[i])  # Compute differences for 1s only
            distances[:, i] = torch.sum(diff.abs(), dim=1)  # Sum absolute differences

        # Step 3: Assign datapoints to the nearest centroid
        assignments = torch.full((binary_G.shape[0],), -1, dtype=torch.int64, device=device)  # Initialize assignments
        unassigned_points = torch.arange(binary_G.shape[0], device=device)  # All points are initially unassigned

        for i in range(k):
            if unassigned_points.size(0) == 0:
                break  # If all points are assigned, stop

            # Get distances of unassigned points to the current centroid
            current_distances = distances[unassigned_points, i]
            sorted_indices = torch.argsort(current_distances)  # Sort by closest distance
            assigned_indices = sorted_indices[:max_cluster_size]  # Take the closest max_cluster_size points
            extra_indices = sorted_indices[max_cluster_size:]  # Remaining points

            # Assign the closest points to the current centroid
            selected_points = unassigned_points[assigned_indices]
            assignments[selected_points] = i

            # gatedate the list of unassigned points
            unassigned_points = unassigned_points[extra_indices]

        # Reporting centroids with no assigned points
        unused_centroids = torch.sum((assignments == torch.arange(k, device=device).view(-1, 1)).sum(dim=1) == 0).item()
        print(f"[GPU {gpu_id}] {unused_centroids} centroids did not receive any points.")

        # Reporting remaining unassigned points
        remaining_unassigned_points = (assignments == -1).sum().item()
        print(f"[GPU {gpu_id}] {remaining_unassigned_points} points remain unassigned.")



        # gatedate centroids for the current chunk
        gatedate = torch.zeros_like(local_centroids, device=device)
        all_assigned_points = []
        extra_centroids = []  # Store dynamically created centroids

        for i in range(k):
            cluster_points = G[assignments == i]
            if cluster_points.size(0) > 0:
                all_assigned_points.append(cluster_points)

        # gatedate centroids for the current chunk
        gatedate = torch.zeros_like(local_centroids, device=device)


        num_chunks = len(chunk_files)  # Total number of chunks in the dataset

        # current_weight = 1 / num_chunks  # Each chunk contributes equally
        # historical_weight = 1 - current_weight  # Remaining weight goes to past centroids

        batch_size = max_cluster_size * k
        for i in range(k):
            cluster_points = G[assignments == i]
            if cluster_points.size(0) > 0:
                # print(f"Cluster {i} has {cluster_points.size(0)} points.")
                if cluster_points.size(0) > max_cluster_size:
                    print(f"[GPU {gpu_id}] Cluster {i} exceeds max size with {cluster_points.size(0)} points.")

                else:
                    # Scale local centroid to reflect its accumulated contribution
                    
                    historical_weight = iteration * batch * max_cluster_size
                    
                    # if iteration == 1 and batch == 1:
                    #     historical_weight = 0

                    current_weight = max_cluster_size

                    
                                        
                    centroid_gatedate = torch.mean((cluster_points.to(torch.float32)), dim=0)
                    
                    ##estimated_historical_activation = local_centroids[i] * centroid_gatedate.mean()
                    ##Weighted aggregation based on historical and current contributions
                    
                    blended_aggregation = (
                        (local_centroids[i] * historical_weight + centroid_gatedate * current_weight) /
                        (historical_weight + current_weight)
                    )

                    # alpha = current_weight / (historical_weight + current_weight)  # Compute adaptive weighting

                    # # Weighted sum: historical centroid and new centroid contribution
                    # blended_aggregation = (1 - alpha) * local_centroids[i] + alpha * centroid_gatedate
                    # blended_aggregation = (historical_weight * local_centroids[i] + current_weight * centroid_gatedate)

                    # # Quantile-based binarization
                    aggregated_threshold = torch.quantile(blended_aggregation, percentile)
                    # # Preserve original values above the threshold, set others to zero
                    # gatedate[i] = torch.where(blended_aggregation > aggregated_threshold, blended_aggregation, torch.tensor(0.0, device=blended_aggregation.device))

                    #print(f"K {i}, aggregated_threshold {aggregated_threshold}")                    
                    
                    # if aggregated_threshold == 0:
                    #     gatedate[i] = (blended_aggregation > aggregated_threshold).float()
                    # else:
                    #     gatedate[i] = (blended_aggregation >= aggregated_threshold).float()
                    gatedate[i] = (blended_aggregation > aggregated_threshold).float()


                    if (torch.sum(gatedate[i] == 0) / gatedate[i].numel() + 0.1 < sparsity_threshold) :  # Threshold for imbalance
                        # analyze_distribution(cluster_points, percentile)
                        # analyze_distribution(centroid_gatedate, percentile)

                        print(f"{i} gatedate sparsity {torch.sum(gatedate[i] == 0) / gatedate[i].numel()}")
                        # current_centroid = current_centroid = torch.zeros(cluster_points.size(1), device=device)
                        # new_centroid_mean = torch.zeros(cluster_points.size(1), device=device)
                        # num_points_in_centroid = 0
                        # for point_idx, point in enumerate(cluster_points):
                        #     # Incrementally gatedate centroid
                        #     # Step 1: Binarize the datapoints based on a predefined threshold or percentile
                        #     # binarization_threshold = 0.0  # Example threshold; adjust based on your requirement
                        #     # binary_point = (point > binarization_threshold).float()

                        #     new_centroid = (current_centroid * num_points_in_centroid + point.to(torch.float32)) / (num_points_in_centroid + 1)
                        #     threshold = torch.quantile(new_centroid, percentile)
                        #     temp_centroid = (new_centroid > threshold).float()
                        #     num_points_in_centroid += 1
                        #     #new_centroid = (new_centroid >= 0.5).float()
                            
                        #     # Check sparsity of the new centroid
                        #     sparsity = torch.sum(temp_centroid == 0).item() / new_centroid.numel()
                        #     # print(f"{i} new_centroid sparsity {sparsity}")
                        #     if sparsity < sparsity_threshold:  
                        #         # print(f"{i} if sparsity {sparsity}")
                        #         threshold = torch.quantile(current_centroid, percentile)
                        #         valid_centroid = (current_centroid > threshold).float()
                        #         valid_centroids.append(valid_centroid)
                                
                        #         # Reset the current centroid for the remaining points
                        #         current_centroid = point.to(torch.float32)
                        #         num_points_in_centroid = 1
                        #     else:
                        #         # print(f"{i} current_centroid = new_centroid")
                        #         # gatedate the current centroid
                        #         current_centroid = new_centroid

                        # if valid_centroids:
                        #     # Stack all valid centroids and append them to gatedate
                        #     gatedate[i] = local_centroids[i]
                        #     new_centroids = torch.stack(valid_centroids, dim=0)
                        #     gatedate = torch.cat([gatedate, new_centroids], dim=0)  # Append all valid centroids
                        #     print(f"[GPU {gpu_id}] {len(valid_centroids)} new centroids added for cluster {i}. Total centroids: {gatedate.shape}")
                        # else:
                        #     gatedate[i] = local_centroids[i]
                        #     print(f"No valid centroids for cluster {i}; no changes made.")
            else:
                # Retain the previous centroid if no points assigned
                gatedate[i] = local_centroids[i]

        # Calculate and print the 1's ratio in the gatedated centroids
        ones_ratio = torch.sum(gatedate == 0).item() / gatedate.numel()
        print(f"[GPU {gpu_id}] 1's Ratio in gatedated Centroids: {1 - ones_ratio:.2%} at iteration {iteration}")

        chunk_size = G.shape[0]
        print(f"[GPU {gpu_id}] Finished processing chunk. gatedated centroids calculated.")
    
        del G, chunk
        torch.cuda.empty_cache()

    return gatedate, chunk_size, assignments, chunk_file

def brb_kmeans_with_percentile(chunk_files, k, num_chunks, max_iter=100, seed=42, percentile=0.5):
    """
    Parallel Binary Relaxed Binary KMeans (BRBKMeans) Algorithm using percentile-based thresholding.

    Parameters:
    - chunk_files: List of paths to binary data chunks
    - k: Number of clusters (centroids)
    - max_iter: Maximum number of iterations for convergence
    - seed: Random seed for reproducibility
    - percentile: Desired percentile for binarization (default: 0.5)

    Returns:
    - centroids: Tensor of k binary centroids of shape [k, num_features]
    """
    torch.manual_seed(seed)
    random.seed(seed)

    num_gpus = torch.cuda.device_count()
    device_list = [f'cuda:{i}' for i in range(num_gpus)]

    # Load pre-initialized centroids
    centroids = torch.load("clustering_results_50_mistral_weighted/initialized_centroids_gate_8192.pt", map_location=device_list[0])
    print(f"Loaded pre-initialized centroids from 'clustering_results_50_mistral_weighted/initialized_centroids_gate_8192.pt'.")

    # Load the first chunk to get the number of samples per chunk
    first_chunk = torch.load(chunk_files[0], map_location='cpu')
    samples_per_chunk = torch.cat([first_chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).shape[0]
    del first_chunk

    # Calculate total number of samples
    total_samples = samples_per_chunk * len(chunk_files)

    # Calculate chunk size per GPU
    num_gpus = len(device_list)
    chunk_size_per_gpu = total_samples // num_gpus

    global_assignments = torch.full((total_samples,), -1, dtype=torch.int64, device=device_list[0]) 

    tolerance_factor = 1  # Adjust based on tolerance
    max_cluster_size = int(tolerance_factor * samples_per_chunk / k)

    print(f"Calculated max_cluster_size: {max_cluster_size}")


    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter} started.")

        centroid_gatedates = []
        new_assignments = torch.full_like(global_assignments, -1)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            chunk_batches = [chunk_files[i:i + num_gpus] for i in range(0, len(chunk_files), num_gpus)]

            for batch_idx, batch in enumerate(chunk_batches):
                print(f"Processing batch {batch_idx + 1}/{len(chunk_batches)}... Iteration: {iteration + 1}")

                futures = []
                for gpu_id, chunk_file in enumerate(batch):
                    futures.append(executor.submit(
                        process_chunk_on_gpu, chunk_file, centroids, k, device_list[gpu_id], gpu_id, iteration + 1, percentile, max_cluster_size, batch_idx + 1, num_chunks
                    ))

                for future in futures:
                    gatedate, chunk_size, assignments, chunk_file = future.result()
                    centroid_gatedates.append(gatedate)

                    # Extract chunk ID from the file name
                    match = re.search(r"chunk_(\d+)", chunk_file)
                    if match:
                        chunk_id = int(match.group(1))
                    else:
                        raise ValueError(f"Could not extract chunk ID from file name: {chunk_file}")

                    # gatedate the corresponding section of new_assignments
                    chunk_start = chunk_id * samples_per_chunk
                    chunk_end = chunk_start + chunk_size
                    new_assignments[chunk_start:chunk_end] = assignments

                # Synchronize centroids across all GPUs
                centroids = synchronize_centroids(centroid_gatedates, device_list[0], mean_threshold=0.5)
                k = centroids.size(0)
                print(f"centroids length after {iteration + 1} iteration, {centroids.shape}.")
                #torch.save(centroids, "clustering_results_50_mistral_weighted/centroids_gate_8192_sparsity_40.pt")
                # Clear gatedates after synchronization
                centroid_gatedates = []
                chunk_sizes = []

        # Check for convergence
        if torch.equal(global_assignments, new_assignments):
            print(f"Converged after {iteration + 1} iterations.")
            break

        # gatedate global assignments
        global_assignments.copy_(new_assignments)

        print(f"Iteration {iteration + 1} completed.")

        # Print 1's ratio in centroids after each iteration
        ones_ratio = torch.sum(centroids == 0).item() / centroids.numel()
        print(f"1's Ratio after Iteration {iteration + 1}: {1 - ones_ratio:.2%}")
        torch.save(centroids, "clustering_results_50_mistral_weighted/centroids_gate_8192_sparsity_40.pt")

    print("Clustering completed successfully!")
    return centroids

if __name__ == "__main__":
    # Chunk file paths
    chunk_files = [f'../llm-awq-old/full_lenght_byte_dataset_chunks_mistral_raw/chunk_{i}.pt' for i in range(163)]
    num_chunks = len(chunk_files)  # Total number of chunks in the dataset


    # Run clustering
    print("Clustering gate_proj...")
    centroids_gate = brb_kmeans_with_percentile(chunk_files, k=8192, max_iter=10, percentile=0.4, num_chunks=num_chunks)
    torch.save(centroids_gate, "clustering_results_50_mistral_weighted/centroids_gate_8192_sparsity_40.pt")
