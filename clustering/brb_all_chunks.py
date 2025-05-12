import torch
import random
import os
import re
# Set CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"  # gatedate based on available GPUs

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

def process_chunk_on_gpu(chunk_file, centroids, k, mean_threshold, device, gpu_id, iteration):
    """
    Process a single chunk on a specified GPU.

    Parameters:
    - chunk_file: Path to the chunk file
    - centroids: Current centroids tensor
    - k: Number of clusters
    - mean_threshold: Threshold for gatedating centroids
    - device: GPU device to use
    - gpu_id: GPU ID for debug information

    Returns:
    - centroid_gatedate: gatedated centroids for the chunk
    - chunk_size: Number of data points in the chunk
    """

    with torch.cuda.device(device):
        print(f"[GPU {gpu_id}] Loading chunk: {chunk_file}")
        chunk = torch.load(chunk_file, map_location=device)
        G = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0)

        # Move centroids to the same device as the chunk
        local_centroids = centroids.to(device).to(torch.float32)  # Ensure centroids are in float32

        print(f"[GPU {gpu_id}] Processing chunk with {G.shape[0]} tokens and {G.shape[1]} features.")
        distances = torch.zeros((G.shape[0], k), device=device)
        for i in range(k):
            diff = G - local_centroids[i]  # Use local_centroids
            distances[:, i] = torch.sum(diff.abs() * (G == 1).to(diff.dtype), dim=1)  # Focus only on 1's
        assignments = torch.argmin(distances, dim=1)

        # gatedate centroids for the current chunk
        gatedate = torch.zeros_like(local_centroids, device=device)
        for i in range(k):
            cluster_points = G[assignments == i]
            if cluster_points.size(0) > 0:
                centroid_gatedate = torch.mean((cluster_points.to(torch.float32)), dim=0)
                gatedate[i] = (centroid_gatedate >= mean_threshold).float()
                if (torch.sum(gatedate[i] == 1) / gatedate[i].numel() > 0.6) or (torch.sum(gatedate[i] == 0) / gatedate[i].numel() > 0.8) :  # Threshold for imbalance
                    gatedate[i] = local_centroids[i]
            else:
                gatedate[i] = local_centroids[i]  # Retain the previous centroid if no points assigned
        
    
        # Calculate and print the 1's ratio in the gatedated centroids
        ones_ratio = torch.sum(gatedate == 1).item() / gatedate.numel()
        print(f"[GPU {gpu_id}] 1's Ratio in gatedated Centroids: {ones_ratio:.2%} threshold: {mean_threshold}")

        chunk_size = G.shape[0]
        print(f"[GPU {gpu_id}] Finished processing chunk. gatedated centroids calculated.")
    
        del G, chunk
        torch.cuda.empty_cache()

    return gatedate, chunk_size, assignments, chunk_file


def brb_kmeans_with_1s_parallel_chunks(chunk_files, k, max_iter=100, seed=42, mean_threshold=0.0005):
    """
    Parallel Binary Relaxed Binary KMeans (BRBKMeans) Algorithm for clustering binary data.

    Parameters:
    - chunk_files: List of paths to binary data chunks
    - k: Number of clusters (centroids)
    - max_iter: Maximum number of iterations for convergence
    - seed: Random seed for reproducibility
    - mean_threshold: Threshold for gatedating centroids (default: 0.2)

    Returns:
    - centroids: Tensor of k binary centroids of shape [k, num_features]
    - clustering_error: Final clustering error (scalar)
    """
    torch.manual_seed(seed)
    random.seed(seed)

    num_gpus = torch.cuda.device_count()
    device_list = [f'cuda:{i}' for i in range(num_gpus)]

    # Load pre-initialized centroids
    centroids = torch.load("clustering_results_50/initialized_centroids_gate_4096.pt", map_location=device_list[0])
    print(f"Loaded pre-initialized centroids from 'clustering_results/initialized_centroids_gate.pt'.")
    
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

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter} started.")

        centroid_gatedates = []
        chunk_sizes = []
        new_assignments = torch.full_like(global_assignments, -1)  # Placeholder for new assignments

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            chunk_batches = [chunk_files[i:i + num_gpus] for i in range(0, len(chunk_files), num_gpus)]

            for batch_idx, batch in enumerate(chunk_batches):
                print(f"Processing batch {batch_idx + 1}/{len(chunk_batches)}... Iteration: {iteration}")


                # Process the current batch of chunks in parallel
                futures = []
                for gpu_id, chunk_file in enumerate(batch):
                    futures.append(executor.submit(
                        process_chunk_on_gpu, chunk_file, centroids, k, mean_threshold, device_list[gpu_id], gpu_id, iteration + 1
                    ))

                for future in futures:
                    gatedate, chunk_size, assignments, chunk_file = future.result()
                    centroid_gatedates.append(gatedate)
                    chunk_sizes.append(chunk_size)

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

                # Synchronize centroids across all GPUs after processing the batch
                centroids = synchronize_centroids(centroid_gatedates, device_list[0], mean_threshold)
                # Print 1's ratio in centroids after each iteration
                ones_ratio = torch.sum(centroids == 1).item() / centroids.numel()
                print(f"1's Ratio after Batch {batch_idx + 1}: {ones_ratio:.2%}")

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
        ones_ratio = torch.sum(centroids == 1).item() / centroids.numel()
        print(f"1's Ratio after Iteration {iteration + 1}: {ones_ratio:.2%}")

        torch.save(centroids, "clustering_results_50/centroids_gate_4096.pt")

        # # Stop if 1's ratio falls below 50%
        # if ones_ratio < 0.5:
        #     print("Stopping as 1's ratio dropped below 50%.")
        #     break

    # # Compute final clustering error
    # clustering_error = 0.0
    # for chunk_file in chunk_files:
    #     chunk = torch.load(chunk_file, map_location=device_list[0])
    #     G = torch.cat([chunk[i]["gate_proj"].squeeze(0) for i in range(32)], dim=0).to(device_list[0])
    #     distances = torch.zeros((G.shape[0], k), device=device_list[0])
    #     for i in range(k):
    #         diff = G - centroids[i]
    #         distances[:, i] = torch.sum(diff.abs() * (G == 1), dim=1)  # Focus only on 1's
    #     assignments = torch.argmin(distances, dim=1)

    #     for i in range(k):
    #         cluster_points = G[assignments == i]
    #         if cluster_points.size(0) > 0:
    #             clustering_error += torch.sum((cluster_points - centroids[i]) ** 2).item()

    #     del G, chunk
    #     torch.cuda.empty_cache()

    print("Clustering completed successfully!")
    return centroids

if __name__ == "__main__":
    # Chunk file paths
    chunk_files = [f'../llm-awq-old/full_lenght_byte_dataset_chunks_50/chunk_{i}.pt' for i in range(163)]
    # Run clustering
    print("Clustering gate_proj...")
    centroids_gate = brb_kmeans_with_1s_parallel_chunks(chunk_files, k=4096, max_iter=200, mean_threshold=0.5)
    torch.save(centroids_gate, "clustering_results_50/centroids_gate_4096.pt")
