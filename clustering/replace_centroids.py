import torch

# Load the centroid files
initialized_centroids_path = "clustering_results_50/initialized_centroids_up_2048.pt"
centroids_path = "clustering_results_50/centroids_up_2048.pt"

# Load centroids
initialized_centroids = torch.load(initialized_centroids_path)
centroids = torch.load(centroids_path)

# # Check if the dimensions are appropriate
# assert initialized_centroids.size(0) >= 16384, "initialized_centroids does not have enough centroids."
# assert centroids.size(0) >= 32768, "centroids does not have enough centroids."

# Copy last 16384 centroids from initialized_centroids
centroids[1725] = initialized_centroids[1725]
centroids[2045] = initialized_centroids[2044]


# Save the updated centroids
output_path = "clustering_results_50/initialized_centroids_down_2048.pt"
torch.save(centroids, output_path)
print(f"Updated centroids saved to {output_path}")
