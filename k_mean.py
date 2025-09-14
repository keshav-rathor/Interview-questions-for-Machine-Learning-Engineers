def kmeans(X, k, max_iters=100):
    # Step 1: Initialize cluster centers (pick first k points for simplicity)
    centers = [X[i] for i in range(k)]
    
    for _ in range(max_iters):
        # Step 2: Assignment step
        clusters = [[] for _ in range(k)]   # empty lists for each cluster
        for point in X:
            # compute distance from this point to each center
            distances = []
            for c in centers:
                dist = 0
                for j in range(len(point)):
                    dist += (point[j] - c[j]) ** 2   # squared distance
                distances.append(dist)
            # find the index of the closest center
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)

        # Step 3: Update step
        new_centers = []
        for cluster in clusters:
            if len(cluster) == 0:  # handle empty cluster
                new_centers.append(np.zeros(len(X[0])))
            else:
                # compute mean manually
                mean = []
                for j in range(len(X[0])):  # for each feature
                    s = 0
                    for p in cluster:
                        s += p[j]
                    mean.append(s / len(cluster))
                new_centers.append(mean)

        # Check for convergence
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    
    return centers, clusters


x = [[1, 2],[1, 4],[5, 8],[6, 8]]
centers, clusters = kmeans(x, k=2, max_iters=10)

print("Centers:", centers)
print("Clusters:", clusters)
