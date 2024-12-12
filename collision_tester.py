import numpy as np
import matplotlib.pyplot as plt
import math
import os
from generate_test_distributions import (
    generate_far_exact_distribution,
    generate_bimodal_distribution,
    generate_normal_distribution,
    generate_exponential_distribution
)

def collision_tester(p: np.ndarray, m: int, epsilon: float) -> bool:
    """
    Implements the CollisionTester algorithm from Lecture 11.

    Parameters:
    p (np.ndarray): The distribution over [n] from which to draw samples.
    m (int): The number of samples to draw.
    epsilon (float): The error parameter.

    Returns:
    bool: True if the distribution p is accepted, False otherwise.
    """
    
    # Step 1: Take m samples x1, ..., xm drawn i.i.d. from p
    samples = np.random.choice(len(p), size=m, replace=True, p=p)

    # Step 2: Compute Yij = 1{xi = xj} and C = sum_{i < j} Yij / (m choose 2)
    Yij = np.sum([samples[i] == samples[j] for i in range(m) for j in range(i+1, m)])
    C = Yij / math.comb(m, 2)

    # Step 3: Accept if and only if C ≤ 1 + ε^2 / m
    threshold = 1 + epsilon**2 / len(p)
    #print(f"Computed C value: {C}")
    #print(f"Decision threshold: {threshold}")
    return C <= threshold

def test_collision_tester(p: np.ndarray, m: int, epsilon: float, dist_name: str):
    """
    Test the collision tester with the given distribution p, number of samples, and epsilon value.

    Parameters:
    p (np.ndarray): The distribution over [n] from which to draw samples.
    m (int): The number of samples to draw.
    epsilon (float): The error parameter.
    dist_name (str): The name of the distribution.
    """
    # Run the collision tester
    result = collision_tester(p, m, epsilon)

    #print(f"Collision tester result for {dist_name} with {m} samples: {'Accepted' if result else 'Rejected'}")

    # Calculate the frequency of each sample
    samples = np.random.choice(len(p), size=m, replace=True, p=p)
    sample_counts = np.bincount(samples, minlength=len(p)) / m

    # Create results directory if it doesn't exist
    os.makedirs("results/collision_tester", exist_ok=True)
    
    # Save the distribution and sample frequencies plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(p)), p, color='blue', alpha=0.7)
    plt.title(f"Known Distribution {dist_name}")
    plt.xlabel("Outcome")
    plt.ylabel("Probability")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(p)), sample_counts, color='green', alpha=0.7)
    plt.title(f"Bar Plot of Samples for {dist_name} with {m} samples")
    plt.xlabel("Outcome")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(f"results/collision_tester/{dist_name}_{m}_samples.png")
    plt.close()
    
    return result

def main():
    # Generate distributions
    n = 30
    far_exact_dist = generate_far_exact_distribution(n)
    bimodal_dist = generate_bimodal_distribution(n)
    normal_dist = generate_normal_distribution(n)
    exponential_dist = generate_exponential_distribution(n)

    # Normalize the distributions to sum to 1
    far_exact_dist /= np.sum(far_exact_dist)
    bimodal_dist /= np.sum(bimodal_dist)
    normal_dist /= np.sum(normal_dist)
    exponential_dist /= np.sum(exponential_dist)

    distributions = {
        'Far Exact': far_exact_dist,
        'Bimodal': bimodal_dist,
        'Normal': normal_dist,
        'Exponential': exponential_dist
    }
    epsilon = 0.1
    sample_sizes = [100, 200, 500, 1000, 2000]

    # Initialize results matrix
    results_matrix = np.zeros((len(distributions), len(sample_sizes)), dtype=bool)

    for dist_idx, (name, p) in enumerate(distributions.items()):
        for size_idx, num_samples in enumerate(sample_sizes):
            print(f"Testing distribution: {name} with {num_samples} samples")
            result = test_collision_tester(p, num_samples, epsilon, name)
            results_matrix[dist_idx, size_idx] = result

    # Print results matrix
    print("\nResults Matrix (True = Accepted, False = Rejected):")
    print("Sample Sizes:", sample_sizes)
    for dist_idx, name in enumerate(distributions.keys()):
        print(f"{name}: {results_matrix[dist_idx]}")

    # Visualize the results matrix
    plt.figure(figsize=(10, 6))
    plt.imshow(results_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Acceptance (True=1, False=0)')
    plt.xticks(ticks=np.arange(len(sample_sizes)), labels=sample_sizes)
    plt.yticks(ticks=np.arange(len(distributions)), labels=distributions.keys())
    plt.xlabel('Sample Size')
    plt.ylabel('Distribution Type')
    plt.title('Collision Tester Results')
    # plt.savefig("results/collision_tester/collision_tester_results_matrix.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()