import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import os
from generate_test_distributions import (
    generate_far_exact_distribution,
    generate_bimodal_distribution,
    generate_normal_distribution,
    generate_exponential_distribution
)

def identity_tester(samples: List[int], q: Dict[int, float], epsilon: float) -> bool:
    """
    Implementation of the Identity Tester algorithm that checks if an unknown distribution p
    is equal to a known distribution q.
    
    Parameters:
    -----------
    samples : List[int]
        List of m samples drawn from the unknown distribution p
    q : Dict[int, float]
        Known distribution q represented as a dictionary mapping indices to probabilities
    epsilon : float
        Accuracy parameter for the test
        
    Returns:
    --------
    bool
        True if the distributions are considered equal, False otherwise
    """
    # Step 1: Compute Ni (frequency counts of samples)
    m = len(samples)
    n = len(q)  # Size of the support
    
    # Count occurrences of each value in samples
    Ni = {}
    for i in q.keys():
        Ni[i] = sum(1 for x in samples if x == i)
    
    # Step 2: Compute set A = {i : qi ≥ epsilon/50n}
    threshold = epsilon / (50 * n)
    A = {i for i, prob in q.items() if prob >= threshold}
    
    # Step 3: Compute Z = Σ(i∈A) (Ni - mqi)^2 - Ni) / (mqi)
    Z = 0
    for i in A:
        mqi = m * q[i]  # Expected count under q
        if mqi > 0:  # Avoid division by zero
            Z += ((Ni[i] - mqi) ** 2 - Ni[i]) / mqi
    
    # Step 4: Accept if and only if Z ≤ mε^2/10
    decision_threshold = (m * epsilon ** 2) / 10
    
    # Debugging print statements
    print(f"Computed Z value: {Z}")
    print(f"Decision threshold: {decision_threshold}")
    
    return Z <= decision_threshold

def test_identity_tester(q: Dict[int, float], num_samples: int, epsilon: float, dist_name: str) -> bool:
    """
    Test the identity tester with the given distribution q, number of samples, and epsilon value.

    Parameters:
    q (Dict[int, float]): Known distribution q represented as a dictionary mapping indices to probabilities.
    num_samples (int): Number of samples to draw from the distribution.
    epsilon (float): Accuracy parameter for the test.
    dist_name (str): The name of the distribution.

    Returns:
    bool: True if the distributions are considered equal, False otherwise.
    """
    np.random.seed(42)
    samples = np.random.choice(list(q.keys()), size=num_samples, p=list(q.values())).tolist()
    
    result = identity_tester(samples, q, epsilon)
    
    print(f"Identity tester result for {dist_name} with {num_samples} samples: {'Accepted' if result else 'Rejected'}")
    
    # Calculate the frequency of each sample
    sample_counts = {i: samples.count(i) for i in q.keys()}
    sample_frequencies = {i: count / len(samples) for i, count in sample_counts.items()}
    
    # Create results directory if it doesn't exist
    os.makedirs("results/identity_tester", exist_ok=True)
    
    # Save the distribution and sample frequencies plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(q.keys(), q.values(), width=0.4, color='blue', alpha=0.7)
    plt.title(f"Known Distribution {dist_name}")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    
    plt.subplot(1, 2, 2)
    plt.bar(sample_frequencies.keys(), sample_frequencies.values(), width=0.4, color='green', alpha=0.7)
    plt.title(f"Bar Plot of Samples for {dist_name} with {num_samples} samples")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(f"results/identity_tester/{dist_name}_{num_samples}_samples.png")
    plt.close()
    
    return result

def main():
    # Generate distributions using the functions from generate_test_distributions.py
    n = 10
    far_exact_dist = generate_far_exact_distribution(n)
    bimodal_dist = generate_bimodal_distribution(n)
    normal_dist = generate_normal_distribution(n)
    exponential_dist = generate_exponential_distribution(n)

    # Convert distributions to dictionaries
    distributions = {
        'Far Exact': {i: far_exact_dist[i] for i in range(n)},
        'Bimodal': {i: bimodal_dist[i] for i in range(n)},
        'Normal': {i: normal_dist[i] for i in range(n)},
        'Exponential': {i: exponential_dist[i] for i in range(n)}
    }
    epsilon = 0.1
    sample_sizes = [100, 200, 500, 1000, 2000]

    # Initialize results matrix
    results_matrix = np.zeros((len(distributions), len(sample_sizes)), dtype=bool)

    for dist_idx, (name, q) in enumerate(distributions.items()):
        for size_idx, num_samples in enumerate(sample_sizes):
            print(f"Testing distribution: {name} with {num_samples} samples")
            result = test_identity_tester(q, num_samples, epsilon, name)
            results_matrix[dist_idx, size_idx] = result

    # Print results matrix
    print("\nResults Matrix (True = Accepted, False = Rejected):")
    print("Sample Sizes:", sample_sizes)
    for dist_idx, name in enumerate(distributions.keys()):
        print(f"{name}: {results_matrix[dist_idx]}")

    # Visualize the results matrix
    os.makedirs("results/identity_tester", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.imshow(results_matrix, aspect='auto')
    plt.colorbar(label='Acceptance (True=1, False=0)')
    plt.xticks(ticks=np.arange(len(sample_sizes)), labels=sample_sizes)
    plt.yticks(ticks=np.arange(len(distributions)), labels=distributions.keys())
    plt.xlabel('Sample Size')
    plt.ylabel('Distribution Type')
    plt.title('Identity Tester Results')
    plt.savefig("results/identity_tester/identity_tester_results_matrix.png")
    plt.close()

if __name__ == "__main__":
    main()