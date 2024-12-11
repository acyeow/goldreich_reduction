import numpy as np
import matplotlib.pyplot as plt
import math
from generate_test_distributions import generate_far_exact_distribution, generate_bimodal_distribution, generate_normal_distribution, generate_exponential_distribution

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
    return C <= 1 + epsilon**2 / len(p)

def test_collision_tester(p: np.ndarray, m: int, epsilon: float):
    """
    Test the collision tester with the given distribution p, number of samples, and epsilon value.

    Parameters:
    p (np.ndarray): The distribution over [n] from which to draw samples.
    m (int): The number of samples to draw.
    epsilon (float): The error parameter.
    """
    # Run the collision tester
    result = collision_tester(p, m, epsilon)

    print(f"Collision tester result: {'Accepted' if result else 'Rejected'}")

    # Plot the distribution p and the histogram of samples
    n = len(p)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(n), p, color='blue', alpha=0.7)
    plt.title("Distribution p")
    plt.xlabel("Outcome")
    plt.ylabel("Probability")

    samples = np.random.choice(len(p), size=m, replace=True, p=p)

    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=n, density=True, alpha=0.7, color='green', edgecolor='black')
    plt.title("Histogram of Samples")
    plt.xlabel("Outcome")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def main():
    # Generate distributions
    n = 100
    far_exact_dist = generate_far_exact_distribution(n)
    bimodal_dist = generate_bimodal_distribution(n)
    normal_dist = generate_normal_distribution(n)
    exponential_dist = generate_exponential_distribution(n)

    # Normalize the distributions to sum to 1
    far_exact_dist /= np.sum(far_exact_dist)
    bimodal_dist /= np.sum(bimodal_dist)
    normal_dist /= np.sum(normal_dist)
    exponential_dist /= np.sum(exponential_dist)

    distributions = [
        far_exact_dist,
        bimodal_dist,
        normal_dist,
        exponential_dist
    ]
    m = 1000
    epsilon = 0.1

    for i, p in enumerate(distributions):
        print(f"Testing distribution {i + 1}")
        test_collision_tester(p, m, epsilon)

if __name__ == "__main__":
    main()