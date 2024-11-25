import numpy as np
import matplotlib.pyplot as plt
import math

def collision_tester(p: np.ndarray, m: int, epsilon: float) -> bool:
    """
    Implements the CollisionTester algorithm from Lecture 13.

    Parameters:
    p (np.ndarray): The distribution over [n] from which to draw samples.
    m (int): The number of samples to draw.
    epsilon (float): The error parameter.

    Returns:
    bool: True if the distribution p is accepted, False otherwise.
    """
    if 2 > m:
        raise ValueError("m must be at least 2.")
    # Step 1: Take m samples x1, ..., xm drawn i.i.d. from p
    samples = np.random.choice(len(p), size=m, replace=True, p=p)

    # Step 2: Compute Yij = 1{xi = xj} and C = sum_{i < j} Yij / (m choose 2)
    Yij = np.sum([samples[i] == samples[j] for i in range(m) for j in range(i+1, m)])
    C = Yij / math.comb(m, 2)

    # Step 3: Accept if and only if C <= 1 + 0.01 * sqrt(n)
    return C <= 1 + 0.01 * epsilon**2 / len(p)

def main():
    n = 10
    m = 100
    epsilon = 0.1

    p = np.random.dirichlet(np.ones(n), size=1)[0]

    result = collision_tester(p, m, epsilon)

    print(f"Collision tester result: {'Accepted' if result else 'Rejected'}")

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

if __name__ == "__main__":
    main()