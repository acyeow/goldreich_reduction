import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

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
    threshold = (m * epsilon ** 2) / 10
    
    return Z <= threshold

def main():
    q = {
        0: 0.25,
        1: 0.25,
        2: 0.25,
        3: 0.25
    }
    
    np.random.seed(42)
    samples = np.random.choice(list(q.keys()), size=1000, p=list(q.values())).tolist()
    
    epsilon = 0.1
    result = identity_tester(samples, q, epsilon)
    
    print(f"Identity tester result: {'Accepted' if result else 'Rejected'}")
    
    # Calculate the frequency of each sample
    sample_counts = {i: samples.count(i) for i in q.keys()}
    sample_frequencies = {i: count / len(samples) for i, count in sample_counts.items()}
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(q.keys(), q.values(), width=0.4, color='blue', alpha=0.7)
    plt.title("Known Distribution q")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    
    plt.subplot(1, 2, 2)
    # plt.hist(samples, bins=len(q), density=True, alpha=0.7, color='green', edgecolor='black')
    # plt.title("Histogram of Generated Samples")
    plt.bar(sample_frequencies.keys(), sample_frequencies.values(), width=0.4, color='green', alpha=0.7)
    plt.title("Bar Plot of Generated Samples")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    return result

if __name__ == "__main__":
    main()