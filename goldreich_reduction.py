import numpy as np
import os
import math
import random
from typing import Dict, List, Tuple
from scipy.stats import chi2
import matplotlib.pyplot as plt
from generate_test_distributions import (
    generate_far_exact_distribution,
    generate_bimodal_distribution,
    generate_normal_distribution,
    generate_exponential_distribution
)

def F_prime(samples: List[int], n: int) -> List[int]:
    """
    The filter F' that outputs i with probability 1/2, and the uniform distribution on [n] otherwise.

    Parameters:
    samples (List[int]): The input samples.
    n (int): The range of the uniform distribution [1, n].

    Returns:
    List[int]: The filtered values.
    """
    return [sample if random.random() < 0.5 else random.randint(0, n-1) for sample in samples]

def mix_with_uniform(distribution: np.ndarray) -> Dict[str, float]:
    """
    Mixes the given distribution with a uniform distribution in a 50-50 ratio.

    Parameters:
    distribution (np.ndarray): The original distribution.

    Returns:
    Dict[str, float]: The mixed distribution.
    """
    n = len(distribution)
    return {str(i): 0.5 * distribution[i] + 0.5 / n for i in range(n)}

def create_grained_distribution(distribution: Dict[str, float], gamma: float = 1/6) -> Dict[str, float]:
    """
    Creates an m-grained distribution by flooring and redistributing the remaining mass.

    Parameters:
    distribution (Dict[str, float]): The input distribution.
    gamma (float): The gamma parameter for quantization.

    Returns:
    Dict[str, float]: The quantized distribution.
    """
    n = len(distribution)
    
    # Floor values
    grained_distribution = {}
    total_mass = 0.0
    
    for i in range(n):
        key = str(i)
        m_i = math.floor(distribution[key] * n / gamma)
        grained_distribution[key] = (m_i * gamma) / n
        total_mass += grained_distribution[key]
    
    # Distribute remaining mass to largest remainders
    remaining_mass = 1.0 - total_mass
    if remaining_mass > 0:
        remainders = [(i, distribution[str(i)] * n / gamma - math.floor(distribution[str(i)] * n / gamma)) 
                      for i in range(n)]
        remainders.sort(key=lambda x: x[1], reverse=True)
        
        units = int(remaining_mass * n / gamma)
        for i in range(units):
            if i < len(remainders):
                idx = str(remainders[i][0])
                grained_distribution[idx] += gamma / n
    
    # Ensure all keys are present in grained_distribution
    for i in range(n + 1):
        if str(i) not in grained_distribution:
            grained_distribution[str(i)] = 0.0
    
    return grained_distribution

def transform_samples_to_grained(p_prime_samples: List[int], q_prime: Dict[str, float], q_grained: Dict[str, float]) -> List[int]:
    """
    Transforms samples from p' to p'' using the ratio q''(i)/q'(i).

    Parameters:
    p_prime_samples (List[int]): Samples from the distribution p'.
    q_prime (Dict[str, float]): The modified distribution q'.
    q_grained (Dict[str, float]): The quantized distribution q''.
    gamma (float): The gamma parameter for quantization.

    Returns:
    List[int]: The transformed samples.
    """
    p_grained_samples = []

    for sample in p_prime_samples:
        sample_str = str(sample)
        if sample_str in q_prime and sample_str in q_grained:
            q_p = q_prime[sample_str]
            q_pp = q_grained[sample_str]

            if q_p > 0:
                keep_probability = q_pp / q_p

                if random.random() < keep_probability:
                    p_grained_samples.append(sample)

    return p_grained_samples

def reduce_to_O_n_grained(p: np.ndarray, q: Dict[str, float], gamma: float, n: int) -> Tuple[List[float], Dict[str, float]]:
    """
    Implement the algorithm proposed in the paper to reduce testing equality to a general distribution
    to testing equality to a O(n)-grained distribution.

    Parameters:
    p (np.ndarray): The original distribution p.
    q (Dict[str, float]): The original distribution q.
    gamma (float): The gamma parameter.
    n (int): The number of elements in the distributions.

    Returns:
    Tuple[List[float], Dict[str, float]]: The reduced distributions p'' and q''.
    """
    # Step 1: Apply filter F' to p and q
    p_prime = [0.5 * p[i] + 0.5 / n for i in range(n)]
    q_prime = mix_with_uniform(q)

    # Step 2: Apply filter F''_q'
    p_double_prime = [0.0] * (n + 1)
    q_double_prime = create_grained_distribution(q_prime, gamma)

    for i in range(n):
        m_i = math.floor(q_prime[str(i)] * n / gamma)
        if m_i > 0:
            p_double_prime[i] = p_prime[i] * (m_i * gamma) / n

    p_double_prime[n] = 1 - sum(p_double_prime[:n])

    return p_double_prime, q_double_prime

def epsilon_tester_uniformity(samples: List[int], m: int, epsilon: float) -> bool:
    """
    Epsilon tester for uniformity over [m].

    Parameters:
    samples (List[int]): List of samples.
    m (int): The parameter m for uniformity.
    epsilon (float): The accuracy parameter.

    Returns:
    bool: True if the samples are uniformly distributed, False otherwise.
    """
    expected_count = len(samples) / m
    counts = [samples.count(i) for i in range(1, m + 1)]
    chi_square_stat = sum((count - expected_count) ** 2 / expected_count for count in counts)
    threshold = chi2.ppf(1 - epsilon, df=m - 1)
    
    # Print the decision threshold
    print(f"Computed chi-square statistic: {chi_square_stat}")
    print(f"Decision threshold: {threshold}")
    
    return chi_square_stat <= threshold

def algorithm_8(p: np.ndarray, q: np.ndarray, epsilon: float, gamma: float = 1/6) -> bool:
    """
    Algorithm 8: Reducing testing equality to a general distribution to testing equality to a O(n)-grained distribution.

    Parameters:
    p (np.ndarray): The original distribution p.
    q (np.ndarray): The original distribution q.
    epsilon (float): The error parameter.

    Returns:
    bool: True if the distributions are considered equal, False otherwise.
    """
    n = len(p)
    
    # Step 1: Generate samples from p
    s = int(np.ceil(np.sqrt(n) / epsilon**2))
    samples = np.random.choice(range(n), size=s, p=p)

    # Step 2: Apply filter F' to the samples
    p_prime_samples = F_prime(samples, n)
    q_prime = mix_with_uniform(q)

    # Step 3: Create the grained distribution q''
    q_grained = create_grained_distribution(q_prime, gamma)

    # Step 4: Transform samples from p' to p''
    p_grained_samples = transform_samples_to_grained(p_prime_samples, q_prime, q_grained)

    return q_grained, p_grained_samples

def algorithm_5(q_grained: Dict[str, float], p_grained_samples: List[int], gamma: float = 1/6):
    n = len(q_grained)
    m = int(n / gamma)
    
    m_i = np.zeros(n)
    remaining = m
    
    # First allocate minimum positions to each used index
    used_indices = set(p_grained_samples)
    min_positions = remaining // len(used_indices)
    for i in used_indices:
        m_i[i] = min_positions
        remaining -= min_positions
    
    # Distribute remaining based on q''
    if remaining > 0:
        weights = [q_grained[str(i)] for i in range(n)]
        total_weight = sum(weights)
        for i in range(n):
            if i in used_indices:
                additional = int((weights[i]/total_weight) * remaining)
                m_i[i] += additional
                remaining -= additional
    
    # Map samples using full range
    ranges = []
    current = 1
    for size in m_i:
        if size > 0:
            ranges.append((current, current + size - 1))
            current += size
        else:
            ranges.append((0, 0))
    
    mapped_samples = []
    for s in p_grained_samples:
        start, end = ranges[s]
        if start > 0:
            mapped = random.randint(start, end)
            mapped_samples.append(mapped)
            
    return mapped_samples, m


def goldreich_reduction(p: np.ndarray, q: np.ndarray, epsilon: float, gamma: float = 1/6) -> bool:
    
    q_grained, p_grained_samples = algorithm_8(p, q, epsilon)
    
    mapped_samples, m = algorithm_5(q_grained, p_grained_samples, gamma)
    
    return mapped_samples, m

