import numpy as np
import math
import random
from typing import Dict, List, Tuple
from scipy.stats import chi2


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
    mixed_distribution = {}
    uniform_prob = 0.5 / n

    for i in range(n):
        mixed_distribution[str(i)] = 0.5 * distribution[i] + uniform_prob

    return mixed_distribution

def create_grained_distribution(distribution: Dict[str, float], gamma: float = 1/6) -> Dict[str, float]:
    """
    Creates an m-grained distribution by flooring and redistributing the remaining mass.

    Parameters:
    distribution (Dict[str, float]): The input distribution.
    gamma (float): The gamma parameter for quantization.

    Returns:
    Dict[str, float]: The quantized distribution.
    """
    num_elements = len(distribution)
    grained_distribution = {}
    total_grained_mass = 0.0

    # Floor values and calculate total grained mass
    for i in range(num_elements):
        key = str(i)
        floored_value = math.floor(distribution[key] * num_elements / gamma)
        grained_distribution[key] = (floored_value * gamma) / num_elements
        total_grained_mass += grained_distribution[key]

    # Distribute remaining probablity mass
    remaining_mass = 1.0 - total_grained_mass
    if remaining_mass > 0:
        remainders = sorted(
            ((i, distribution[str(i)] * num_elements / gamma - math.floor(distribution[str(i)] * num_elements / gamma)) for i in range(num_elements)),
            key=lambda x: x[1], reverse=True
        )
        units_to_distribute = int(remaining_mass * num_elements / gamma)
        for i in range(units_to_distribute):
            if i < len(remainders):
                idx = str(remainders[i][0])
                grained_distribution[idx] += gamma / num_elements

    # Ensure all keys are present in the grained_distribution
    for i in range(num_elements + 1):
        grained_distribution.setdefault(str(i), 0.0)

    return grained_distribution

def transform_samples_to_grained(p_prime_samples: List[int], q_prime: Dict[str, float], q_grained: Dict[str, float]) -> List[int]:
    """
    Transforms samples from p' to p'' using the ratio q''(i)/q'(i).

    Parameters:
    p_prime_samples (List[int]): Samples from the distribution p'.
    q_prime (Dict[str, float]): The modified distribution q'.
    q_grained (Dict[str, float]): The quantized distribution q''.

    Returns:
    List[int]: The transformed samples.
    """
    transformed_samples = []

    for sample in p_prime_samples:
        sample_key = str(sample)
        q_prime_value = q_prime.get(sample_key, 0)
        q_grained_value = q_grained.get(sample_key, 0)

        if q_prime_value > 0:
            acceptance_probability = q_grained_value / q_prime_value
            if random.random() < acceptance_probability:
                transformed_samples.append(sample)

    return transformed_samples

def reduce_to_O_n_grained(p: np.ndarray, q: Dict[str, float], gamma: float, n: int) -> Tuple[List[float], Dict[str, float]]:
    """
    Reduces testing equality to a general distribution to testing equality to a O(n)-grained distribution.

    Parameters:
    p (np.ndarray): The original distribution p.
    q (Dict[str, float]): The original distribution q.
    gamma (float): The gamma parameter.
    n (int): The number of elements in the distributions.

    Returns:
    Tuple[List[float], Dict[str, float]]: The reduced distributions p'' and q''.
    """
    # Step 1: Apply filter F' to p
    p_prime = F_prime(list(range(n)), n)
    
    # Step 2: Mix q with uniform distribution
    q_prime = mix_with_uniform(q)
    
    # Step 3: Create the grained distribution q''
    q_grained = create_grained_distribution(q_prime, gamma)
    
    # Step 4: Transform p' to p''
    p_double_prime = transform_samples_to_grained(p_prime, q_prime, q_grained)
    
    return p_double_prime, q_grained

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
    counts = np.bincount(samples, minlength=m+1)[1:]  # Count occurrences of each sample
    chi_square_stat = np.sum((counts - expected_count) ** 2 / expected_count)
    threshold = chi2.ppf(1 - epsilon, df=m - 1)
    
    # Print the decision threshold
    print(f"Computed chi-square statistic: {chi_square_stat}")
    print(f"Decision threshold: {threshold}")
    
    return chi_square_stat <= threshold

def algorithm_8(p: np.ndarray, q: np.ndarray, epsilon: float, gamma: float = 1/6) -> Tuple[Dict[str, float], List[int]]:
    """
    Algorithm 8: Reducing testing equality to a general distribution to testing equality to a O(n)-grained distribution.

    Parameters:
    p (np.ndarray): The original distribution p.
    q (np.ndarray): The original distribution q.
    epsilon (float): The error parameter.
    gamma (float): The gamma parameter for quantization.

    Returns:
    Tuple[Dict[str, float], List[int]]: The grained distribution q'' and the transformed samples p''.
    """
    n = len(p)
    
    # Step 1: Generate samples from p
    s = int(np.ceil(np.sqrt(n) / epsilon**2))
    samples = np.random.choice(range(n), size=s, p=p)

    # Step 2: Apply filter F' to the samples
    p_prime_samples = F_prime(samples, n)
    
    # Step 3: Mix q with uniform distribution
    q_prime = mix_with_uniform(q)

    # Step 4: Create the grained distribution q''
    q_grained = create_grained_distribution(q_prime, gamma)

    # Step 5: Transform samples from p' to p''
    p_grained_samples = transform_samples_to_grained(p_prime_samples, q_prime, q_grained)

    return q_grained, p_grained_samples

def algorithm_5(q_grained: Dict[str, float], p_grained_samples: List[int], gamma: float = 1/6) -> Tuple[List[int], int]:
    """
    Distributes positions among elements based on their weights and maps samples to these positions.
    Args:
        q_grained (Dict[str, float]): A dictionary where keys are element indices (as strings) and values are their corresponding weights.
        p_grained_samples (List[int]): A list of sample indices to be mapped to positions.
        gamma (float, optional): A parameter that determines the total number of positions. Default is 1/6.
    Returns:
        Tuple[List[int], int]: A tuple containing:
            - A list of mapped sample positions.
            - The total number of positions allocated.
    """
    num_elements = len(q_grained)
    total_positions = int(num_elements / gamma)
    
    # Initialize position allocation array
    position_allocation = np.zeros(num_elements)
    remaining_positions = total_positions
    
    # Allocate minimum positions to each used index
    unique_samples = set(p_grained_samples)
    min_positions_per_index = remaining_positions // len(unique_samples)
    for index in unique_samples:
        position_allocation[int(index)] = min_positions_per_index
        remaining_positions -= min_positions_per_index
    
    # Distribute remaining positions based on q_grained
    if remaining_positions > 0:
        weights = np.array([q_grained[str(i)] for i in range(num_elements)])
        total_weight = np.sum(weights)
        for i in range(num_elements):
            if i in unique_samples:
                additional_positions = int((weights[i] / total_weight) * remaining_positions)
                position_allocation[i] += additional_positions
                remaining_positions -= additional_positions
    
    # Create ranges for mapping samples
    sample_ranges = []
    current_position = 1
    for allocation in position_allocation:
        if allocation > 0:
            sample_ranges.append((current_position, current_position + int(allocation) - 1))
            current_position += int(allocation)
        else:
            sample_ranges.append((0, 0))
    
    # Map samples to the full range
    mapped_samples = []
    for sample in p_grained_samples:
        start, end = sample_ranges[sample]
        if start > 0:
            mapped_samples.append(random.randint(start, end))
            
    return mapped_samples, total_positions


def goldreich_reduction(p: np.ndarray, q: np.ndarray, epsilon: float, gamma: float = 1/6) -> bool:
    """
    Performs the Goldreich reduction algorithm to test uniformity of a distribution.
    This function applies the Goldreich reduction to test whether the distribution `p` is 
    epsilon-close to the uniform distribution over the domain defined by `q`. It uses 
    auxiliary algorithms (algorithm_8 and algorithm_5) to achieve this.
    Parameters:
    p (np.ndarray): The distribution to be tested.
    q (np.ndarray): The reference distribution.
    epsilon (float): The closeness parameter for the uniformity test.
    gamma (float, optional): The parameter for the reduction algorithm. Default is 1/6.
    Returns:
    bool: True if the distribution `p` is epsilon-close to the uniform distribution, False otherwise.
    """
    
    q_grained, p_grained_samples = algorithm_8(p, q, epsilon)
    
    mapped_samples, m = algorithm_5(q_grained, p_grained_samples, gamma)
    
    return epsilon_tester_uniformity(mapped_samples, m, epsilon)