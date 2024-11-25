import numpy as np
from math import factorial
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