import random
import math
import matplotlib.pyplot as plt

def F_prime(i, n):
    """
    The filter F' that outputs i with probability 1/2, and the uniform distribution on [n] otherwise.

    Parameters:
    i (int): The input value.
    n (int): The range of the uniform distribution [1, n].

    Returns:
    int: The filtered value.
    """
    if random.random() < 0.5:
        return i
    else:
        return random.randint(1, n)

def F_double_prime_q_prime(i, q_prime, gamma, n):
    """
    The filter F''_q' that outputs i with probability m_i * gamma / n, where m_i = floor(q'(i) * n / gamma),
    and outputs n+1 otherwise.

    Parameters:
    i (int): The input value.
    q_prime (list): The modified distribution q'.
    gamma (float): The gamma parameter.
    n (int): The range of the uniform distribution [1, n].

    Returns:
    int: The filtered value.
    """
    m_i = math.floor(q_prime[i] * n / gamma)
    if random.random() < m_i * gamma / n:
        return i
    else:
        return n + 1

def reduce_to_O_n_grained(p, q, gamma, n):
    """
    Implement the algorithm proposed in the paper to reduce testing equality to a general distribution
    to testing equality to a O(n)-grained distribution.

    Parameters:
    p (list): The original distribution p.
    q (list): The original distribution q.
    gamma (float): The gamma parameter.
    n (int): The number of elements in the distributions.

    Returns:
    tuple: The reduced distributions p'' and q''.
    """
    # Step 1: Apply filter F'
    p_prime = [0.5 * p[i] + 0.5 / n for i in range(n)]
    q_prime = [0.5 * q[i] + 0.5 / n for i in range(n)]

    # Step 2: Apply filter F''_q'
    p_double_prime = [0] * (n + 1)
    q_double_prime = [0] * (n + 1)

    for i in range(n):
        p_double_prime[i] = p_prime[i] * (math.floor(p_prime[i] * n / gamma)) * gamma / n
        q_double_prime[i] = q_prime[i] * (math.floor(q_prime[i] * n / gamma)) * gamma / n

    p_double_prime[n] = 1 - sum(p_double_prime[:n])
    q_double_prime[n] = 1 - sum(q_double_prime[:n])

    return p_double_prime, q_double_prime

def main():
    n = 20
    gamma = 0.1

    p = [random.random() for _ in range(n)]
    q = [random.random() for _ in range(n)]

    p = [p_i / sum(p) for p_i in p]
    q = [q_i / sum(q) for q_i in q]

    print("Original distributions:")
    print("p:", p)
    print("q:", q)

    p_double_prime, q_double_prime = reduce_to_O_n_grained(p, q, gamma, n)

    print("\nReduced distributions:")
    print("p_double_prime:", p_double_prime)
    print("q_double_prime:", q_double_prime)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.bar(range(n), p, color='blue', alpha=0.7)
    plt.title("Original Distribution p")
    plt.xlabel("Value")
    plt.ylabel("Probability")

    plt.subplot(2, 2, 2)
    plt.bar(range(n), q, color='blue', alpha=0.7)
    plt.title("Original Distribution q")
    plt.xlabel("Value")
    plt.ylabel("Probability")

    plt.subplot(2, 2, 3)
    plt.bar(range(n + 1), p_double_prime, color='green', alpha=0.7)
    plt.title("Reduced Distribution p_double_prime")
    plt.xlabel("Value")
    plt.ylabel("Probability")

    plt.subplot(2, 2, 4)
    plt.bar(range(n + 1), q_double_prime, color='green', alpha=0.7)
    plt.title("Reduced Distribution q_double_prime")
    plt.xlabel("Value")
    plt.ylabel("Probability")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()