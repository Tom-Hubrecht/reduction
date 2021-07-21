# ##############################################################################
#  Shortest Vector Problem pour un réseau avec la norme infinie
# ##############################################################################

from math import ceil, log

import numpy as np


def infinite_norm(x):
    """Renvoie la norme infinie du vecteur"""
    return max(abs(x))


def get_index(y, R, gamma):
    """
    Renvoie les coordonnées de y dans le découpage en cubes de côté gamma
    """
    return np.floor((y + R) / (gamma * R))


def fundamental_domain(x, b):
    """
    Renvoie x mod B i.e. y in P(B) tel que x - y in L
    """
    pass


def sample(b, d, m):
    """
    Renvoie (e, y) in B_n^{(infnty)} times P(B) avec y - e in L
    """
    rng = np.random.default_rng()

    e = rng.uniform(-d, d, m)
    y = fundamental_domain(e, b)
    return e, y


def sieve(S, gamma, R, xi):
    """
    Renvoie une filtration de S
    """
    S_prime = set()
    C = dict()

    for e, y in S:
        if infinite_norm(y) <= gamma * R:
            S_prime.add((e, y))
        else:
            i = get_index(y, R, gamma)
            if i in C:
                e_c, c = C[i]
                S_prime.add((e, y - c + e_c))
            else:
                C[i] = (e, y)

    return S_prime


def svp_infinite(b, gamma, xi, _lambda, N):
    """
    Renvoie le plus petit vecteur du réseau de base b
    """
    S = set()

    (n, m) = b.shape
    R = n * max(np.apply_along_axis(infinite_norm, 1, b))

    for i in range(N):
        e, y = sample(b, xi * _lambda)
        S.add((e, y))

    k = ceil(log((xi / (n * R * (1 - gamma))), gamma))
    for j in range(k):
        S = sieve(S, gamma, R, xi)
        R = gamma * R + xi * _lambda

    v_0 = None
    for e_i, y_i in S:
        for e_j, y_j in S:
            v = (y_i - e_i) - (y_j - e_j)
            if v.any() and (v_0 is None or infinite_norm(v) < infinite_norm(v_0)):
                v_0 = v

    return v_0
