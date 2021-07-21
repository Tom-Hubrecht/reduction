# ##############################################################################
#  Réduction de base de réseau avec la norme infinie
# ##############################################################################

import numpy as np


def infinite_norm(x):
    """Renvoie la norme infinie du vecteur"""
    return max([abs(x_i) for x_i in x])


def smallest_vector(m):
    """Renvoie le plus petit vecteur au sens de la norme infine"""
    if m.size == 0:
        return None

    normes = np.array([infinite_norm(x) for x in m])
    return m[np.argmin(normes)]


def modular_exponent(b, e, m):
    """
    Renvoie b^e [m]
    (Implantation adaptée de https://en.wikipedia.org/wiki/Modular_exponentiation)
    """
    if m == 1:
        return 0

    r = 1
    while e > 0:
        if e % 2:
            r = (r * b) % m

        e = e // 2
        b = (b * b) % m

    return r


def build_lattice_base_even(p, n, gamma):
    """
    Renvoie une base du réseau de la forme :
    |     p     0  0  ···  0 |
    |    -g     1  0  ···  0 |
    |   -g^2    0  1  ···  0 |
    |     ·     ·  ·   ·   · |
    | -g^{n-1}  0  0  ···  1 |
    """
    base = np.eye(n, dtype=object)
    base[0, 0] = p

    for i in range(1, n):
        base[i, 0] = -modular_exponent(gamma, i, p)

    return base


def build_lattice_base_odd(p, n, gamma):
    """
    Renvoie une base du réseau de la forme :
    |    p     0  0  ···  0 |
    |   t_1    1  0  ···  0 |
    |   t_2    0  1  ···  0 |
    |    ·     ·  ·   ·   · |
    | t_{n-1}  0  0  ···  1 |
    avec t_i = -g^i + (g^i [2]) * p
    """
    base = np.eye(n, dtype=object)
    base[0, 0] = p

    for i in range(1, n):
        t = modular_exponent(gamma, i, p)
        base[i, 0] = -t + p if t % 2 else -t

    return base


def infinite_distance(x, b, i):
    """
    Calcule F_i(x) = min_{z_k in R} || x - sum_{j = 1}^{i - 1} z_j b_j ||_infnty
    """
    (n,) = x.shape

    # On minimise le système de contraintes avec PPL
    p = MixedIntegerLinearProgram(maximization=False, solver="GLPK/exact")  # noqa
    z = p.new_variable()
    p.add_constraint(z[i] >= 0)
    p.add_constraint(z[i] <= infinite_norm(x))

    for k in range(n):
        p.add_constraint(p.sum(z[j] * b[j, k] for j in range(i)) - z[i] <= -x[k])
        p.add_constraint(-p.sum(z[j] * b[j, k] for j in range(i)) - z[i] <= x[k])

    p.set_objective(z[i])
    return int(p.solve(objective_only=True))


def get_minimal_mu(b, k, j):
    """
    Renvoie un nombre entier qui minimise F_j(b_k - mu * b_j)
    """
    (m, n) = b.shape

    # On minimise le système de contraintes avec PPL
    p = MixedIntegerLinearProgram(maximization=False, solver="PPL")  # noqa
    z = p.new_variable()
    mu = p.new_variable(integer=True)
    p.add_constraint(z[j] >= 0)
    p.add_constraint(z[j] <= infinite_norm(b[k]))

    for q in range(n):
        p.add_constraint(
            p.sum(z[i] * b[i, q] for i in range(j)) - z[j]
            <= -(b[k, q] - mu[0] * b[j, q])
        )
        p.add_constraint(
            -p.sum(z[i] * b[i, q] for i in range(j)) - z[j] <= b[k, q] - mu[0] * b[j, q]
        )

    p.set_objective(z[j])
    p.solve()

    return int(p.get_values(mu[0]))


def reduction_LS(b, delta):
    """
    Renvoie une base réduite de paramètre delta selon l'algorithme de Lovász et Scarf
    """
    k = 1
    m, n = b.shape

    while k < n:
        # Minimisation de b_k
        for j in range(k - 1, -1, -1):
            # Minimisation de mu
            mu_j = get_minimal_mu(b, k, j)

            # On modifie b_k
            b[k] -= mu_j * b[j]

        if infinite_distance(b[k], b, k - 1) < delta * infinite_distance(
            b[k - 1], b, k - 1
        ):
            # On échange les deux vecteurs et on recommence
            b[k - 1], b[k] = b[k].copy(), b[k - 1].copy()
            k = max(1, k - 1)
        else:
            k = k + 1

    return b


def reduction_double(b, delta):
    """
    Renvoie une base résuite selon l'algorithme LLL en norme 2 puis Delta-LS en norme infinie
    """
    _b = matrix(b).LLL(delta=np.sqrt(delta), algorithm="NTL:LLL")  # noqa: F821
    return reduction_LS(np.array(_b), delta)


def find_valid_vector_even(p, n, gamma):
    """
    Renvoie le plus petit vecteur valide dans la base réduite lorsque lambda est pair
    """
    _b = build_lattice_base_even(p, n, gamma)
    b = reduction_LS(_b, 1)

    return smallest_vector(np.array([v for v in b if v[0] % 2]))


def build_weight_vector(k, n):
    """
    Renvoie l'écriture binaire de k dans un vecteur de taille n
    """
    return np.array([int(b) for b in f"{{:0{n}b}}".format(k)], dtype=object)


def check_vector_odd(v, e, R):
    """
    Renvoie True si le vecteur v est valide lorsque lambda est impair
    """
    v_bool = v % 2

    if np.count_nonzero(v_bool) % 2:
        m = R(list(v_bool))
        return e.gcd(m) == 1

    return False


def find_valid_vector_odd(p, n, gamma):
    """
    Renvoie le plus petit vecteur valide dans la base réduite lorsque lambda est impair
    """
    R.<x> = GF(2)[]  # noqa

    e = R(x ** n - 1)

    _b = build_lattice_base_odd(p, n, gamma)
    b = reduction_LS(_b, 1)

    v_max = 1 << n

    vectors = []
    for k in range(1, v_max):
        v_k = build_weight_vector(k, n)
        v_sum = np.zeros(n, dtype=object)

        for i in range(n):
            v_sum += v_k[i] * b[i]

        if check_vector_odd(v_sum, e, R):
            vectors.append(v_sum)

    return smallest_vector(np.array(vectors, dtype=object))


def find_valid_vector(p, n, gamma, _lambda):
    """
    Renvoie un vecteur valide suivant la parité de lambda
    """
    if _lambda % 2:
        return find_valid_vector_odd(p, n, gamma)

    return find_valid_vector_even(p, n, gamma)


def is_power_of_2(n):
    """
    Si n = 2^k, n = 0b10...0 et n - 1 = 0b01...1, dans le cas contraire, n & (n - 1) != 0
    """
    return n & (n - 1) == 0


def log2(n):
    """
    Renvoie le nombre de bits nécessaire pour stocker des entiers strictement inférieurs
    à n, si n = 2^k, on a besoin de k - 1 bits
    """
    return n.bit_length() - is_power_of_2(n)


def compute_rho_log2(w_size, n, _lambda, phi, r_coefficients):
    """
    Renvoie log_2(rho) où rho est la valeur absolue maximale des coefficients de l'AMNS
    """
    r_max = infinite_norm(r_coefficients)

    if r_max == 0:
        return 0

    w = 1 + (n - 1) * abs(_lambda)

    rho = 2 * w * r_max
    rho_log2 = log2(int(rho))

    _phi = 2 * w * rho

    # On a au plus w_size - 1 bits utilisables car les entiers sont signés
    if rho_log2 > w_size - 1:
        return 0

    if _phi > phi:
        return 0

    return rho_log2


def is_suitable(n):
    """
    Renvoie True si n est de la forme ±2^k ou ±2^a ±2^b
    """
    n_abs = abs(n)
    bits = f"{n:b}".strip("0")

    # Si |n| = 2^k, bits vaudra '1'
    # Si |n| = 2^a + 2^b, bits sera de la forme '10...01'
    # Si |n| = 2^a - 2^b, bits sera de la forme '1...1'
    return bits.count("1") <= 2 or bits.count("0") == 0


def timeout(func, args=(), kwargs={}, timeout_duration=30):
    @fork(timeout=timeout_duration, verbose=True)
    def my_new_func():
        return func(*args, **kwargs)

    return my_new_func()


def find_roots(F, n, _lambda, find_all=True):
    """
    Calcule les racines de lambda dans F si find_all vaut True, la première sinon
    """

    roots = timeout(F(_lambda).nth_root, args=(n,), kwargs={"all": find_all})

    if type(roots) is str:
        return []

    if find_all:
        return roots

    return [roots]


def polynomial_to_array(p):
    """
    Renvoie un array Numpy des coefficients du polynôme
    """
    return np.array(list(p))


def compute_inv_internal(n, phi, p_ext, p_int):
    """
    Calcule -M^{-1} [E, phi] où M = p_int et E = p_ext
    """
    P.<x> = ZZ.quo(phi)[]  # noqa
    return -P(p_int.inverse_mod(p_ext))


def internal_reduction(a, p_ext, p_int, p_inv, phi, R, P):
    """
    Calcule une représentation de x * phi^{-1}
    """
    # q = (P(a) * P(p_inv)).mod(P(p_ext))
    # return (a + (R(q) * p_int).mod(p_ext)) / phi

    # def amns_red_int(op, ext_pol, ri_poly, neg_inv_ri, R, PP, phi):
    q = (P(a) * p_inv).mod(P(p_ext))
    r0 = a + (R(q) * p_int).mod(p_ext)
    return r0 / phi


def convert_to_amns(a, n, p, gamma, phi, p_ext, p_int, p_inv, rho):
    """
    Convertit a dans le système AMNS décrit par les arguments
    """
    F = GF(p)
    R.<x> = QQ[]  # noqa
    P.<x> = ZZ.quo(phi)[]  # noqa

    _rep = Integer(F(a * modular_exponent(phi, n - 1, p)))
    rep = internal_reduction(R(_rep), p_ext, p_int, p_inv, phi, R, P)

    for _ in range(n - 2):
        rep = internal_reduction(rep, p_ext, p_int, p_inv, phi, R, P)

    if F(rep(R(gamma))) != F(a):
        raise RuntimeError("Conversion vers l'AMNS ratée")

    if rep.norm(infinity) >= rho:
        raise RuntimeError("Valeur en dehors de l'AMNS après conversion")

    return rep


class AMNS:
    """
    Classe représentant un AMNS
    """

    def __init__(self, rho_log2, p, n, _lambda, gamma, phi, r_coefficients, w_size):
        # Constantes de l'AMNS
        self.rho_log2 = rho_log2
        self.w_size = w_size
        self.phi = phi
        self.nb_coefficients = n
        self.prime = p
        self.gamma = gamma

        R.<x> = QQ[]  # noqa

        # Polynôme de réduction externe
        self.p_external = R(x ** n - _lambda)

        # Polynôme de réduction interne
        self.p_internal = R(list(r_coefficients))

        # -M^{-1} où M est le polynôme de réduction interne
        self.inv_internal = compute_inv_internal(
            n, phi, self.p_external, self.p_internal
        )

        phi_2 = phi ** 2
        rho = 2 ** rho_log2
        self.convert = {
            "P0": convert_to_amns(
                phi_2,
                n,
                p,
                gamma,
                phi,
                self.p_external,
                self.p_internal,
                self.inv_internal,
                rho,
            ),
            "P1": convert_to_amns(
                phi_2 * rho,
                n,
                p,
                gamma,
                phi,
                self.p_external,
                self.p_internal,
                self.inv_internal,
                rho,
            ),
        }

    def __repr__(self):
        return (
            "{"
            f"'rho_log2': {self.rho_log2}, "
            f"'w_size': {self.w_size}, "
            f"'n': {self.nb_coefficients}, "
            f"'p': {self.prime}, "
            f"'gamma': {self.gamma}, "
            f"'p_ext': {self.p_external}, "
            f"'p_int': {self.p_internal}, "
            f"'p_inv': {self.inv_internal}, "
            f"'convert': {self.convert}"
            "}"
        )


# @parallel
def check_lambda(w_size, F, p, n, phi, _lambda, find_all=True):
    """
    Renvoie la liste des AMNS générés avec lambda s'il en existe
    """
    roots = find_roots(F, n, _lambda, find_all)
    amns = []

    for gamma in roots:
        # La racine 1 est peu intéressante
        if gamma != 1:
            r_coefficients = find_valid_vector(p, n, int(gamma), _lambda)
            rho_log2 = compute_rho_log2(w_size, n, _lambda, phi, r_coefficients)

            if rho_log2 > 0:
                amns.append(
                    AMNS(
                        rho_log2, p, n, _lambda, int(gamma), phi, r_coefficients, w_size
                    )
                )
    print(".", end="", flush=True)
    return amns


# @parallel
def build_amns(w_size, F, p, n, phi, m_lambda, return_first=False, find_all=True):
    """
    Renvoie les AMNS construits pour |lambda| < m_lambda non nuls si build_all vaut True,
    renvoie le premier trouvé sinon
    """
    # print(f"n = {n} ", end="", flush=True)
    amns = []
    # lambdas = [_lambda for _lambda in range(1, m_lambda + 1) if is_suitable(_lambda)]

    # args = (
    #     [(w_size, F, p, n, phi, _lambda, find_all) for _lambda in lambdas]
    #     + [(w_size, F, p, n, phi, -_lambda, find_all) for _lambda in lambdas]
    # )

    # amns = list(check_lambda(args))

    for a_lambda in range(1, m_lambda + 1):
        if is_suitable(a_lambda):
            p_amns = check_lambda(w_size, F, p, n, phi, a_lambda, find_all)
            print(".", end="", flush=True)

            n_amns = check_lambda(w_size, F, p, n, phi, -a_lambda, find_all)
            print(".", end="", flush=True)

            amns += p_amns + n_amns

            if return_first and amns != []:
                return [amns[0]]

    # print(" [done]")

    return amns


def build_amns_range(
    w_size, p, m_lambda, n_min, n_max, return_first=False, find_all=True
):
    """
    Calcule les AMNS avec un nombre de coefficients entre n_min et n_max
    """
    phi = 2 ** w_size
    F = GF(p)

    amns = []

    # args = [
    #     (w_size, F, p, n, phi, m_lambda, return_first, find_all) for n in range(n_min, n_max + 1)
    # ]
    # amns = list(build_amns(args))
    for n in range(n_min, n_max + 1):
        print(f"n = {n} ", end="", flush=True)
        amns += build_amns(w_size, F, p, n, phi, m_lambda, return_first, find_all)
        print(" [done]")

        if return_first and amns != []:
            return [amns[0]]

    return amns
