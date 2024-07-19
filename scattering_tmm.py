import numpy as np

def calculate_parameters_for_layer(layer, k_x, k_y):
    # Account for layer formatting difference
    if len(layer) == 3:
        epsilon, myu, length  = layer
        k_z = np.sqrt(epsilon * myu  - k_x ** 2 - k_y ** 2)
        omega = 1j * k_z * np.eye(2)
        Q = (1 / myu) * np.matrix([
            [k_x * k_y, epsilon * myu - k_x ** 2],
            [k_y ** 2 - epsilon * myu, -k_x * k_y]])
    
    elif len(layer) == 2:
        n, length = layer
        k_z = np.sqrt(n**2 - k_x**2 - k_y**2)
        omega = 1j * k_z * np.eye(2)
        Q = np.matrix([
            [k_x * k_y, n**2 - k_x**2],
            [k_y**2 - n**2, -k_x * k_y]])
    
    V = Q * np.linalg.inv(omega)
    return V, omega, length

def calculate_s_matrix(V, omega, L, k_0, V_gap):
    lambda_ = omega
    X = np.diag(np.exp(np.diag(lambda_) * L * k_0))
    eta = np.linalg.inv(V) * V_gap
    A = np.eye(2) + eta
    B = np.eye(2) - eta
    XB =  X * B
    A_inv = np.linalg.inv(A)
    D = A - X * B * A_inv * XB
    D_inv = np.linalg.inv(D)
    S_11_i = S_22_i = D_inv * (XB * A_inv * X * A - B)
    S_12_i = S_21_i = D_inv * X * (A - B * A_inv * B)
    
    return S_11_i, S_12_i, S_21_i, S_22_i

def redheffer_product(S_global, S_i):
    S_11, S_12, S_21, S_22 = S_global
    S_11_i, S_12_i, S_21_i, S_22_i = S_i
    
    D = S_12 * np.linalg.inv(np.eye(2) - S_11_i * S_22)
    F = S_21_i * np.linalg.inv(np.eye(2) - S_22 * S_11_i)
    
    S_11, S_12, S_21, S_22 = \
        S_11 + D * S_11_i * S_21, \
        D * S_12_i, \
        F * S_21, \
        S_22_i + F * S_22 * S_12_i
    
    return S_11, S_12, S_21, S_22

def tmm_solve(layers, theta, phi, pte, ptm, wavelength):
    if len(layers) < 2:
        return -1, -1

    # Calculate tranverse wave vectors
    k_0 = 2 * np.pi / wavelength
    n_inc = np.sqrt(layers[0][0] * layers[0][1]) if len(layers) == 3 else layers[0][0] # sqrt (epsilon & myu)
    k_x = k_0 * n_inc * np.sin(theta) * np.cos(phi)
    k_y = k_0 * n_inc * np.sin(theta) * np.sin(phi)
    k_z = k_0 * n_inc * np.cos(theta)
    k_inc = np.matrix([[k_x], [k_y], [k_z]])
    surface_normal = np.matrix([[0],[0],[1]])

    # Calculate polarization vector
    x = np.cross(surface_normal.T, k_inc.T).T
    ate = np.matrix([[0], [1], [0]]) if theta == 0 else (x / np.linalg.norm(x))
    x = np.cross(k_inc.T, ate.T).T
    atm = x / np.linalg.norm(x)
    
    P = pte * ate + ptm * atm
    P = P / np.linalg.norm(P) # Normalize so magnitude is one

    # Calculate gap medium parameters
    Q_gap = np.matrix([
        [k_x * k_y, 1 + k_y ** 2],
        [-(1 + k_x ** 2), -k_x * k_y]
    ])
    
    V_gap = -1j * Q_gap

    # Scattering matrix structure
    # S11 S12
    # S21 S22

    # Initialize global scattering matrix
    S_11, S_12, S_21, S_22 = 0, 1, 1, 0 

    # Iterate
    for layer in layers[1:-1]:
        V, omega, L = calculate_parameters_for_layer(layer, k_x, k_y)
        S_i = calculate_s_matrix(V, omega, L, k_0, V_gap)
        S_11, S_12, S_21, S_22 = redheffer_product([S_11, S_12, S_21, S_22], S_i)

    # E_ref and E_src calculations
    e_src = np.matrix([[P[0,0]], [P[1,0]]])
                  
    [[E_ref_x], [E_ref_y]] = S_11 * e_src
    [[E_trn_x], [E_trn_y]] = S_21 * e_src
    [[E_ref_prime_x], [E_ref_prime_y]] = S_22 * e_src
    [[E_trn_prime_x], [E_trn_prime_y]] = S_12 * e_src
    
    E_ref_z = -1 * (k_x * E_ref_x + k_y * E_ref_y) / k_z
    E_trn_z = -1 * (k_x * E_trn_x + k_y * E_trn_y) / k_z
    E_ref_prime_z = -1 * (k_x * E_ref_prime_x + k_y * E_ref_prime_y) / k_z
    E_trn_prime_z = -1 * (k_x * E_trn_prime_x + k_y * E_trn_prime_y) / k_z
    
    R = np.linalg.norm(np.matrix([[E_ref_x[0,0]], [E_ref_y[0,0]], [E_ref_z[0,0]]]) ) ** 2
    T = np.linalg.norm(np.matrix([[E_trn_x[0,0]], [E_trn_y[0,0]], [E_trn_z[0,0]]]) ) ** 2
    #R_prime = np.linalg.norm(np.matrix([[E_ref_prime_x[0,0]], [E_ref_prime_y[0,0]], [E_ref_prime_z[0,0]]]) ) ** 2
    #T_prime = np.linalg.norm(np.matrix([[E_trn_prime_x[0,0]], [E_trn_prime_y[0,0]], [E_trn_prime_z[0,0]]]) ) ** 2

    return R, T


# Example usage
if __name__ == "__main__":
    # There are two accepted formats for layer input
    #
    # A layer can be described with its permittivity and permeability
    #   layer = [epsilon, myu, thickness]
    #
    # or it can be represented with it's refractive index like so:
    #   layer = [refractive index, thickness]
    # 
    # Thickness is assumed to be semi-infinite for the reflection and transmission layers
    # As a result we leave their values as 1

    layers = [
        [1.0, 1.0, None], # reflection layer
        [2.5, 2.0, 5e-4], # epsilon, myu, thickness
        [1.5     , 2e-4], # refractive index, thickness
        [2.5, 2.0, 5e-4], # epsilon, myu, thickness
        [2.5     , 2e-4], # refractive index, thickness
        [2.5, 1.0, 5e-4], # epsilon, myu, thickness
        [0.5, 1.5, 2e-4], # epsilon, myu, thickness
        [1.0, 1.0, None], # transmission layer
    ] 

    theta = 0
    phi = 0
    pte = 0.5
    ptm = 0.5
    wavelength = 1e-6

    R, T = tmm_solve(layers, theta, phi, pte, ptm, wavelength)

    print(R, T)