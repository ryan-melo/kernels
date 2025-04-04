import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import expm

def regularized_laplacian(L_n, sigma):
    I = np.eye(L_n.shape[0])
    return np.linalg.inv(I + sigma**2 * L_n)

def diffusion_process(L_n, sigma):
    return expm(- (sigma**2 / 2) * L_n)

def p_step_random_walk(L_n, a, p):
    I = np.eye(L_n.shape[0])
    return np.linalg.matrix_power(a * I - L_n, p)

def inverse_cosine(L_n):
    return np.cos(L_n * np.pi / 4)

G = nx.erdos_renyi_graph(5, 0.5)  # Grafo aleatório com 5 nós e probabilidade de conexão 0.5

# laplaciana normalizada
L = nx.normalized_laplacian_matrix(G).toarray()

# Parâmetros
sigma = 1.0
a = 2
p = 3

K_reg = regularized_laplacian(L, sigma)
K_diff = diffusion_process(L, sigma)
K_walk = p_step_random_walk(L, a, p)
K_cos = inverse_cosine(L)

# visualização
df_K_reg = pd.DataFrame(K_reg, index=G.nodes, columns=G.nodes)
df_K_diff = pd.DataFrame(K_diff, index=G.nodes, columns=G.nodes)
df_K_walk = pd.DataFrame(K_walk, index=G.nodes, columns=G.nodes)
df_K_cos = pd.DataFrame(K_cos, index=G.nodes, columns=G.nodes)

print("Regularized Laplacian Kernel:\n", df_K_reg)
print("\nDiffusion Process Kernel:\n", df_K_diff)
print("\np-Step Random Walk Kernel:\n", df_K_walk)
print("\nInverse Cosine Kernel:\n", df_K_cos)
