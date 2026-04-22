import numpy as np
import matplotlib.pyplot as plt
from splines_edo_implicite import splines_edo_implicite
from rk4 import rk4


# Paramètres du problème
t0, tf = np.pi/4, 3
alpha = (np.sqrt(2)/2) * np.exp(np.pi/4)
beta = np.sqrt(2) * np.exp(np.pi/4)

def exact_sol(t):
    return np.exp(t) * np.sin(t)

# Selon la solution exacte, la fonction f est déduite comme suit
f_spline = lambda t, y, dy: 2*dy - 2*np.exp(t)*np.sin(t)

# --- FIGURE 1 : Splines Cubiques (N=16) ---
N_16 = 16
coeffs_16 = splines_edo_implicite(alpha, beta, f_spline, t0, tf, N_16)

t_eval = np.linspace(t0, tf, 500)
y_spline = np.zeros_like(t_eval)
h = (tf - t0) / N_16

for j, t in enumerate(t_eval):
    idx = min(int((t - t0) / h), N_16 - 1)
    C = coeffs_16[idx]
    y_spline[j] = C[0]*t**3 + C[1]*t**2 + C[2]*t + C[3]

plt.figure(1)
plt.plot(t_eval, exact_sol(t_eval), 'k-', label='Solution exacte', linewidth=2)
plt.plot(t_eval, y_spline, 'r--', label='Spline cubique (N=16)')
plt.xlabel('Temps t')
plt.ylabel('y(t)')
plt.title('Approximation par Spline Cubique vs Solution Exacte')
plt.legend()
plt.grid(True)

# --- FIGURE 2 : Analyse de Convergence Globale ---
N_vals = [2**i for i in range(6, 11)]
erreurs = []
h_vals = []

for N in N_vals:
    h_temp = (tf - t0) / N
    h_vals.append(h_temp)
    coeffs_temp = splines_edo_implicite(alpha, beta, f_spline, t0, tf, N)
    
    # Évaluation de l'erreur globale
    err_max = 0
    t_nodes = np.linspace(t0, tf, N+1)
    for i, t_node in enumerate(t_nodes):
        if i == N: idx = N - 1
        else: idx = i
        C = coeffs_temp[idx]
        y_approx = C[0]*t_node**3 + C[1]*t_node**2 + C[2]*t_node + C[3]
        err_max = max(err_max, abs(exact_sol(t_node) - y_approx))
    erreurs.append(err_max)

plt.figure(2)
plt.loglog(h_vals, erreurs, 'o-', markerfacecolor='orange')
plt.xlabel('Taille du pas h')
plt.ylabel('Erreur Globale E(h)')
plt.title('Convergence de la Spline Cubique')
plt.grid(True, which="both", ls="--")

# La pente du graphique log-log te donnera l'ordre de convergence (O(h^4) typiquement pour les splines cubiques).

# --- FIGURE 3 : Comparaison avec RK4 ---
def f_rk4(t, u):
    # u[0] = y, u[1] = y'
    return np.array([u[1], 2*u[1] - 2*np.exp(t)*np.sin(t)])

t_rk4, u_rk4 = rk4(f_rk4, t0, tf, [alpha, beta], h)
y_rk4 = u_rk4[0, :]

plt.figure(3)
plt.plot(t_eval, exact_sol(t_eval), 'k-', label='Solution exacte', linewidth=2)
plt.plot(t_eval, y_spline, 'r--', label='Spline cubique (N=16)')
plt.plot(t_rk4, y_rk4, 'bo', label='RK4 (N=16)', markersize=4)
plt.xlabel('Temps t')
plt.ylabel('y(t)')
plt.title('Comparaison Spline Cubique et RK4')
plt.legend()
plt.grid(True)

plt.show()