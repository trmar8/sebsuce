import numpy as np
import matplotlib.pyplot as plt
from splines_edo_implicite import splines_edo_implicite
from rk4 import rk4


#pvi du seconde ordre
t0, tf = np.pi/4, 3
alpha = (np.sqrt(2)/2) * np.exp(np.pi/4)
beta = np.sqrt(2) * np.exp(np.pi/4)

def solution(t):
    return np.exp(t) * np.sin(t)

def spline(t, y, dy):
    return 2*dy - 2*np.exp(t)*np.sin(t)

def spline_aux_noeuds(N):
    h = (tf - t0) / N
    coefficient = splines_edo_implicite(alpha, beta, spline, t0, tf, N)

    t_noeuds = np.linspace(t0, tf, N+1)
    y_approx = np.zeros(N+1)
    y_approx[0] = alpha

    for i in range(N):
        a, b, c, d = coefficient[i]
        t_suivant = t_noeuds[i+1]
        y_approx[i+1] = a*t_suivant**3 + b*t_suivant**2 + c*t_suivant + d

    return t_noeuds, y_approx, h


t_noeuds_16, y_spline_16, h_16 = spline_aux_noeuds(16)
t_lisse = np.linspace(t0, tf, 200)

plt.figure(1)
plt.plot(t_lisse, solution(t_lisse), 'k-', label='Solution exacte')
plt.plot(t_noeuds_16, y_spline_16, 'ro--', label='Spline cubique (N=16)')
plt.xlabel('Temps t')
plt.ylabel('y(t)')
plt.title('Figure 1: Approximation Spline Cubique vs Solution Exacte')
plt.legend()
plt.grid(True)


liste_N = [2**i for i in range(6, 11)]
liste_h = []
liste_erreur = []

for N in liste_N:
    t_noeudsN, y_approx, h = spline_aux_noeuds(N)
    
    y_exact = solution(t_noeudsN)
    
    erreur_globale = np.max(np.abs(y_exact - y_approx))

    liste_h.append(h)
    liste_erreur.append(erreur_globale)

pente = (np.log(liste_erreur[-1]) - np.log(liste_erreur[0])) / (np.log(liste_h[-1]) - np.log(liste_h[0]))
print(pente)

plt.figure(2)
plt.loglog(liste_h, liste_erreur, 'o-', label='Erreur globale $E(h)$')
plt.xlabel('Pas h')
plt.ylabel('Erreur Globale E(h)')
plt.title("Figure 2: Convergence de l'erreur en fonction du pas h")
plt.legend()
plt.grid(True)

def systeme(t, Y):
    y = Y[0]
    dy = Y[1]

    y_prime = dy
    dy_prime = 2*dy - 2*np.exp(t)*np.sin(t)

    return np.array([y_prime, dy_prime])

h_16_rk4 = (tf - t0) / 16
Y0 = [alpha, beta]

ti, Yi = rk4(systeme, t0, tf, Y0, h_16)
y_rk4_16 = Yi[0, :]

plt.figure(3)
plt.plot(t_lisse, solution(t_lisse), 'k-', label='Solution exacte')
plt.plot(t_noeuds_16, y_spline_16, 'ro--', label='Spline cubique (N=16)')
plt.plot(ti, y_rk4_16, 'bx:', label='RK4 (N=16)', markersize=8)
plt.xlabel('Temps t')
plt.ylabel('y(t)')
plt.title('Figure 3 : Comparaison des méthodes numériques (N=16)')
plt.legend()
plt.grid(True)

plt.show()