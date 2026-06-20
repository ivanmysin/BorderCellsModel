import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import erf
import matplotlib.pyplot as plt

# Параметры нейронов LIF
tau_m   = 20e-3       # с
V_th    = 20e-3       # В
V_r     = 10e-3       # В
tau_ref = 2e-3        # с
mu0     = 15e-3       # В, внешний ток (ниже порога)
sigma   = 5e-3        # В, амплитуда шума
w       = 0.05e-3     # В/Гц, сила обратной связи

# Параметры симуляции
N       = 2000        # число нейронов
dt      = 0.1e-3      # шаг по времени
T       = 2.0         # длительность (с)
tau_smooth = 5e-3     # окно сглаживания

# Формула Зигерта (точная Phi для LIF + белый шум)
def phi(mu, sigma):
    """Средняя частота популяции LIF в диффузионном приближении"""
    if sigma == 0:
        if mu <= V_th:
            return 0.0
        return 1.0 / (tau_ref + tau_m * np.log((mu - V_r) / (mu - V_th)))
    x_th = (V_th - mu) / sigma
    x_r  = (V_r - mu) / sigma

    def integrand(u):
        return np.sqrt(np.pi) * np.exp(u**2) * (1 + erf(u))

    integral, _ = quad(integrand, x_r, x_th)
    return 1.0 / (tau_ref + tau_m * integral)

# Модель Вильсона–Кована с сигмоидой Зигерта
tau_r = tau_m

def wc_ode(t, y):  # <-- принимаем y как массив
    r = y[0]       # извлекаем скаляр
    mu = mu0 + w * r
    return (-r + phi(mu, sigma)) / tau_r

# Решаем ОДУ (меньше точек для скорости)
t_eval = np.arange(0, T, 1e-3)  # 1 мс разрешение для графика
sol = solve_ivp(wc_ode, [0, T], [0.0], t_eval=t_eval, method='RK45', rtol=1e-6)

# Микроскопическая симуляция популяции LIF
t_full = np.arange(0, T, dt)
n_steps = len(t_full)

V = np.ones(N) * V_r
last_spike = -np.inf * np.ones(N)

A_smooth = 0.0
A_history = np.zeros(n_steps)
decay = np.exp(-dt / tau_smooth)

for step in range(n_steps):
    time = t_full[step]
    mu = mu0 + w * A_smooth

    in_ref = (time - last_spike) < tau_ref
    dW = np.sqrt(dt) * np.random.randn(N)
    V += (-V + mu) / tau_m * dt + sigma * dW
    V[in_ref] = V_r

    spikes = V >= V_th
    num_spikes = np.sum(spikes)
    if num_spikes > 0:
        V[spikes] = V_r
        last_spike[spikes] = time

    inst_rate = num_spikes / (N * dt)
    A_smooth = A_smooth * decay + (1 - decay) * inst_rate
    A_history[step] = A_smooth

# Графики
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_full, A_history, lw=0.8, label='LIF-популяция (сглаженная A(t))')
plt.plot(sol.t, sol.y[0], 'r--', lw=2, label='W-C + формула Зигерта')
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц)')
plt.legend()
plt.title('Динамика популяционной активности')

plt.subplot(2, 1, 2)
mu_range = np.linspace(10e-3, 30e-3, 100)
phi_vals = [phi(mu, sigma) for mu in mu_range]
plt.plot(np.array(mu_range)*1000, phi_vals, 'k-', label='Φ(μ, σ)')

mask = t_full > (T - 0.5)
avg_rate = np.mean(A_history[mask])
avg_mu   = mu0 + w * avg_rate
plt.plot(avg_mu*1000, avg_rate, 'bo', markersize=8, label='Микро (стац. точка)')
plt.xlabel('Средний вход μ (мВ)')
plt.ylabel('Частота (Гц)')
plt.legend()
plt.tight_layout()
plt.show()