import numpy as np
import matplotlib.pyplot as plt


class TM:
    def __init__(self, Uinc, tau_f, tau_r, tau_d, dt=0.1, SF=1):

        self.Uinc = Uinc
        self.tau_f = tau_f
        self.tau_r = tau_r
        self.tau_d = tau_d

        self.dt = dt

        self.SF = SF

        # self.Uinc = self.Uinc * self.SF

        self.tau1r = np.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)

        self.exp_tau_d = np.exp(-self.dt / self.tau_d)
        self.exp_tau_f = np.exp(-self.dt / self.tau_f)
        self.exp_tau_r = np.exp(-self.dt / self.tau_r)

    def get_initial_state(self):


        R = 0
        U = 0
        A = 0

        initial_state = [R, U, A]

        return initial_state

    def call(self, t, state, FRpre):
        R, U, A = state


        FRpre_normed = FRpre * 0.001 * self.dt #  * self.SF


        a_ = A * self.exp_tau_d
        r_ = self.SF + (R - self.SF + self.tau1r * A) * self.exp_tau_r  - self.tau1r * A
        u_ = U * self.exp_tau_f

        U = u_ + self.Uinc * (1.0 - u_) * FRpre_normed
        A = a_ + U * r_ * FRpre_normed
        R = r_ - U * r_ * FRpre_normed # * self.SF

        return [R, U, A]
##########################
Uinc, tau_f, tau_r, tau_d = 0.238, 13.6, 7.88,783.2
dt = 0.1
T = 200
syn = TM(Uinc, tau_f, tau_r, tau_d, dt = dt, SF=1)

SF = 100.0
syn_scaled = TM(Uinc, tau_f, tau_r, tau_d, dt = dt, SF=SF)

state = syn.get_initial_state()
state_scaled = syn_scaled.get_initial_state()


t = np.arange(0, T, dt)

FRpre = 0.1*(0.5 * ( np.cos(2*np.pi*t*8*0.001) + 1 ))**6


states = []
states_scaled = []

for i in range(len(t)):
    state = syn.call(t[i], state, FRpre[i])
    states.append(state)

    state_scaled = syn_scaled.call(t[i], state_scaled, FRpre[i])
    states_scaled.append(state_scaled)


states = np.stack(states, axis=1)
states_scaled = np.stack(states_scaled, axis=1)
# states_scaled[0] /= SF   # R
# # states_scaled[1] /= SF   # U
# states_scaled[2] /= SF   # A — здесь нужно квадратичное масштабирование!

fig, ax = plt.subplots(3, 1, figsize=(10, 10))

for i in range(3):
    ax[i].plot(t, states[i], color='red', linewidth=5)
    ax[i].plot(t, states_scaled[i], color='blue', linewidth=2)
plt.show()


