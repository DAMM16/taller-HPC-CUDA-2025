import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# def flujo_godunov(qL, qR):
#     """
#     Flujo de Godunov exacto para Burgers:
#       f(q)=q^2/2
#     usando el estado interface q* del Riemann exacto.
#     """
#     # Selección de q*
#     if qL <= qR:
#         if qL <= 0.0 <= qR:
#             q_star = 0.0
#         elif qL > 0.0:
#             q_star = qL
#         else:
#             q_star = qR
#     else:               
#         if 0.5 * (qL + qR) > 0.0:
#             q_star = qL
#         else:
#             q_star = qR

#     return 0.5 * q_star**2


# def metodo_step_burgers(Q, dt, dx, bc="periodic"):
#     """
#     Un paso FV de Godunov (orden 1) para Burgers.
#     Q: array (N,)
#     bc: 'periodic' o 'outflow'
#     """
#     N = Q.size
#     lam = dt / dx

#     # Ghost cells
#     Qg = np.empty(N + 2)
#     Qg[1:-1] = Q

#     if bc == "periodic":
#         Qg[0]  = Q[-1]
#         Qg[-1] = Q[0]
#     elif bc == "outflow":
#         Qg[0]  = Q[0]
#         Qg[-1] = Q[-1]
#     else:
#         raise ValueError("bc debe ser 'periodic' o 'outflow'")

#     # Flujos en interfaces i+1/2 para i=0..N (sobre Qg)
#     # F[k] corresponde a F_{(k-1)+1/2} = F_{k-1/2} en notación de celda interior
#     F = np.zeros(N + 1)
#     for k in range(N + 1):
#         qL = Qg[k]
#         qR = Qg[k + 1]
#         F[k] = flujo_godunov(qL, qR)

#     # Update FV: Q_i^{n+1} = Q_i^n - lam (F_{i+1/2} - F_{i-1/2})
#     Qn = Qg[1:-1] - lam * (F[1:] - F[:-1])
#     return Qn


# def godunov_burgers(L=1.0, N=400, CFL=0.9, T=0.4, bc="periodic"):
#     x = np.linspace(0.0, L, N, endpoint=False)
#     dx = L / N

#     Q0 = np.exp(-((x - 0.25 * L) / (0.07 * L))**2)
#     # Q0 = 0.2 * (x < 0.4 * L) + 0.5 * (x > 0.6 * L)

#     # dt por CFL usando max|q| (velocidad característica = q)
#     max_speed0 = np.max(np.abs(Q0))
#     dt = CFL * dx / max(max_speed0, 1e-14)
#     nsteps = int(np.ceil(T / dt))
#     dt = T / nsteps  # ajustar para llegar exacto a T

#     Q = Q0.copy()
#     return x, Q0, Q, dx, dt, nsteps, bc


# L = 1.0
# N = 400
# CFL = 0.9
# T = 0.6
# bc = "outflow"

# x, Q0, Q, dx, dt, nsteps, bc = godunov_burgers(L=L, N=N, CFL=CFL, T=T, bc=bc)

# fig, ax = plt.subplots()
# ax.set_title("Burgers 1D - Godunov FVM")
# ax.set_xlabel("x")
# ax.set_ylabel("q(x,t)")
# ax.set_xlim(x.min(), x.max())
# ax.set_ylim(Q0.min() - 0.3, Q0.max() + 0.3)
# ax.grid(True, alpha=0.25)

# (line,) = ax.plot(x, Q, lw=2, label="Godunov")
# ax.plot(x, Q0, lw=1, alpha=0.7, label="Inicial")
# time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
# ax.legend(loc="upper right")

# # animación: pasos por frame
# steps_per_frame = max(1, nsteps // 300)

# state = {"Q": Q.copy(), "n": 0}

# def init():
#     line.set_ydata(state["Q"])
#     time_text.set_text("t = 0.000")
#     return line, time_text

# def update(_frame):
#     Qcur = state["Q"]
#     ncur = state["n"]

#     for _ in range(steps_per_frame):
#         if ncur >= nsteps:
#             break

#         # (opcional) dt adaptativo por CFL con el estado actual:
#         # max_speed = np.max(np.abs(Qcur))
#         # dt_eff = min(dt, CFL*dx/max(max_speed,1e-14))
#         # Qcur = metodo_step_burgers(Qcur, dt_eff, dx, bc=bc)

#         Qcur = metodo_step_burgers(Qcur, dt, dx, bc=bc)
#         ncur += 1

#     state["Q"] = Qcur
#     state["n"] = ncur

#     tcur = ncur * dt
#     cfl_now = (np.max(np.abs(Qcur)) * dt / dx) if np.max(np.abs(Qcur)) > 1e-14 else 0.0
#     line.set_ydata(Qcur)
#     time_text.set_text(f"t = {tcur:.3f}  |  CFL ≈ {cfl_now:.2f}")
#     return line, time_text

# anim = FuncAnimation(fig, update, init_func=init, interval=30, blit=True)
# plt.show()

# Métodos numéricos Eq. Burgers
# Mas directo sin tanta vuelta
def flujo_godunov(qL, qR):
    """
    Flujo de Godunov exacto para Burgers.
    Resuelve el Riemann exacto en la interfaz.
    """
    if qL <= qR:          # rarefacción
        if qL <= 0.0 <= qR:
            q_star = 0.0
        elif qL > 0.0:
            q_star = qL
        else:
            q_star = qR
    else:                # choque
        if 0.5 * (qL + qR) > 0.0:
            q_star = qL
        else:
            q_star = qR

    return 0.5 * q_star**2


def flujo_roe(qL, qR):
    """
    Flujo de Roe para la ecuación de Burgers:
        q_t + (q^2/2)_x = 0
    """
    # Flujo físico
    fL = 0.5 * qL**2
    fR = 0.5 * qR**2

    # Velocidad de Roe
    a_tilde = 0.5 * (qL + qR)

    # Flujo de Roe
    F = 0.5 * (fL + fR) - 0.5 * abs(a_tilde) * (qR - qL)

    return F

def flujo_hll(qL, qR):
    """
    Flujo HLL para Burgers:
        q_t + (q^2/2)_x = 0

    Para Burgers, una elección estándar de velocidades de onda es:
        S_L = min(qL, qR)
        S_R = max(qL, qR)
    porque la velocidad característica es f'(q)=q.
    """
    fL = 0.5 * qL**2
    fR = 0.5 * qR**2

    SL = min(qL, qR)
    SR = max(qL, qR)

    if 0.0 <= SL:
        return fL
    elif SR <= 0.0:
        return fR
    else:
        return (SR * fL - SL * fR + SL * SR * (qR - qL)) / (SR - SL)

def flujo_lax_friedrichs(qL, qR, alpha):
    """
    Flujo de Lax-Friedrichs para Burgers:
        q_t + (q^2/2)_x = 0
    alpha >= max |f'(q)| = |q|
    """
    fL = 0.5 * qL**2
    fR = 0.5 * qR**2
    return 0.5 * (fL + fR) - 0.5 * alpha * (qR - qL)



def metodo_step(Q, dt, dx, bc="periodic", flux_method="godunov"):
    N = Q.size
    lam = dt / dx

    # Ghost cells
    Qg = np.empty(N + 2)
    Qg[1:-1] = Q

    if bc == "periodic":
        Qg[0]  = Q[-1]
        Qg[-1] = Q[0]
    elif bc == "outflow":
        Qg[0]  = Q[0]
        Qg[-1] = Q[-1]
    else:
        raise ValueError("bc debe ser 'periodic' o 'outflow'")

    F = np.zeros(N + 1)
    for i in range(N + 1):
        if flux_method == "godunov":
            F[i] = flujo_godunov(Qg[i], Qg[i + 1])
        elif flux_method == "roe":
            F[i] = flujo_roe(Qg[i], Qg[i + 1])
        elif flux_method == "hll":
            F[i] = flujo_hll(Qg[i], Qg[i + 1])
        elif flux_method == "lax-friedrichs":
            alpha = np.max(np.abs(Q))  # estimación de max|q|
            F[i] = flujo_lax_friedrichs(Qg[i], Qg[i + 1], alpha)
        else:
            raise ValueError("flux_method debe ser 'godunov', 'roe', 'hll', o 'lax-friedrichs'")

    Qnew = Qg[1:-1] - lam * (F[1:] - F[:-1])
    return Qnew

L=1.0
N=300
CFL=0.8
T=0.2
bc="outflow"

x = np.linspace(0.0, L, N, endpoint=False)
dx = L / N

# Condición inicial
# Q0 = np.exp(-((x - 0.25 * L) / (0.07 * L))**2)
# Q0 += 0.5 * (x > 0.6 * L)
# Q0 = 1.0*(x <= 0.5 * L) + 0.2 * (x > 0.5 * L)
Q0 = 4.0*(x <= 0.3 * L) +0.1 * ((x > 0.3 * L) & (x <= 0.6 * L)) + 1.0 * (x > 0.6 * L)
# Q0 = 0.1*(x <= 0.3 * L) + 3.0 * ((x > 0.3 * L) & (x <= 0.6 * L)) - 2.0 * (x > 0.6 * L)


Q = Q0.copy()
t = 0.0
flux_method="godunov"  # 'godunov', 'roe', 'hll', 'lax-friedrichs'



plt.ion()
fig, ax = plt.subplots(figsize=(8,4))
line0, = ax.plot(x, Q0, "--", lw=1.5, label="Inicial")
line,  = ax.plot(x, Q,  lw=2, label=flux_method)
txt = ax.text(0.02, 0.92, "", transform=ax.transAxes)

ax.set_xlabel("x"); ax.set_ylabel("q(x,t)")
ax.set_title(f"Eq. Burgers - {flux_method}")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plot_every = 5    # actualiza cada 10 pasos (ajusta)
k = 0

while t < T:
    max_speed = np.max(np.abs(Q))
    dt = CFL * dx / max(max_speed, 1e-14)
    if t + dt > T:
        dt = T - t

    Q = metodo_step(Q, dt, dx, bc=bc, flux_method=flux_method)
    t = t + dt
    k = k + 1

    if k % plot_every == 0 or t >= T:
        line.set_ydata(Q)
        txt.set_text(f"t = {t:.3f}   CFL = {max_speed*dt/dx:.2f}")
        ax.relim()
        ax.autoscale_view(scaley=True)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

plt.ioff()
plt.show()

