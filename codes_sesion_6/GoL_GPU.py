import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

life_code = r"""
extern "C"
__global__ void LifeStep(const int *old_g, int *new_g, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // fila
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col
    if (i >= n || j >= n) return;

    int im1 = (i + n - 1) % n;
    int ip1 = (i + 1) % n;
    int jm1 = (j + n - 1) % n;
    int jp1 = (j + 1) % n;

    #define G(ii,jj) old_g[(ii)*n + (jj)]

    int alive = G(i,j);
    int neigh =
        G(im1,jm1) + G(im1,j) + G(im1,jp1) +
        G(i,  jm1)           + G(i,  jp1) +
        G(ip1,jm1) + G(ip1,j) + G(ip1,jp1);

    int out = 0;
    if (alive) {
        if (neigh == 2 || neigh == 3) {
            out = 1;   // sobrevive
        } else {
            out = 0;   // muere
        }
    } else {
        if (neigh == 3) {
            out = 1;   // nace
        } else {
            out = 0;   // sigue muerta
        }
    }

    new_g[i*n + j] = out;

    #undef G
}
"""
mod = SourceModule(life_code, options=["-std=c++11"])
LifeStep = mod.get_function("LifeStep")

# -----------------------------
# Parámetros y condición inicial
# -----------------------------
n = 256
T = 1000
p = 0.5

A0 = (np.random.rand(n, n) < p).astype(np.int32)
cond_ini = np.ascontiguousarray(A0).ravel()

old_gpu = drv.mem_alloc(cond_ini.nbytes)
new_gpu = drv.mem_alloc(cond_ini.nbytes)
drv.memcpy_htod(old_gpu, cond_ini)
drv.memcpy_htod(new_gpu, np.zeros_like(cond_ini))

block_size = (16, 16, 1)
grid_size  = (int(np.ceil(n / block_size[0])), int(np.ceil(n / block_size[1])), 1)

try:
    soluciones  = []
    times       = []
    sim_guardar = T
    modulo      = max(1, int(T / sim_guardar))
    time = 0

    for tt in range(1, T + 1):
        LifeStep(old_gpu, new_gpu, np.int32(n), block=block_size, grid=grid_size)
        old_gpu, new_gpu = new_gpu, old_gpu

        time += 1
        if time % modulo == 0:
            data = np.empty_like(cond_ini)
            drv.memcpy_dtoh(data, old_gpu)
            soluciones.append(data.reshape(n, n))
            times.append(tt)

finally:
    old_gpu.free()
    new_gpu.free()
    print("Memoria liberada")

# (Opcional) plot del estado final
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# soluciones: lista de matrices (n x n) que ya guardaste en el loop
# (si no las guardaste todas, asegúrate sim_guardar = T y modulo = 1)

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(soluciones[0], cmap="binary", interpolation="nearest", aspect="equal")
ax.axis("off")
title = ax.set_title(f"Conway's Game of Life (t = {times[0]})")

def update(k):
    im.set_data(soluciones[k])
    title.set_text(f"Conway's Game of Life (t = {times[k]})")
    return (im, title)

anim = FuncAnimation(fig, update, frames=len(soluciones), interval=50, blit=False)
plt.show()
