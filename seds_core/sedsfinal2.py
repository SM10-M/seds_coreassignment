import numpy as np
import matplotlib.pyplot as plt

n0 = 1.5
a = 100


def deriv_fn(state):

    x, y, px, py = state

    r = np.sqrt(x**2 + y**2)

    n = n0 / (1 + (r/a)**2)

    return np.array([
        px / n,
        py / n,
        -(2 * n0 * x) / (a**2 * (1 + (r/a)**2)**2),
        -(2 * n0 * y) / (a**2 * (1 + (r/a)**2)**2)
    ])


def my_rk4(state, deriv_fn, dt):

    k1 = deriv_fn(state)
    k2 = deriv_fn(state + 0.5 * dt * k1)
    k3 = deriv_fn(state + 0.5 * dt * k2)
    k4 = deriv_fn(state + dt * k3)

    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------
# Multiple Rays
# ---------------------------------

plt.figure(figsize=(8, 8))

x0 = -100
y0 = 0

angles = np.linspace(-35, 35, 15)

dt = 0.1
num_steps = 2500

for theta in angles:

    theta = np.radians(theta)

    r0 = np.sqrt(x0**2 + y0**2)
    n_init = n0/(1 + (r0/a)**2)

    px0 = n_init*np.cos(theta)
    py0 = n_init*np.sin(theta)

    state = np.array([x0, y0, px0, py0])

    x_vals = []
    y_vals = []

    for i in range(num_steps):

        x, y, px, py = state

        x_vals.append(x)
        y_vals.append(y)

        state = my_rk4(state, deriv_fn, dt)

    plt.plot(x_vals, y_vals, linewidth=1)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Maxwell Fish-Eye Lens")
plt.axis("equal")
plt.grid(True)
plt.show()

L_vals = []

for i in range(num_steps):

    x, y, px, py = state

    L = x*py - y*px
    L_vals.append(L)

    state = my_rk4(state, deriv_fn, dt)

plt.figure(figsize=(8, 4))
plt.plot(L_vals)
plt.xlabel("Step")
plt.ylabel("Bouguer's Invariant (L)")
plt.title("Conservation of Bouguer's Invariant")
plt.grid(True)
plt.show()
