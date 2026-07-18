import numpy as np
import matplotlib.pyplot as plt

a = 100


def deriv_fn(state):

    x, y, px, py = state

    r = np.sqrt(x**2 + y**2)

    if r <= a:
        n = np.sqrt(2 - (r/a)**2)
        dndx = -x/(a**2 * n)
        dndy = -y/(a**2 * n)
    else:
        n = 1.0
        dndx = 0.0
        dndy = 0.0

    return np.array([
        px/n,
        py/n,
        dndx,
        dndy
    ])


def my_rk4(state, deriv_fn, dt):

    k1 = deriv_fn(state)
    k2 = deriv_fn(state + 0.5*dt*k1)
    k3 = deriv_fn(state + 0.5*dt*k2)
    k4 = deriv_fn(state + dt*k3)

    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


dt = 0.05
num_steps = 6000

plt.figure(figsize=(10, 6))

for y0 in np.linspace(-80, 80, 15):

    x0 = -150

    px0 = 1.0
    py0 = 0.0

    state = np.array([x0, y0, px0, py0])

    x_vals = []
    y_vals = []

    for i in range(num_steps):

        x, y, px, py = state

        x_vals.append(x)
        y_vals.append(y)

        state = my_rk4(state, deriv_fn, dt)

    plt.plot(x_vals, y_vals, color="black")

# Lens boundary


plt.xlabel("x")
plt.ylabel("y")
plt.title("Luneburg Lens")
plt.axis("equal")
plt.grid(True)

plt.show()
