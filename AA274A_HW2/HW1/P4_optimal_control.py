import typing as T

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from scipy.optimize import minimize, Bounds  # type: ignore
from utils import save_dict, maybe_makedirs

N = 20  # Number of time discretization nodes (0, 1, ... N).
s_dim = 3  # State dimension; 3 for (x, y, th).
u_dim = 2  # Control dimension; 2 for (V, om).
v_max = 0.5  # Maximum linear velocity.
om_max = 1.0  # Maximum angular velocity.

s_0 = np.array([0, 0, -np.pi / 2])  # Initial state.
s_f = np.array([5, 5, -np.pi / 2])  # Final state.


def pack_decision_variables(t_f: float, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Packs decision variables (final time, states, controls) into a 1D vector.

    Args:
        t_f: Final time, a scalar.
        s: States, an array of shape (N + 1, s_dim).
        u: Controls, an array of shape (N, u_dim).

    Returns:
        An array `z` of shape (1 + (N + 1) * s_dim + N * u_dim,).
    """
    return np.concatenate([[t_f], s.ravel(), u.ravel()])


def unpack_decision_variables(z: np.ndarray) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Unpacks a 1D vector into decision variables (final time, states, controls).

    Args:
        z: An array of shape (1 + (N + 1) * s_dim + N * u_dim,).

    Returns:
        t_f: Final time, a scalar.
        s: States, an array of shape (N + 1, s_dim).
        u: Controls, an array of shape (N, u_dim).
    """
    t_f = float(z[0])
    s = z[1:1 + (N + 1) * s_dim].reshape(N + 1, s_dim)
    u = z[-N * u_dim:].reshape(N, u_dim)
    return t_f, s, u


def objective(z, time_weight):
    t_f, s, u = unpack_decision_variables(z)
    
    N = 20
    dt = t_f/N

    s_0 = np.array([0, 0, -np.pi / 2])  # Initial state.
    s_f = np.array([5, 5, -np.pi / 2])  # Final state.

    obj = dt * np.sum(time_weight + np.square(u[:,0]) + np.square(u[:,1]))

    return obj


def constraint1(z):
    t_f, s, u = unpack_decision_variables(z)

    N = 20

    s_0 = np.array([0, 0, -np.pi / 2])  # Initial state.
    s_f = np.array([5, 5, -np.pi / 2])  # Final state.

    constraint_list = [s[0] - s_0, s[-1] - s_f]

    dt = t_f/N
    
    for i in range(N):
        constraint_list.append(s[i+1,:] - (s[i,:] + dt * np.array([u[i,0]*np.cos(s[i,2]), u[i,0]*np.sin(s[i,2]), u[i,1]])))
    return np.concatenate(constraint_list)


def constraint2(z):
    t_f, s, u = unpack_decision_variables(z)

    N = 20

    constraint_list = []

    V_max = 0.5
    om_max = 1.0

    for i in range(N):
        constraint_list.append([V_max - u[i,0]])
        constraint_list.append([u[i,0] + V_max])
        constraint_list.append([om_max - u[i,1]])
        constraint_list.append([u[i,1] + om_max])

    return np.concatenate(constraint_list)


def optimize_trajectory(
    time_weight: float = 1.0,
    verbose: bool = True
) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Computes the optimal trajectory as a function of `time_weight`.

    Args:
        time_weight: \lambda in the HW writeup.

    Returns:
        t_f_opt: Optimal final time, a scalar.
        s_opt: Optimal states, an array of shape (N + 1, s_dim).
        u_opt: Optimal controls, an array of shape (N, u_dim).
    """

    # NOTE: When using `minimize`, you may find the utilities
    # `pack_decision_variables` and `unpack_decision_variables` useful.

    # WRITE YOUR CODE BELOW ###################################################

    N = 20
    s_0 = np.array([0, 0, -np.pi / 2])  # Initial state.
    s_f = np.array([5, 5, -np.pi / 2])  # Final state.
    s_dim = 3
    u_dim = 2
    V_max = 0.5
    om_max = 1.0
    
    z_guess = np.concatenate([[20], np.linspace(s_0, s_f, N+1).ravel(), 0.25*np.ones((N, u_dim)).ravel()])

    con1 = {'type': 'eq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}

    cons = [con1, con2]

    result = minimize(objective, z_guess, args=(time_weight), constraints=cons, options={'maxiter': 1000})

    t_f_opt, s_opt, u_opt = unpack_decision_variables(result.x)
    print(t_f_opt)
    return t_f_opt, s_opt, u_opt

    ###########################################################################


if __name__ == '__main__':
    for time_weight in (1.0, 0.2):
        t_f, s, u = optimize_trajectory(time_weight)
        V = u[:, 0]
        om = u[:, 1]
        t = np.linspace(0, t_f, N + 1)[:-1]
        x = s[:, 0]
        y = s[:, 1]
        th = s[:, 2]
        data = {'t_f': t_f, 's': s, 'u': u}
        save_dict(data, f'data/optimal_control_{time_weight}.pkl')
        maybe_makedirs('plots')

        # plotting
        # plt.rc('font', weight='bold', size=16)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'k-', linewidth=2)
        plt.quiver(x, y, np.cos(th), np.sin(th))
        plt.grid(True)
        plt.plot(0, 0, 'go', markerfacecolor='green', markersize=15)
        plt.plot(5, 5, 'ro', markerfacecolor='red', markersize=15)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis([-1, 6, -1, 6])
        plt.title(f'Optimal Control Trajectory (lambda = {time_weight})')

        plt.subplot(1, 2, 2)
        plt.plot(t, V, linewidth=2)
        plt.plot(t, om, linewidth=2)
        plt.grid(True)
        plt.xlabel('Time [s]')
        plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
        plt.title(f'Optimal control sequence (lambda = {time_weight})')
        plt.tight_layout()
        plt.savefig(f'plots/optimal_control_{time_weight}.png')
        plt.show()
