from initialize import getInitialConditions
import constants as c
import numeric as num
import numpy as np
import scipy.integrate as odeint
import plot


def propagate():

    x, p = getInitialConditions(c.xmethod)

    if len(x) != c.N:
        raise ValueError(
                "length of x is not equal to the number of progpagated points"
                )

    initialConditions = np.concatenate([x, p], axis=0)

    # initialise stepping object
    stepper = odeint.RK45(
            lambda t, y: num.equation_of_motion(t, y),
            c.ts, initialConditions, c.tf,
            max_step=c.maxstep, rtol=c.rtol, atol=c.atol
            )

    x = stepper.y[:c.N]
    p = stepper.y[c.N:]

    H = num.Hamiltonian(x, p)

    # array to store trajectories
    trajectory = []
    trajectory.append([stepper.t] + stepper.y.tolist() + H.tolist())
    maxstep = 0.
    countstep = 0

    # propragate Hamilton's eqn's
    while stepper.t < c.tf:
        try:

            stepper.step()

            x = stepper.y[:c.N]
            p = stepper.y[c.N:]
            H = num.Hamiltonian(x, p)

            # force small step for close-encounters
#            if Rmax < 10.:
#                stepper.max_step = 1e-1
#            else:
#                stepper.max_step = c.maxstep

            trajectory.append([stepper.t] + stepper.y.tolist() + H.tolist())

            maxstep = np.max([stepper.step_size, maxstep])
            countstep = countstep + 1

        except RuntimeError as e:
            print(e)
            break

    trajectory = np.array(trajectory)

    return trajectory


if __name__ == "__main__":
    traj = propagate()
    plot.plot3Danim(data=traj)
