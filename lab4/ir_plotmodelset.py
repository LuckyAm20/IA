import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scaler import SCALE

# SCALE = 1/16384


def local_print_intervals(ys_int, ys_ext, Xs_lvls, actually_draw=True):
    ys_int = ys_int * (SCALE) - 0.5
    ys_ext = ys_ext * (SCALE) - 0.5
    ys_int_to_plot = [np.average(i) for i in ys_int]
    ys_ext_to_plot = [np.average(i) for i in ys_ext]

    def gen_yi1(ys_int_to_plot):
        return np.abs(ys_int[:, 0] - ys_int_to_plot)

    def gen_yi2(ys_int_to_plot):
        return np.abs(ys_int[:, 1] - ys_int_to_plot)

    def gen_ye1(ys_ext_to_plot):
        return np.abs(ys_ext[:, 0] - ys_ext_to_plot)

    def gen_ye2(ys_ext_to_plot):
        return np.abs(ys_ext[:, 1] - ys_ext_to_plot)

    yerr_int = [
        gen_yi1(ys_int_to_plot),
        gen_yi2(ys_int_to_plot)
    ]
    yerr_ext = [
        gen_ye1(ys_ext_to_plot),
        gen_ye2(ys_ext_to_plot)
    ]
    
    if actually_draw:
        plt.title('Оценки')
        plt.errorbar(Xs_lvls, ys_int_to_plot, yerr=yerr_int, marker=".", linestyle='none',
                    ecolor='k', elinewidth=0.8, capsize=4, capthick=1)
        plt.errorbar(Xs_lvls, ys_ext_to_plot, yerr=yerr_ext, linestyle='none',
                    ecolor='r', elinewidth=0.8, capsize=4, capthick=1)
        plt.show()
    else:
        plt.errorbar(Xs_lvls, ys_int_to_plot, yerr=yerr_int, marker=".", linestyle='none',
                    ecolor='k', elinewidth=0.8, capsize=4, capthick=1)
        plt.errorbar(Xs_lvls, ys_ext_to_plot, yerr=yerr_ext, linestyle='none',
                    ecolor='r', elinewidth=0.8, capsize=4, capthick=1)


def ir_plotmodelset(irproblems, xlimits=None, ys_int=None, ys_ext=None, Xs_lvls=None):
    colors = ['#9b59b6', '#3498db', '#3498db', '#f1c40f', '#2ecc71', '#34495e', '#e74c3c', '#1abc9c']

    colors.reverse()
    if len(irproblems) > len(colors):
        raise ValueError(f"Max len of irproblems is {len(colors)}")

    for i in range(len(irproblems)):
        irproblem = irproblems[i]

        X = irproblem['X']

        if X.shape[0] < 2:
            raise ValueError("Not enough data")
        if X.shape[1] == 2 and np.all(X[:, 0] == 1):
            xcol = 1
        elif X.shape[1] == 2 and np.all(X[:, 1] == 1):
            xcol = 0
        elif X.shape[1] == 1:
            xcol = 0
        else:
            raise ValueError("Wrong X size")

        if xlimits is None:
            Xbefore = 2 * X[0, :] - X[1, :]
            Xafter = 2 * X[-1, :] - X[-2, :]
        else:
            Xbefore = X[0, :].copy()
            Xbefore[xcol] = xlimits[0]
            Xafter = X[0, :].copy()
            Xafter[xcol] = xlimits[1]

        Xp = np.vstack((Xbefore, X, Xafter))

        x = Xp[:, xcol]

        yp, betap, exitcode, active = ir_predict(irproblem, Xp)

        px = np.concatenate((x, x[::-1]))
        py = np.concatenate((yp[:, 0], yp[:, 1][::-1]))

        plt.fill(px, py, color=colors[i], alpha=0.5)
        plt.plot(x, yp[:, 0], color=colors[i], linewidth=1)
        plt.plot(x, yp[:, 1], color=colors[i], linewidth=1)
    if ys_int is not None:
        local_print_intervals(ys_int, ys_ext, Xs_lvls, actually_draw=False)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('IR Model Set Plot')
    # plt.xlim(-0.05, 0.025)
    # plt.ylim(-0.04, 0.02)

    plt.show()


def ir_predict(irproblem, Xp):

    X = np.array(irproblem['X'])
    y = np.array(irproblem['y'])
    epsilon = irproblem['epsilon']
    C = np.array(irproblem['C']) if 'C' in irproblem else np.array([])
    d = np.array(irproblem['d']) if 'd' in irproblem else np.array([])
    ctype = irproblem['ctype'] if 'ctype' in irproblem else []

    lb = irproblem.get('lb', -np.inf * np.ones(X.shape[1]))
    ub = irproblem.get('ub', np.inf * np.ones(X.shape[1]))

    if Xp.shape[1] != X.shape[1]:
        raise ValueError("Xp must have the same number of columns as X")

    if C.size != 0:
        A = np.vstack((X, -X, C))
    else:
        A = np.vstack((X, -X))

    if d.size != 0:
        b = np.concatenate((y + epsilon, -y + epsilon, d))
    else:
        b = np.concatenate((y + epsilon, -y + epsilon))

    k = Xp.shape[0]
    n, m = X.shape

    ctype_full = ['U'] * (2 * n) + list(ctype)
    vartype = ['C'] * m
    sense = 1

    SIGNIFICANT = 1e-7

    yp = np.zeros((k, 2))
    betaplow = np.zeros_like(Xp)
    betaphigh = np.zeros_like(Xp)
    active = []

    for i in range(k):
        result_low = linprog(c=Xp[i, :], A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method='highs')
        betalow = result_low.x
        flow = result_low.fun
        actlow = np.where(np.abs(result_low.get('lambda', np.zeros(A.shape[0]))) > SIGNIFICANT)[0]

        result_high = linprog(c=-Xp[i, :], A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method='highs')
        betahigh = result_high.x
        fhigh = result_high.fun

        # print("fhigh: ", fhigh)

        actupp = np.where(np.abs(result_high.get('lambda', np.zeros(A.shape[0]))) > SIGNIFICANT)[0]

        yp[i, :] = [flow, -fhigh]
        betaplow[i, :] = np.minimum(betalow, betahigh)
        betaphigh[i, :] = np.maximum(betalow, betahigh)

        active.append({'lower': actlow, 'upper': actupp})

    betap = np.stack((betaplow, betaphigh), axis=-1)

    return yp, betap, 0, active
