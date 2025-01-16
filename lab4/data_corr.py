import numpy as np
np.float_ = np.float64
import intvalpy as ip
from intvalpy import mid, rad
from tolsolvty import tolsolvty
from ir_problem import ir_problem, ir_outer
from ir_plotmodelset import ir_plotmodelset
import matplotlib.pyplot as plt
from scaler import SCALE, RADIUS
ip.precision.extendedPrecisionQ = False




def plot_tol_sys(Xi, Ysint, Ysout, fname, title="Допусковое множество", ):
    print(f'Ysint = {Ysint}')
    print(f'Ysout = {Ysout}')
    vert1 = ip.IntLinIncR2(Xi, Ysint, consistency='tol', show=False)
    print(f'vert1 = {vert1}')
    vert = ip.IntLinIncR2(Xi, Ysout, consistency='tol', show=False)
    print(f'vert = {vert}')
    for ortant in range(len(vert)):
        if len(vert[ortant]) != 0:
            vert_x = []
            vert_y = []
            for x in vert[ortant]:
                if len(x) != 0:
                    vert_x.append(x[0])
                    vert_y.append(x[1])

            x_0 = vert_x[0]
            y_0 = vert_y[0]
            vert_x.append(x_0)
            vert_y.append(y_0)

            plt.scatter(vert_x, vert_y, color="darkgreen", marker=".")
            plt.fill(vert_x, vert_y, linestyle='-', linewidth=1, color="darkgreen", alpha=0.7)

    for ortant in range(len(vert1)):
        if len(vert1[ortant]) != 0:
            vert1_x = []
            vert1_y = []
            for x in vert1[ortant]:
                if len(x) != 0:
                    vert1_x.append(x[0])
                    vert1_y.append(x[1])
            x_0 = vert1_x[0]
            y_0 = vert1_y[0]
            vert1_x.append(x_0)
            vert1_y.append(y_0)
            plt.scatter(vert1_x, vert1_y, color='green', marker=".")
            plt.fill(vert1_x, vert1_y, linestyle='-', linewidth=1, color='green', alpha=0.7)

    plt.title(title)
    plt.xlabel("β₀")
    plt.ylabel("β₁")

    plt.savefig(f"{fname}")
    plt.show()


def data_corr_naive(Ysint, Ysout, Xi, graphics=False, ys_int=None, ys_ext=None, Xs_lvls=None):
    y = mid(Ysint)*(SCALE) - 0.5
    epsilon = rad(Ysint)*RADIUS

    if graphics:
        plot_tol_sys(Xi, Ysint * (SCALE) - 0.5, Ysout * (SCALE) - 0.5, "tol-before-alg")

    irp_DRSout = ir_problem(ip.inf(Xi), ip.mid(Ysout)*(SCALE) - 0.5, ip.rad(Ysout)*(RADIUS))

    tolmax, argmax, env, ccode = tolsolvty(ip.inf(Xi), ip.sup(Xi),
                                           ip.inf(y - epsilon).reshape(-1, 1), ip.sup(y + epsilon).reshape(-1, 1))
    print("\ntolmax: ", tolmax)
    print("\nargmax: ", argmax)
    print("\nenv: ", env)

    if tolmax > 0:
        print("\n!______tolmax > 0______!")
        

        print("\ntolmax: ", tolmax)
        print("\nargmax: ", argmax)
        print("\nenv: ", env)

        irp_DRSint = ir_problem(ip.inf(Xi), y, epsilon)

        if graphics:
            print('')
            ir_plotmodelset([irp_DRSout, irp_DRSint], ys_int=ys_int, ys_ext=ys_ext, Xs_lvls=Xs_lvls)

        b_int = ir_outer(irp_DRSint)
        return b_int, []  # indtoout = None

    print('!______tolmax < 0______!')

    envnegind = np.where(env[:, 1] < 0)[0]
    indtoout = env[envnegind, 0]

    for idx in indtoout:
        idx = int(idx-1)
        y[idx] = mid(Ysout[idx])*(SCALE) - 0.5
        epsilon[idx] = rad(Ysout[idx])*(RADIUS)

    if graphics:
        plot_tol_sys(Xi, ip.Interval((y - epsilon), (y + epsilon)), Ysout * (SCALE) - 0.5,
                     "tol-after-alg", "Внутренние оценки tolₘ < 0 -> внешние оценки")

    tolmax, argmax, env, ccode = tolsolvty(ip.inf(Xi), ip.sup(Xi),
                                           (y - epsilon).reshape(-1, 1), (y + epsilon).reshape(-1, 1))

    print("\ntolmax: ", tolmax)
    print("\nargmax: ", argmax)
    print("\nenv: ", env)

    irp_DRSint = ir_problem(ip.inf(Xi), y, epsilon)

    if graphics:
        ir_plotmodelset([irp_DRSout, irp_DRSint], ys_int=ys_int, ys_ext=ys_ext, Xs_lvls=Xs_lvls)

    b_int = ir_outer(irp_DRSint)

    return b_int, indtoout

