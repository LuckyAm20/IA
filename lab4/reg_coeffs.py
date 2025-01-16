import matplotlib.pyplot as plt
import numpy as np
np.float_ = np.float64
import intvalpy as ip
from ir_problem import ir_problem, ir_outer
from estimates import calibration_data_all_bins
from data_corr import data_corr_naive
from read_dir import rawData_instance
from colorama import init, Fore
import pickle
from scaler import SCALE, RADIUS

init(autoreset=True)


def print_intervals(ys_int, ys_ext, Xs_lvls, actually_draw=True):
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
                    ecolor='g', elinewidth=0.8, capsize=4, capthick=1)
        plt.errorbar(Xs_lvls, ys_ext_to_plot, yerr=yerr_ext, linestyle='none',
                    ecolor='r', elinewidth=0.8, capsize=4, capthick=1)
        # plt.xlim(-0.5, -0.46)
        # plt.ylim(1000, 3000)
        plt.show()
    else:
        plt.errorbar(Xs_lvls, ys_int_to_plot, yerr=yerr_int, marker=".", linestyle='none',
                    ecolor='g', elinewidth=0.8, capsize=4, capthick=1)
        plt.errorbar(Xs_lvls, ys_ext_to_plot, yerr=yerr_ext, linestyle='none',
                    ecolor='r', elinewidth=0.8, capsize=4, capthick=1)


def get_estimations(ch, cells, data):
    ys_int = calibration_data_all_bins(ch, cells, "Int", data)
    ys_ext = calibration_data_all_bins(ch, cells, "Ext", data)
    return ys_int, ys_ext


def regression_coeff(Ysint, Ysout, Xi, graphics=False, ys_int=None, ys_ext=None, Xs_lvls=None):
    y_out = ip.mid(Ysout)*(SCALE) - 0.5
    epsilon_out = ip.rad(Ysout)*(RADIUS)

    irp_DRSout = ir_problem(Xi, y_out, epsilon_out)

    b_out = ir_outer(irp_DRSout)

    b_int, ind_to_out = data_corr_naive(Ysint, Ysout, ip.asinterval(Xi), graphics, ys_int, ys_ext, Xs_lvls)

    non_comp_count = len(ind_to_out)

    return b_int, b_out, non_comp_count


def calc_lvl(b_int, b_out, Yint, Yout):
    b_0_int = ip.Interval(b_int[0][0])
    b_1_int = ip.Interval(b_int[0][1])
    b_0_out = ip.Interval(b_out[0][0])
    b_1_out = ip.Interval(b_out[0][1])

    Xint = (Yint - b_0_int) / b_1_int
    Xout = (Yout - b_0_out) / b_1_out

    return Xint, Xout


def calibrate(ch, cells, graphics=False):
    print(Fore.MAGENTA + f"\n\nCHANNEL: {ch}, CELL: {cells}\n\n")

    ys_int, ys_ext = get_estimations(ch, cells, rawData_instance)
    print("\nin-estimation:", ys_int)
    print("\nex-estimation:", ys_ext)
    Xs_lvls_ = rawData_instance.lvls

    Xs_lvls_ind = np.argsort(Xs_lvls_)
    Xs_lvls = sorted(Xs_lvls_)
    ys_int = ys_int[Xs_lvls_ind]
    ys_ext = ys_ext[Xs_lvls_ind]

    if graphics:
        print_intervals(ys_int, ys_ext, Xs_lvls)

    Ysint = ip.Interval(ys_int)
    Ysout = ip.Interval(ys_ext)
    Xi = np.vstack(([1] * len(Xs_lvls), Xs_lvls)).T

    b_int, b_out, non_comp_count = regression_coeff(Ysint, Ysout, Xi, graphics, ys_int, ys_ext, Xs_lvls)

    if b_int[1] == 0:
        print(Fore.RED + f"\n\nb_int: ", b_int[0])
        print("\nactive constraints: ", b_int[2])
    else:
        print(f"\nerror: exit code {b_int[1]}")

    if b_out[1] == 0:
        print(Fore.RED + f"\n\nb_out: ", b_out[0])
        print("\nactive constraints: ", b_out[2])
    else:
        print(f"\nerror: exit code {b_out[1]}")

    print(Fore.BLUE + f"\n\nПРОВЕРКА ПОЛУЧЕННЫХ КОЭФФИЦИЕНТОВ:")

    for ind in range(len(Xs_lvls)):
        x1, x2 = calc_lvl(b_int, b_int, Ysint[ind] * (SCALE) - 0.5, Ysout[ind] * (SCALE) - 0.5)
        print(Fore.BLUE + "\nXi: ", Xi[ind][1])
        if Xi[ind][1] in x1:
            print("\nx1: ", Fore.GREEN + f"{x1}")
            print("\nx2: ", Fore.GREEN + f"{x2}")
        else:
            print("\nx1: ", Fore.RED + f"{x1}")
            print("\nx2: ", Fore.RED + f"{x2}")

    return b_int, b_out


if __name__ == "__main__":
    b_int, b_out = calibrate(1, 2, True)





