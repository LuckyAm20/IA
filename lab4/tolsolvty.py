from numpy import (size, all, newaxis, ones, ceil, any, abs, maximum, min, argmin, max, zeros, eye, finfo, mod, roll,
                   sum, c_, arange, sort, argsort, remainder)
from numpy.linalg import svd, lstsq, norm


def tolsolvty(infA, supA, infb, supb, *varargin):
    #   проверка корректности входных данных
    m_inf, n_inf = size(infA, 0), size(infA, 1)
    m_sup, n_sup = size(supA, 0), size(supA, 1)
    if m_inf == m_sup:   # m - количество уравнений в системе
        m = m_sup
    else:
        raise ValueError('Количество строк в матрицах левых и правых концов неодинаково')
    if n_inf == n_sup:
        n = n_sup
    else:
        raise ValueError('Количество столбцов в матрицах левых и правых концов неодинаково')
    k_inf, k_sup = size(infb, 0), size(supb, 0)
    if k_inf == k_sup:
        k = k_sup
    else:
        raise ValueError('Количество компонент у векторов левых и правых концов неодинаково')
    if k != m:
        raise ValueError('Размеры матрицы системы не соответствуют размерам правой части')
    if not all(all(infA <= supA, 0)[newaxis]):  # интервальная арифметика каухера
        raise ValueError('В матрице системы задан неправильный интервальный элемент')
    if not all(infb <= supb):  # интервальная арифметика каухера
        raise ValueError('В векторе правой части задана неправильная интервальная компонента')
    #  задание параметров алгоритма суперградиентного подъёма и прочих
    maxitn = 2000           # ограничение на количество шагов алгоритма
    nsims = 30              # допустимое количество одинаковых шагов
    epsf = 1.e-6            # допуск на изменение значения функционала
    epsx = 1.e-6            # допуск на изменение аргумента функционала
    epsg = 1.e-6            # допуск на норму суперградиента функционала
    alpha = 2.3             # коэффициент растяжения пространства в алгоритме
    hs = 1.                 # начальная величина шага одномерного поиска
    nh = 3                  # число одинаковых шагов одномерного поиска
    q1 = 0.9                # q1, q2 - параметры адаптивной регулировки
    q2 = 1.1                # шагового множителя
    iprn = 1                # печать о ходе процесса через каждые iprn-итераций (если iprn < 0, то печать подавляется)
    weight = ones((m, 1))   # задание вектора весовых коэффициентов для образующих
    #  формирование строковых констант для оформления протокола работы
    HorLine = '-------------------------------------------------------------'
    TitLine = 'Протокол максимизации распознающего функционала Tol'
    TabLine = 'Шаг        Tol(x)         Tol(xx)   ВычФун/шаг  ВычФун'

    #   переназначение параметров алгоритма, заданных пользователем
    nargin = 4 + len(varargin)
    if nargin >= 5:
        iprn = ceil(varargin[0])
        if nargin >= 6:
            weight = varargin[1]
            if size(weight, 0) != m:
                raise ValueError('Размер вектора весовых коэффициентов задан некорректно')
            if any(weight <= 0):
                raise ValueError(' Вектор весовых коэффициентов должен быть положительным')
            if nargin >= 7:
                epsf = varargin[2]
                if nargin >= 8:
                    epsx = varargin[3]
                    if nargin >= 9:
                        epsg = varargin[4]
                        if nargin >= 10:
                            maxitn = varargin[5]

    def calcfg(x):
        #   функция, которая вычисляет значение f максимизируемого распознающего
        #   функционала и его суперградиент g;  кроме того, она выдаёт вектор tt
        #   из значений образующих функционала в данной точке аргумента
        #
        #   для быстрого вычисления образующих распознающего функционала
        #   используются сокращённые формулы умножения интервальной матрицы
        #   на точечный вектор, через середину и радиус
        absx = abs(x)
        Ac_x = Ac @ x
        Ar_absx = Ar @ absx
        infs = bc - (Ac_x + Ar_absx)
        sups = bc - (Ac_x - Ar_absx)
        tt = weight * (br - maximum(abs(infs), abs(sups)))

        # print("TT: ", tt)

        #   сборка значения всего распознающего функционала
        [f, mc] = min(tt), argmin(tt)

        #   вычисление суперградиента той образующей распознающего функционала,
        #   на которой достигается предыдущий минимум
        infA_mc = infA[[mc], :].conj().T
        supA_mc = supA[[mc], :].conj().T
        x_neg = x < 0
        x_nonneg = x >= 0
        dl = infA_mc * x_neg + supA_mc * x_nonneg
        ds = supA_mc * x_neg + infA_mc * x_nonneg
        if -infs[mc, 0] <= sups[mc, 0]:
            g = weight[mc, 0] * ds
        else:
            g = -weight[mc, 0] * dl
        return f, g, tt

    #   формируем начальное приближение x как решение либо псевдорешение 'средней' точечной системы,
    #   если она не слишком плохо обусловлена, иначе берём начальным приближением нулевой вектор
    #
    Ac = 0.5 * (infA + supA)
    Ar = 0.5 * (supA - infA)
    bc = 0.5 * (infb + supb)
    br = 0.5 * (supb - infb)
    sv = svd(Ac, compute_uv=False)[:, newaxis]
    minsv = min(sv)
    maxsv = max(sv)

    if (minsv != 0 and maxsv / minsv < 1.e+12):
        x = lstsq(Ac, bc, rcond=None)[0]
    else:
        x = zeros((n, 1))

    #   Рабочие массивы:
    #       B - матрица обратного преобразования пространства
    #       vf - вектор приращений функционала на последних шагах алгоритма
    #       g, g0, g1 - используются для хранения вспомогательных векторов,
    #           суперградиента минимизируемого функционала и др.

    B = eye(n, n)                                 # инициализируем единичной матрицей
    vf = finfo(float).max * ones((nsims, 1))      # инициализируем самыми большими числами

    #   установка начальных параметров
    w = 1. / alpha - 1.
    lp = iprn

    [f, g0, tt] = calcfg(x)
    ff, xx = f, x
    cal, ncals = 1, 1

    if iprn > 0:
        print('\n\t%52s' % TitLine)
        print('%65s' % HorLine)
        print('\t%50s' % TabLine)
        print('%65s' % HorLine)
        print('\t%d\t%f\t%f\t%d\t%d' % (0, f, ff, cal, ncals))

    #   основной цикл алгоритма:
    #       itn - счётчик числа итераций
    #       xx  - приближение к аргументу максимума функционала
    #       ff  - приближение к максимуму функционала
    #       cal - количество вычислений функционала на текущем шаге
    #       ncals - общее количество вычислений целевого функционала
    #
    for itn in range(1, maxitn + 1):
        vf[nsims - 1, 0] = ff
        #   критерий останова по норме суперградиента
        if norm(g0) < epsg:
            ccode = 2
            break
        #   вычисляем суперградиент в преобразованном пространстве,
        #   определяем направление подъёма
        g1 = B.conj().T @ g0
        g = B @ g1 / norm(g1)
        normg = norm(g)
        #   одномерный подъём по направлению g:
        #       cal - счётчик шагов одномерного поиска,
        #       deltax - вариация аргумента в процессе поиска
        r = 1
        cal = 0
        deltax = 0
        while (r > 0. and cal <= 500):
            cal = cal + 1
            x = x + hs * g
            deltax = deltax + hs * normg
            [f, g1, tt] = calcfg(x)
            if f > ff:
                ff = f
                xx = x
            #   если прошло nh шагов одномерного подъёма,
            #   то увеличиваем величину шага hs
            if mod(cal, nh) == 0:
                hs = hs * q2
            r = g.conj().T @ g1
        #   если превышен лимит числа шагов одномерного подъёма, то выход
        if cal > 500:
            ccode = 5
            break
        #   если одномерный подъём занял один шаг,
        #   то уменьшаем величину шага hs
        if cal == 1:
            hs = hs * q1
        #   уточняем статистику и при необходимости выводим её
        ncals = ncals + cal
        if itn == lp:
            print('\t%d\t%f\t%f\t%d\t%d' % (itn, f, ff, cal, ncals))
            lp = lp + iprn
        #   если вариация аргумента в одномерном поиске мала, то выход
        if deltax < epsx:
            ccode = 3
            break
        #   пересчитываем матрицу преобразования пространства
        dg = B.conj().T @ (g1 - g0)
        xi = dg / norm(dg)
        B = B + w * (B @ xi) @ xi.conj().T
        g0 = g1
        #   проверка изменения значения функционала, относительного
        #   либо абсолютного, на последних nsims шагах алгоритма
        vf = roll(vf, 1)
        vf[0, 0] = abs(ff - vf[0, 0])
        if abs(ff) > 1:
            deltaf = sum(vf) / abs(ff)
        else:
            deltaf = sum(vf)
        if deltaf < epsf:
            ccode = 1
            break
        ccode = 4

    tolmax = ff
    argmax = xx

    #   сортируем образующие распознающего функционала по возрастанию
    tt = c_[arange(1, m + 1)[newaxis].conj().T, tt]
    [z, ind] = sort(tt[:, [1]], 0), argsort(tt[:, [1]], 0)
    envs = tt[ind[:, 0], :]

    if iprn > 0:
        if remainder(itn, iprn) != 0:
            print('\t%d\t%f\t%f\t%d\t%d' % (itn, f, ff, cal, ncals))
        print('%65s' % HorLine)

    if tolmax >= 0:
        print('Допусковое множество решений интервальной линейной системы непусто')
    else:
        print('Допусковое множество решений интервальной линейной системы пусто')

    if tolmax < 0. and abs(tolmax / epsx) < 10:
        print('Абсолютное значение вычисленного максимума находится в пределах заданной точности. Перезапустите'
              ' программу с меньшими значениями  epsf и/или epsx для получения большей информации о разрешимости'
              ' рассматриваемой задачи о допусках')

    return tolmax, argmax, envs, ccode
