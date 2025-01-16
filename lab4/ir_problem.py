import numpy as np
# np.float_ = np.float64
from scipy.optimize import linprog
from ir_plotmodelset import ir_plotmodelset


def ir_problem(*args):

    """
    Parses data for an interval regression problem and validates input dimensions.

    Args:
        *args:  Variable number of arguments.  Can be a dictionary (irp), or X, y, epsilon, [lb], [ub], [C], [d], [ctype].

    Returns:
        A dictionary containing the parsed data (irp) or raises ValueError with descriptive error messages.
    """

    if len(args) == 1 and isinstance(args[0], dict):
        # Case 1: irp dictionary is passed as the first argument.
        irp = args[0].copy()  # Create a copy to avoid modifying the original dictionary
        args = args[1:]
    else:
        # Case 2: X, y, epsilon, and optionally constraints are passed as separate arguments.
        if len(args) < 3:
            raise ValueError("At least three arguments (X, y, epsilon) are required.")

        X = np.array(args[0])
        y = np.array(args[1])
        # epsilon = np.array(args[2])
        epsilon = args[2]

        # print(X)
        # print(y)
        # print(epsilon)

        if not is_irp_dimension_valid(X, y, epsilon):
            raise ValueError("Invalid dimensions: X and y must have the same number of rows. "
                             "Epsilon must be a scalar or have the same dimensions as y.")

        if X.size == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty.")

        irp = {'X': X, 'y': y, 'epsilon': epsilon}
        args = args[3:]

    n, m = irp['X'].shape

    # Parse lower bounds (lb)
    lb = args[0] if len(args) > 0 else -np.inf * np.ones(m)
    if len(args) > 0 and lb is not None:
        lb = np.array(lb)
        if lb.shape != (1, m):
            raise ValueError(f"Invalid dimensions for lb: must be (1, {m})")
    irp['lb'] = lb

    # Parse upper bounds (ub)
    ub = args[1] if len(args) > 1 else np.inf * np.ones(m)
    if len(args) > 1 and ub is not None:
        ub = np.array(ub)
        if ub.shape != (1, m):
            raise ValueError(f"Invalid dimensions for ub: must be (1, {m})")
    irp['ub'] = ub

    # Parse inequality constraints (C, d, ctype)
    if len(args) == 3:
        C, d, ctype = args
        if C is not None and C.shape[1] != m:
            raise ValueError("Invalid dimensions for C: the number of columns must match the number of columns in X.")
        if d is not None and d.shape != (C.shape[0],):  # Check for correct dimensions for d
            raise ValueError("Invalid dimensions for d: must match the number of rows in C.")
        if ctype is not None and len(ctype) != C.shape[0]:
            raise ValueError("Invalid dimensions for ctype: length must match the number of rows in C.")
        if ctype is not None and not all(c.upper() in ('L', 'U') for c in ctype):
            raise ValueError("ctype must contain only 'U' or 'L' symbols.")
        irp['C'] = C
        irp['d'] = d
        irp['ctype'] = ctype
    elif len(args) > 3 or (len(args) > 0 and len(args) < 3):
        raise ValueError("Wrong number of parameters. "
                         "Inequality constraints must be specified by C, d, and ctype (or omitted).")
    else:
        irp['C'] = []
        irp['d'] = []
        irp['ctype'] = []

    return irp


def is_irp_dimension_valid(X, y, epsilon):
    """Checks the validity of dimensions for interval regression problem."""

    if y.shape[0] != X.shape[0]:
        return False
    if not np.isscalar(epsilon):
        if epsilon.shape != y.shape:
            return False
    return True


def ir_outer(irproblem):
    """
    Python translation of the Matlab ir_outer function.  Solves a series of linear programs to
    determine parameter bounds and active constraints in an interval regression problem.

    Args:
        irproblem: A dictionary containing the problem data (X, y, epsilon, lb, ub, C, d, ctype).

    Returns:
        A tuple containing:
            - beta: A NumPy array of parameter bounds (lower and upper bounds for each parameter).
            - exitcode: An integer indicating success (0) or failure (negative error code).
            - active: A dictionary containing indices of active lower and upper bound constraints.
    """

    X = np.array(irproblem['X'])
    y = np.array(irproblem['y'])
    epsilon = irproblem['epsilon']
    C = np.array(irproblem['C']) if 'C' in irproblem else np.array([])
    d = np.array(irproblem['d']) if 'd' in irproblem else np.array([])
    ctype = irproblem['ctype'] if 'ctype' in irproblem else []

    lb = irproblem.get('lb', -np.inf * np.ones(X.shape[1]))
    ub = irproblem.get('ub', np.inf * np.ones(X.shape[1]))

    if X.shape[0] != y.shape[0]:
        raise ValueError("Dimensions of X and y must match.")
    if not np.isscalar(epsilon):
        if epsilon.shape != y.shape:
            raise ValueError("epsilon must be a scalar or a vector with the same length as y.")
    if C.size > 0 and C.shape[1] != X.shape[1]:
        raise ValueError("Number of columns in C must match the number of columns in X.")
    if d.size > 0 and d.shape != (C.shape[0],):
        raise ValueError("Dimensions of C and d are inconsistent.")
    if ctype and len(ctype) != d.shape[0]:
        raise ValueError("Length of ctype must match the number of rows in C.")

    n, m = X.shape
    if C.size != 0:
        A = np.vstack((X, -X, C))
    else:
        A = np.vstack((X, -X))

    if d.size != 0:
        b = np.concatenate((y + epsilon, -y + epsilon, d))
    else:
        b = np.concatenate((y + epsilon, -y + epsilon))

    ctype_full = ['U'] * (2 * n) + list(ctype)
    vartype = ['C'] * m
    sense = 1

    SIGNIFICANT = 1e-7

    beta = []
    L = []

    for i in range(m):
        c = np.zeros(m)
        c[i] = 1
        try:
            res_low = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)),
                              options={'disp': False})
            if res_low.success:
                flow = res_low.fun
            else:
                print(f"Warning: Lower bound LP failed for parameter {i + 1}. Check input data.")
                flow = np.nan

            res_high = linprog(-c, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)),
                               options={'disp': False})
            if res_high.success:
                fhigh = res_high.fun
            else:
                print(f"Warning: Upper bound LP failed for parameter {i + 1}. Check input data.")
                fhigh = np.nan

            lambda_low = res_low.slack if res_low.success else np.array([])
            lambda_high = res_high.slack if res_high.success else np.array([])

            L = np.unique(np.concatenate(
                (L, np.where(np.abs(lambda_low) > SIGNIFICANT)[0], np.where(np.abs(lambda_high) > SIGNIFICANT)[0])))

            beta.append([flow, -fhigh])

        except ValueError as e:
            print(f"Error during LP solve for parameter {i + 1}: {e}")
            return None, -1, None

    beta = np.array(beta)
    active = {'lower': L[L > n] - n, 'upper': L[L <= n]}

    return beta, 0, active


if __name__ == "__main__":

    # К-матрица (k x m) для коэффициентов дополнительных ограничений
    C = np.array([
        [1, 0],  # Первый коэффициент
        [0, 1],  # Второй коэффициент
        [1, 1]  # Третий коэффициент
    ])

    # d - вектор (k)-правая сторона ограничений
    d = np.array([10, 5, 15])

    # ctype - вектор (k) специфицирующий тип ограничений
    ctype = np.array(['U', 'S', 'L'])

    # print("\nC: ", C)
    # print("\nd: ", d)
    # print("\nctype: ", ctype)

    def generate_constraints(d, ctype):
        constraints = []
        for i in range(len(ctype)):
            if ctype[i] == 'U':
                constraints.append(f"C[{i}, :] * beta <= {d[i]}")
            elif ctype[i] == 'S':
                constraints.append(f"C[{i}, :] * beta = {d[i]}")
            elif ctype[i] == 'L':
                constraints.append(f"C[{i}, :] * beta >= {d[i]}")
        return constraints

    constraints = generate_constraints(d, ctype)
    for constraint in constraints:
        print(constraint)

    X = [
        [1, 1],
        [1, 1.5],
        [1, 2]
    ]

    y = [
        [1.5],
        [1.4],
        [2.5]
    ]

    # print(len(X), len(y))

    epsilon = 0.75

    # print(np.isscalar(epsilon))

    ir_problem_example = ir_problem(X, y, epsilon)

    print(ir_problem_example)

    ir_plotmodelset([ir_problem_example])

    # beta, exitcode, active = ir_outer(ir_problem_example)
    #
    # # ir_plotbeta(ir_problem_example)
    #
    # if exitcode == 0:
    #     print("Beta:", beta)
    #     print("Active constraints:", active)
    # else:
    #     print(f"Error: Exit code {exitcode}")



