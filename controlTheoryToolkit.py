import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import scipy.integrate

def any_of(arr, unary_pred):
    for a in arr:
        if unary_pred(a):
            return True
    return False

def all_of(arr, unary_pred):
    for a in arr:
        if not unary_pred(a):
            return False
    return True

def controlMatrix(A, B):
    '''
    Returns C{A,B} = (B, AB, A^2B, ...) control matrix
    A - N x N matrix, B - N x 1 column vector
    '''

    N = B.shape[0]
    C = sp.Matrix()
    temp_C_col = B
    for i in range(N):
        C = C.col_insert(i, temp_C_col)
        temp_C_col = A * temp_C_col
    
    return C

def frTransform(A, B, C=None):
    '''
    Finds Frobenius controlable form of system x' = Ax + Bu
    Return (A*, B*, C*, P), where P is transition matrix to Frobenius contrl. form
    '''

    N = B.shape[0]
    E = sp.eye(N)

    charPolyCoeffs = A.charPoly().coeffs()
    frALastRow =  list(reversed(charPolyCoeffs[-N:]))
    frALastRow = [-i for i in frALastRow]
    frB = E.col(N-1)
    frA = sp.Matrix()
    frA = frA.row_insert(E[1:N-1, :])
    frA = frA.row_insert(N-1, frALastRow)
    
    frC = sp.Matrix()
    temp_frC_col = frB
    frC = frC.col_insert(0, temp_frC_col)
    for i in range(1, N):
        temp_frC_col = frA * temp_frC_col
        frC = frC.col_insert(i, temp_frC_col)

    # P^-1 = frC * C^-1
    if (C == None):
        C = controlMatrix(A, B)
        if (C.rank() < N):
            raise ValueError("C is singular matrix")
    
    invP = frC * C.inv()
    return frA, frB, frC, P.inv()

def AlgebraicRiccatiSol(A, B, Q, R, evalf=True):
    '''
    Finds positive-defined symmetric solution of Algebraic Riccati equation
    w/ given matrices
    !Before call this function check condition of coresponding theorem about
    existence of that specific form solution
    '''
    N = B.shape[0]
    # Create a symmetric symbolic matrix
    X = sp.MatrixSymbol('X', N, N).as_explicit()
    X = X.upper_triangular(1) + X.upper_triangular().T

    RiccatiEq = Q + A.T * X + X * A - X * B * R.inv() * B.T * X
    solution = sp.solve(RiccatiEq, X)

    approx_form = []
    for sol in solution:
        t = []
        for x in sol:
            if evalf:
                t.append(x.evalf())
            else:
                t.append(x)
        sol_matrix = sp.zeros(N)

        for i in range(N):
            for j in range(N - i):
                sol_matrix[i, i + j] = t[j + i * (N - i  + 1)]
        sol_matrix = sol_matrix + sol_matrix.upper_triangular(1).T
        approx_form.append(sol_matrix)

    for sol in approx_form:
        evals = sol.eigenvals()
        if (all_of(evals, lambda x : x.is_real and x > 0)):
            return sol
    return []

def solveCommonSystem(t, x, F, u, x_0, t_lims, t_eval):
    '''
    Finds numerical solution for equation of form
    x'(t) = F(t, x, u), where x(t) is m-dim vector
    F(t, x) is m-dim vector w/ equations and u(t, x) is input
    Returns solution for x and calculated values of u in t_eval points
    '''
    splam = sp.lambdify((t, x), (F))
    
    def rhs(t, x):
        s = splam(t, x)
        return [item[0] for item in s]

    sol = scipy.integrate.solve_ivp(rhs, t_lims, x_0, t_eval=t_eval)
    cont = sp.lambdify((t, x), u)
    U = [cont(t, y)[0][0] for t, y in zip(t_eval, sol.y.T)]
    return sol, U

def solveLinearSystem(t, x, A, u, B, x_0, t_lims, t_eval):
    '''
    Finds numerical solution for equation of form
    x'(t) = Ax(t) + Bu(t, x)
    Returns solution for x and calculated values of u in t_eval points
    '''
    return solveCommonSystem(t, x, A*x + B*u, u, x_0, t_lims, t_eval)




