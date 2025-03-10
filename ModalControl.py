import sympy as sp

#
# Control by Frobenius controlable form
#

print("|==============================|\n|====== Frobenius method ======|\n|==============================|")
# x' = Ax + Bu
A = sp.Matrix([[1, -1], [0, 1]])
B = sp.Matrix([[0], [1]])
N = B.shape[0]
E = sp.eye(N)

# target eigenvalues
targetSpec = [-1, -2]
p = sp.symbols('p')
targetCharPoly = 1
for i in range(N):
  targetCharPoly *= (p - targetSpec[i])
targetCharPoly = sp.Poly(sp.expand(targetCharPoly))
print(f"Target charpoly:\n {targetCharPoly}\n==============")

# Characterisic polynom
charPoly = A.charpoly()
print(f"Charaterisitc polynom of A:\n {charPoly}\n==============")

# Control Matrix C{A, B}
C = sp.Matrix()
temp_C_col = B
for i in range(N):
  C = C.col_insert(i, temp_C_col)
  temp_C_col = A * temp_C_col
print(f"Control matrix C:\n {C}\n rank C = {C.rank()}\n==============")

# Frobenius form of A, B, C
revNegCoeffs = list(reversed(charPoly.all_coeffs()[-N:]))
revNegCoeffs = [-i for i in revNegCoeffs]

frA = sp.Matrix()
frA = frA.row_insert(0, E[1:N, :])
frA = frA.row_insert(N - 1, sp.Matrix(1, N, revNegCoeffs))

frB = E[N - 1, :].T
print(f"matrix A in Frobenius form:\n {frA}\n And matrix B:\n {frB}\n==============")

frC = sp.Matrix()
temp_frC_col = frB
for i in range(N):
  frC = frC.col_insert(i, temp_frC_col)
  temp_frC_col = frA * temp_frC_col
print(f"Control matrix in Frobenius form\n {frC}\n==============")

# P transform matrix frX = P*X
invP = frC * C.inv()
P = invP.inv()
print(f"Inversed transform matrix P^-1:\n {invP}\n==============")
print(f"Transform matrix P:\n {P}\n==============")

# A_c control A
Qsym = sp.symbols(['Q' + str(i) for i in range(N)])
Theta = sp.Matrix(1, N, Qsym)
Ac = frA + frB * Theta
print(f"A_c matrix for Frobenius form:\n {Ac}\n==============")

targetCharPolyCoeffs = targetCharPoly.all_coeffs()
revNegCoeffsTarget = list(reversed(targetCharPolyCoeffs[-N:]))
revNegCoeffsTarget = [-i for i in revNegCoeffsTarget]
print(revNegCoeffsTarget)

eqCoeffs = Ac.row(N - 1)
Qvalues = []
for i in range(N):
  Qvalues.append(list(sp.roots(eqCoeffs[i] - revNegCoeffsTarget[i], Qsym[i]).keys())[0])

frQTheta = sp.Matrix(Qvalues)
print(f"Theta for Frobenius form:\n {frQTheta}\n==============")

qTheta = invP.T * frQTheta
print(f"Theta for original system:\n {qTheta.T}\n==============")

#
# Akkerman formula
#
print("|==============================|\n|====== Akkerman formula ======|\n|==============================|")

# inv C{A,B}
invC = C.inv()
print(f"Inversed C:\n {invC}\n==============")

targetPolyA = targetCharPoly.as_expr() + targetCharPolyCoeffs[-1] * (-1)
targetPolyA = sp.Poly(targetPolyA)
targetPolyA = targetPolyA.as_expr().subs(p, A) + targetCharPolyCoeffs[-1] * E
print(f"Characteristic of A:\n {targetPolyA}\n==============")

ak_qTheta = targetPolyA.T * invC.T * (-1 * E.col(N - 1))
print(f"Theta by Akkerman:\n {ak_qTheta}\n==============")