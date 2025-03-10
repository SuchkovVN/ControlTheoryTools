import controlTheoryToolkit as ctt
import sympy as sp

A = sp.Matrix([[4, 3, 2],
               [1, -2, 1],
               [-5, -3, -3]])
B = sp.Matrix([[-1], [1], [1]])
N = B.shape[0]
E = sp.eye(N)

C = ctt.controlMatrix(A, B)
print(C)

print(C.rank())

S = sp.Matrix([[-1, 1, 0],
               [1, -2, 0],
               [1, -1, 1]])
print(S.rank())

invS = S.inv()
print(invS)

clA = invS * A * S
print(clA)
print(invS * B)

eg = A.eigenvects()
print(eg)

P = eg[2][2][0]
P = P.row_join(eg[0][2][0])
P = P.row_join(eg[1][2][0])
# P = P.T
invP = P.inv()
print(P)
print(invP)

Aspec = invP * A * P
Bspec = invP * B
print(Aspec)
print(Bspec)

Qsym = sp.symbols(['Q' + str(i) for i in range(N)])
Thetaspec = sp.Matrix(1, N, [Qsym[0], Qsym[1], 0])
Ac = Aspec + Bspec * Thetaspec
print(Bspec * Thetaspec)
print(Ac)
S = 2
poly = Ac[0:S, 0:S].charpoly()
print(poly)
Thetaspec = Thetaspec.subs(Qsym[0], -20)
Theta = Thetaspec * invP
print(Theta)