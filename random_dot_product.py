import numpy as np
import matplotlib.pyplot as plt


Pa = np.array([[2.0, 1.0], [1.0, 2.0]])
mua = np.array([[1.0, 0.0]]).T
Px = np.array([[3.0, 0.5], [0.5, 1.0]])
mux = np.array([[0.0, 2.0]]).T
La = np.linalg.cholesky(Pa)
Lx = np.linalg.cholesky(Px)

N = 1000

x = mux + Lx @ np.random.randn(mux.shape[0], N)
a = mua + La @ np.random.randn(mua.shape[0], N)

dot_prod_samples = a.T @ x

print(f"Empirical Var: {np.var(dot_prod_samples)}")
print(f"Calculated Var: {np.trace(Pa @ Pa) + mux.T @ Pa @ mux + mua.T @ Px @ mua}")