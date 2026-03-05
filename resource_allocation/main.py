import pickle
import numpy as np
from dual_prices import score_under_fractions_dual

with open("datasets/routerbench_0shot_scores.pkl", "rb") as f:
    data = pickle.load(f)
print(data.keys())
S = data["scores"]
w = np.array([0.4, 0.1, 0.5])

S_hat, assign, counts, c, alpha = score_under_fractions_dual(
    S, w,
    max_iter=3000,
    eta0=1e-3,   # you can try 1e-2 or 5e-4 depending on score scale
    tol=0,
    do_repair=True
)

print("Target counts:", c)
print("Actual counts:", counts)
print("S_hat(w):", S_hat)
print("alpha:", alpha)