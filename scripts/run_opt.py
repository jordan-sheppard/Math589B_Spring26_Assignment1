from __future__ import annotations
import sys
from pathlib import Path

# Allow running this script directly from the repo root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from elastic_rod.model import RodEnergy, RodParams
from elastic_rod.utils import random_loop, pack, unpack
from elastic_rod.bfgs import bfgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X0 = random_loop(args.N, radius=7.0, noise=0.05, seed=args.seed)
    x0 = pack(X0)

    # Tuned defaults that (once WCA is implemented) tend to produce packed/coiled states.
    params = RodParams(kb=1.0, ks=80.0, l0=0.5, kc=0.02, eps=1.0, sigma=0.35)
    model = RodEnergy(params)

    def f_and_g(x):
        return model.value_and_grad(x)

    res = bfgs(f_and_g, x0, max_iter=args.steps, tol=1e-6)
    print("converged:", res.converged, "iters:", res.n_iter, "f:", res.f, "||g||:", np.linalg.norm(res.g))

    X = unpack(res.x)

    # Close the filament
    X = np.vstack([X, X[0,]])


    plt.figure()
    plt.plot(res.history["f"])
    plt.xlabel("iteration")
    plt.ylabel("energy")
    plt.title("BFGS energy history")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:,0], X[:,1], X[:,2], marker="o", markersize=2)
    ax.set_title("Optimized filament (closed loop)")
    plt.show()

if __name__ == "__main__":
    main()
