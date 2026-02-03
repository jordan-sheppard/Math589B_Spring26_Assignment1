from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

ValueGrad = Callable[[np.ndarray], Tuple[float, np.ndarray]]

@dataclass
class BFGSResult:
    x: np.ndarray
    f: float
    g: np.ndarray
    n_iter: int
    n_feval: int
    converged: bool
    history: Dict[str, Any]

def backtracking_line_search(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_steps: int = 30,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Simple Armijo backtracking line search.
    Returns (alpha, f_new, g_new, n_feval_increment).
    """
    # Armijo: f(x + a p) <= f(x) + c1 a g^T p
    alpha = alpha0
    
    # Pre-compute the directional derivative (slope along p)
    # This is g dot p. Since p = -Hg, and H is positive definite, this should be negative.
    directional_derivative = np.dot(g, p)
    
    # Just in case p isn't a descent direction (e.g. numerical errors in BFGS), 
    # we can't expect a decrease.
    if directional_derivative > 0:
        # In a robust solver, you might reset H here, but for this assignment, 
        # we'll just proceed or print a warning.
        # print("LINE SEARCH WARNING: p is not a descent direction!")
        pass 

    current_fevals = 0              # Counts number of function evaluations
    
    for i in range(max_steps):
        # 1. Candidate position
        x_new = x + alpha * p           # Move in step size alpha in descent direction
        
        # 2. Evaluate function and gradient at candidate
        # We need the gradient later for the BFGS update, so we get it now.
        f_new, g_new = f_and_g(x_new)       # Function and gradient at new point
        current_fevals += 1
        
        # 3. Check Armijo condition
        # Target: f_new <= f_current + c1 * alpha * (slope)
        target_value = f + c1 * alpha * directional_derivative
        
        if f_new <= target_value:
            # Success!
            return alpha, f_new, g_new, current_fevals
        
        # 4. Backtrack
        # If the energy didn't drop enough, try a smaller step (scale by tau)
        alpha *= tau
        
    # If we run out of steps, return the last attempt (even if it failed the condition)
    # so the optimizer doesn't crash, though 'converged' flags will likely fail later.
    # print(f"LINE SEARCH WARNING: Did not converge in max_steps={max_steps} attempts. Continuing anyways...")
    return alpha, f_new, g_new, current_fevals

def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Minimize f(x) with BFGS.

    You should:
    - maintain an approximation H_k to the inverse Hessian
    - compute p_k = -H_k g_k
    - perform a line search to get step alpha_k
    - update x, f, g
    - update H via the BFGS formula (with curvature checks)

    Return BFGSResult with a small iteration history useful for plotting.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    n_feval = 1

    n = x.size
    H = np.eye(n)  # inverse Hessian approximation

    hist = {"f": [f], "gnorm": [np.linalg.norm(g)], "alpha": []}

    for k in range(max_iter):
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval,
                              converged=True, history=hist)

        # Search direction
        p = -H @ g

        # Line search
        alpha, f_new, g_new, inc = backtracking_line_search(
            f_and_g, x, f, g, p, alpha0=alpha0
        )
        x_new = x + alpha * p   # Use optimal step size to descent to new point
        n_feval += inc          # Increment number of function evaluations

        # Update step: Update inverse Hessian approximation H
        s = x_new - x
        y = g_new - g
        ys = np.dot(y, s)
        if ys > 1e-10:          # Ensure positive so no division by zero
            I = np.eye(len(s))
            L = I - (np.outer(s, y) / ys)
            R = I - (np.outer(y, s) / ys)
            C = np.outer(s, s) / ys
            H = L @ H @ R + C
        
        # Track the history
        hist["f"].append(f_new)
        hist["gnorm"].append(np.linalg.norm(g_new))
        hist["alpha"].append(alpha)

        # Update variables for next iteration
        f = f_new 
        g = g_new 
        x = x_new

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval,
                      converged=False, history=hist)
