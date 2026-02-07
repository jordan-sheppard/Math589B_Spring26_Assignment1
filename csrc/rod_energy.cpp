#include <cmath>
#include <algorithm>
#include <iostream>

extern "C" {

// Bump when you change the exported function signatures.
int rod_api_version() { return 2; }

// Subroutine implementing Algorithm 1 from Assignment 1, Q5.
// Minimizes f(s,t) = a*s^2 + 2*b*s*t + c*t^2 + d*s + e*t
// Subject to 0 <= s, t <= 1.
// Requires: a > 0 and a*c - b^2 > 0 (strictly convex).
void minimize_quadratic_box(
    double& a, double& b, double& c, double& d, double& e,
    double* s_out, double* t_out
) {
    // 1. Compute the unconstrained minimizer
    double det = b*b - a*c; 
    
    // Check strict convexity/parallelism
    if (std::abs(det) < 1e-14) {
        *s_out = 0.5; *t_out = 0.5; 
        return; 
    }

    double s_hat = -1.0 + (c*d - b*e) / det;
    double t_hat = -1.0 + (a*e - b*d) / det;

    // 2. Check if the unconstrained minimum is feasible
    if (std::abs(s_hat) <= 1.0 && std::abs(t_hat) <= 1.0) {
        *s_out = (s_hat + 1.0) / 2.0;
        *t_out = (t_hat + 1.0) / 2.0;
        return;
    }

    // 3. Analyze boundaries if unconstrained is infeasible
    auto eval_f = [&](double s, double t) {
        return a*s*s + 2.0*b*s*t + c*t*t + d*s + e*t;
    };

    double best_s = 0.0, best_t = 0.0;
    double min_val = 1e30; 

    // Case A: Boundary constrained by s
    if (std::abs(s_hat) > 1.0) {
        double s_hat_2 = (s_hat > 0.0) ? 1.0 : -1.0; 
        double t_hat_2 = -1.0 - (b * (1.0 + s_hat_2) + e) / c;
        t_hat_2 = std::max(-1.0, std::min(1.0, t_hat_2));
        
        double s2 = (s_hat_2 + 1.0) / 2.0;
        double t2 = (t_hat_2 + 1.0) / 2.0;
        double f2 = eval_f(s2, t2);
        if (f2 < min_val) {
            min_val = f2;
            best_s = s2;
            best_t = t2;
        }
    }

    // Case B: Boundary constrained by t
    if (std::abs(t_hat) > 1.0) {
        double t_hat_3 = (t_hat > 0.0) ? 1.0 : -1.0; 
        double s_hat_3 = -1.0 - (b * (1.0 + t_hat_3) + d) / a;
        s_hat_3 = std::max(-1.0, std::min(1.0, s_hat_3));
        
        double s3 = (s_hat_3 + 1.0) / 2.0;
        double t3 = (t_hat_3 + 1.0) / 2.0;
        double f3 = eval_f(s3, t3);
        if (f3 < min_val) {
            min_val = f3;
            best_s = s3;
            best_t = t3;
        }
    }

    *s_out = best_s;
    *t_out = best_t;
}

// Exported API 
void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double kc,
    double eps,
    double sigma,
    double* energy_out,
    double* grad_out
) {
    const int M = 3*N;
    for (int i = 0; i < M; ++i) grad_out[i] = 0.0;
    double E = 0.0;

    auto idx = [N](int i) {
        int r = i % N;
        return (r < 0) ? (r + N) : r;
    };
    auto get = [&](int i, int d) -> double {
        return x[3*idx(i) + d];
    };
    auto addg = [&](int i, int d, double v) {
        grad_out[3*idx(i) + d] += v;
    };

    // ---- Bending ----
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double b = get(i+1,d) - 2.0*get(i,d) + get(i-1,d);
            E += kb * b * b;
            const double c = 2.0 * kb * b;
            addg(i-1, d, c);
            addg(i,   d, -2.0*c);
            addg(i+1, d, c);
        }
    }

    // ---- Stretching ----
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i+1,0) - get(i,0);
        double dx1 = get(i+1,1) - get(i,1);
        double dx2 = get(i+1,2) - get(i,2);
        double r = std::sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
        r = std::max(r, 1e-12);
        double diff = r - l0;
        E += ks * diff * diff;

        double coeff = 2.0 * ks * diff / r;
        addg(i+1,0,  coeff * dx0);
        addg(i+1,1,  coeff * dx1);
        addg(i+1,2,  coeff * dx2);
        addg(i,0,   -coeff * dx0);
        addg(i,1,   -coeff * dx1);
        addg(i,2,   -coeff * dx2);
    }

    // ---- Confinement ----
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            double xi = get(i,d);
            E += kc * xi * xi;
            addg(i,d, 2.0 * kc * xi);
        }
    }

    // ---- Segment-segment WCA self-avoidance ----
    const double rc = std::pow(2.0, 1.0/6.0) * sigma;

    auto dot3 = [](double ax, double ay, double az,
                   double bx, double by, double bz) {
        return ax*bx + ay*by + az*bz;
    };

    for (int i = 0; i < N; ++i) {
        int i1 = idx(i+1);

        // Segment i endpoints
        double xi0 = get(i,0),  xi1 = get(i,1),  xi2 = get(i,2);
        double xi10 = get(i1,0), xi11 = get(i1,1), xi12 = get(i1,2);

        double di0 = xi10 - xi0;
        double di1 = xi11 - xi1;
        double di2 = xi12 - xi2;

        for (int j = i+1; j < N; ++j) {

            // ---- Exclusions: skip segments too close along the chain ----
            int dj = std::abs(j - i);
            dj = std::min(dj, N - dj);   // circular distance
            if (dj <= 2) continue;

            int j1 = idx(j+1);

            // Segment j endpoints
            double xj0 = get(j,0),  xj1 = get(j,1),  xj2 = get(j,2);
            double xj10 = get(j1,0), xj11 = get(j1,1), xj12 = get(j1,2);

            double dj0 = xj10 - xj0;
            double dj1 = xj11 - xj1;
            double dj2 = xj12 - xj2;

            // Solve closest points between two segments
            // Following standard segment–segment distance formula
            double r0 = xi0 - xj0;
            double r1 = xi1 - xj1;
            double r2 = xi2 - xj2;

            double a = dot3(di0,di1,di2, di0,di1,di2);
            double b = dot3(di0,di1,di2, dj0,dj1,dj2);
            double c = dot3(dj0,dj1,dj2, dj0,dj1,dj2);
            double d = dot3(di0,di1,di2, r0,r1,r2);
            double e = dot3(dj0,dj1,dj2, r0,r1,r2);

            double denom = a*c - b*b;

            double u, v;
            if (denom > 1e-12) {
                u = (b*e - c*d) / denom;
                v = (a*e - b*d) / denom;
            } else {
                // Nearly parallel segments
                u = 0.0;
                v = (b > c ? d/b : e/c);
            }

            // Clamp to [0,1]
            u = std::min(1.0, std::max(0.0, u));
            v = std::min(1.0, std::max(0.0, v));

            // Closest points
            double px0 = xi0 + u*di0;
            double px1 = xi1 + u*di1;
            double px2 = xi2 + u*di2;

            double qx0 = xj0 + v*dj0;
            double qx1 = xj1 + v*dj1;
            double qx2 = xj2 + v*dj2;

            double rx0 = px0 - qx0;
            double rx1 = px1 - qx1;
            double rx2 = px2 - qx2;

            double dist2 = rx0*rx0 + rx1*rx1 + rx2*rx2;
            double dist = std::sqrt(dist2);
            dist = std::max(dist, 1e-12);

            if (dist >= rc) continue;

            // ---- WCA energy ----
            double inv = sigma / dist;
            double inv6 = std::pow(inv, 6);
            double inv12 = inv6 * inv6;

            double U = 4.0 * eps * (inv12 - inv6) + eps;
            E += U;

            // ---- WCA force magnitude ----
            double dUdd = 24.0 * eps * (2.0*inv12 - inv6) / dist;

            double fx0 = dUdd * rx0 / dist;
            double fx1 = dUdd * rx1 / dist;
            double fx2 = dUdd * rx2 / dist;

            // ---- Distribute forces to endpoints ----
            // Segment i
            addg(i,  0, -(1.0 - u) * fx0);
            addg(i,  1, -(1.0 - u) * fx1);
            addg(i,  2, -(1.0 - u) * fx2);

            addg(i1, 0, -u * fx0);
            addg(i1, 1, -u * fx1);
            addg(i1, 2, -u * fx2);

            // Segment j
            addg(j,  0, +(1.0 - v) * fx0);
            addg(j,  1, +(1.0 - v) * fx1);
            addg(j,  2, +(1.0 - v) * fx2);

            addg(j1, 0, +v * fx0);
            addg(j1, 1, +v * fx1);
            addg(j1, 2, +v * fx2);
        }
    }

    *energy_out = E;
} 

} // extern "C"