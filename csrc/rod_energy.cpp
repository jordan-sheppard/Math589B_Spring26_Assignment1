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
    
    // Constants
    const double WCA_CUTOFF_SQ = std::pow(2.0, 1.0/3.0) * sigma * sigma;
    const double SIGMA_SQ = sigma * sigma;

    // Helper lambda: Computes WCA energy and gradient factor
    auto compute_wca = [&](double d2, double* e_val, double* g_fac) -> bool {
        if (d2 >= WCA_CUTOFF_SQ || d2 < 1e-14) return false;

        // Prevent division by zero or massive explosion for overlapping segments
        // Treat anything closer than 1e-6 as effectively 1e-6 deep
        double effective_d2 = std::max(d2, 1e-12);

        double R = SIGMA_SQ / effective_d2; 
        double R3 = R * R * R;      
        double R6 = R3 * R3;        

        *e_val = 4.0 * eps * (R6 - R3) + eps;
        *g_fac = (24.0 * eps / effective_d2) * (R3 - 2.0 * R6);
        return true;
    };

    // Loop unique pairs.
    // i only needs to go up to N-3, because j starts at i+3.
    for (int i = 0; i < N - 3; ++i) {
        
        // Start j at i + 3 to automatically exclude:
        //  - self (i)
        //  - adjacent (i+1)
        //  - next-nearest (i+2)
        for (int j = i + 3; j < N; ++j) {
            
            // WRAP-AROUND CHECK:
            // Check if j is too close to i from the OTHER side of the loop.
            // The distance walking backwards from i to j is N - (j - i).
            // We want to exclude if that distance is 1 (neighbor) or 2 (next-nearest).
            int dist_backward = N - (j - i);
            if (dist_backward <= 2) {
                continue; 
            }
            // 1. Quadratic Coefficients
            double pi[3], pi_next[3], pj[3], pj_next[3];
            double u_vec[3], v_vec[3], r0[3];
            
            for (int k = 0; k < 3; ++k) {
                pi[k]      = get(i, k);
                pi_next[k] = get(i + 1, k);
                pj[k]      = get(j, k);
                pj_next[k] = get(j + 1, k);
                u_vec[k]   = pi_next[k] - pi[k];
                v_vec[k]   = pj_next[k] - pj[k];
                r0[k]      = pi[k] - pj[k];
            }

            double a_coeff = 0, b_coeff = 0, c_coeff = 0, d_coeff = 0, e_coeff = 0;
            for (int k = 0; k < 3; ++k) {
                a_coeff += u_vec[k] * u_vec[k];
                c_coeff += v_vec[k] * v_vec[k];
                b_coeff -= u_vec[k] * v_vec[k];
                d_coeff += 2.0 * r0[k] * u_vec[k];
                e_coeff -= 2.0 * r0[k] * v_vec[k];
            }

            // 2. Minimize Distance
            double u_star, v_star;
            double det_check = a_coeff * c_coeff - b_coeff * b_coeff;

            if (det_check > 1e-12) {
                minimize_quadratic_box(a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, &u_star, &v_star);
            } else {
                u_star = 0.5; v_star = 0.5;
            }

            // 3. Recompute Geometry
            double r_vec[3];
            double d2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                r_vec[k] = r0[k] + u_star * u_vec[k] - v_star * v_vec[k];
                d2 += r_vec[k] * r_vec[k];
            }

            // 4. Energy & Force
            double wca_E = 0.0;
            double wca_factor = 0.0;

            if (compute_wca(d2, &wca_E, &wca_factor)) {
                E += wca_E;

                // Distribute forces barycentrically
                for (int k = 0; k < 3; ++k) {
                    double f_val = wca_factor * r_vec[k];

                    addg(i,   k, (1.0 - u_star) * f_val);
                    addg(i+1, k, u_star         * f_val);
                    addg(j,   k, -(1.0 - v_star) * f_val);
                    addg(j+1, k, -v_star        * f_val);
                }
            }
        }
    }

    *energy_out = E;
} 

} // extern "C"