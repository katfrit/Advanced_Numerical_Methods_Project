import numpy as np

# Wolfe line search function
def wolfe_line_search(f, grad_f, x, p, c1=1e-4, c2=0.9, alpha_init=1.0):
    alpha = alpha_init
    alpha_prev = 0
    max_iter = 50
    iter_count = 0

    while iter_count < max_iter:
        iter_count += 1
        phi = f(x + alpha * p)
        phi0 = f(x)
        dphi = np.dot(grad_f(x + alpha * p), p)
        dphi0 = np.dot(grad_f(x), p)

        # Armijo condition
        if phi > phi0 + c1 * alpha * dphi0:
            alpha = (alpha + alpha_prev) / 2.0
        # Curvature condition
        elif dphi < c2 * dphi0:
            alpha_prev = alpha
            alpha *= 2.0
        else:
            return alpha

    return alpha