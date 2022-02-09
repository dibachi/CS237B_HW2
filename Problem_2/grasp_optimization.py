#!/usr/bin/env python

# from Problem_1.form_force_closure import cross_matrix
import cvxpy as cp
import numpy as np
import pdb  

from utils import *
# import utils

def solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False):
    """
    Solves an SOCP of the form:

    minimize(h^T x)
    subject to:
        ||A_i x + b_i||_2 <= c_i^T x + d_i    for all i
        F x == g

    Args:
        x       - cvx variable.
        As      - list of A_i numpy matrices.
        bs      - list of b_i numpy vectors.
        cs      - list of c_i numpy vectors.
        ds      - list of d_i numpy vectors.
        F       - numpy matrix.
        g       - numpy vector.
        h       - numpy vector.
        verbose - whether to print verbose cvx output.

    Return:
        x - the optimal value as a numpy array, or None if the problem is
            infeasible or unbounded.
    """
    objective = cp.Minimize(h.T @ x)
    constraints = []
    for A, b, c, d in zip(As, bs, cs, ds):
        constraints.append(cp.SOC(c.T @ x + d, A @ x + b))
    constraints.append(F @ x == g)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    if prob.status in ['infeasible', 'unbounded']:
        return None

    return x.value

def grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext):
    """
    Solve the grasp force optimization problem as an SOCP. Handles 2D and 3D cases.

    Args:
        grasp_normals   - list of M surface normals at the contact points, pointing inwards.
        points          - list of M grasp points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).
        wrench_ext      - external wrench applied to the object.

    Return:
        f
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)
    transformations = [compute_local_transformation(n) for n in grasp_normals]

    ########## Your code starts here ##########
    As = []
    bs = []
    cs = []
    ds = []
    # print(f"wrench_size: {N}")
    h = np.zeros(D*M+1)
    h[-1] = 1
    # h = np.expand_dims(h, axis=0).T
    x = cp.Variable(D*M+1)
    F = np.zeros((N, D*M+1))
    for i in range(M):
        # A = np.identity(M+1)
        A = np.zeros((D*M+1, D*M + 1))
        Aprime = np.zeros((D*M+1, D*M + 1))
        A[D*i, D*i] = 1
        Aprime[D*i, D*i] = 1
        Aprime[D*i +1, D*i +1] = 1
        if D == 3:
            A[D*i +1, D*i +1] = 1
            Aprime[D*i +2, D*i +2] = 1
        
        b = 0
        bprime = 0
        c = np.zeros(D*M+1)
        cprime = np.zeros(D*M+1)
        if D == 3:
            c[D*i + 2] = friction_coeffs[i]
        else:
            c[D*i + 1] = friction_coeffs[i]
        cprime[-1] = 1
        # c = np.expand_dims(c, axis=0).T
        d = 0
        dprime = 0
        fc = transformations[i] 
        mc = cross_matrix(points[i]) @ fc
        F[:,D*i : D*i + D] = np.vstack((fc, mc))
        As.append(A)
        As.append(Aprime)
        bs.append(b)
        bs.append(bprime)
        cs.append(c)
        cs.append(cprime)
        ds.append(d)
        ds.append(dprime)
    print(F)
    # g = -wrench()
    g = -wrench_ext
    
    x = solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False)

    # TODO: extract the grasp forces from x as a stacked 1D vector
    f = x[:-1]

    ########## Your code ends here ##########

    # Transform the forces to the global frame
    F = f.reshape(M,D)
    forces = [T.dot(f) for T, f in zip(transformations, F)]
    print(f"forces: {forces}") #debugging
    return forces

def precompute_force_closure(grasp_normals, points, friction_coeffs):
    """
    Precompute the force optimization problem so that force closure grasps can
    be found for any arbitrary external wrench without redoing the optimization.

    Args:
        grasp_normals   - list of M contact normals, pointing inwards from the object surface.
        points          - list of M contact points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).

    Return:
        force_closure(wrench_ext) - a function that takes as input an external wrench and
                                    returns a set of forces that maintains force closure.
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)

    ########## Your code starts here ##########
    # TODO: Precompute the optimal forces for the 12 signed unit external
    #       wrenches and store them as rows in the matrix F. This matrix will be
    #       captured by the returned force_closure() function.
    F = np.zeros((2*N, M*D))
    buh = np.vstack((np.identity(N), -np.identity(N)))
    unit_wrenches = [buh[i,:] for i in range(2*N)]
    for i in range(2*N):
        # print(f"shape of unit wrenches: {unit_wrenches[0].shape}")
        opt_forces = grasp_optimization(grasp_normals, points, friction_coeffs, unit_wrenches[i])
        # print(f"shape of x: {x.shape}")
        # print(len(x))
        F[i,:] = np.reshape(opt_forces, -1)
    print(f"big F: {F}")

    ########## Your code ends here ##########

    def force_closure(wrench_ext):
        """
        Return a set of forces that maintain force closure for the given
        external wrench using the precomputed parameters.

        Args:
            wrench_ext - external wrench applied to the object.

        Return:
            f - grasp forces as a list of M numpy arrays.
        """

        ########## Your code starts here ##########
        # TODO: Compute the force closure forces as a stacked vector of shape (M*D)
        # f = np.zeros(M*D)
        wrench_p = np.maximum(np.zeros(N), wrench_ext)
        wrench_n = np.maximum(np.zeros(N), -wrench_ext)
        # print(f"shape of wrench_p: {wrench_p.shape}")
        # print(f"shape of concat: {np.hstack((wrench_p, wrench_n))}")
        wrench_decomp = np.hstack((wrench_p, wrench_n)) #np.expand_dims(np.hstack((wrench_p, wrench_n)), axis=0)
        # print(f"shape of wrench decomp: {wrench_decomp.shape}")
        f = wrench_decomp.T @ F 
  
        ########## Your code ends here ##########

        forces = [f_i for f_i in f.reshape(M,D)]
        print(f"suboptimal forces: {forces}")
        return forces

    return force_closure
