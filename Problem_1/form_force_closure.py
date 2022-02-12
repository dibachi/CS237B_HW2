import cvxpy as cp
import numpy as np

def cross_matrix(x):
    """
    Returns a matrix x_cross such that x_cross.dot(y) represents the cross
    product between x and y.

    For 3D vectors, x_cross is a 3x3 skew-symmetric matrix. For 2D vectors,
    x_cross is a 2x1 vector representing the magnitude of the cross product in
    the z direction.
     """
    D = x.shape[0]
    if D == 2:
        return np.array([[-x[1], x[0]]])
    elif D == 3:
        return np.array([[0., -x[2], x[1]],
                         [x[2], 0., -x[0]],
                         [-x[1], x[0], 0.]])
    raise RuntimeError("cross_matrix(): x must be 2D or 3D. Received a {}D vector.".format(D))

def wrench(f, p):
    """
    Computes the wrench from the given force f applied at the given point p.
    Works for 2D and 3D.

    Args:
        f - 2D or 3D contact force.
        p - 2D or 3D contact point.

    Return:
        w - 3D or 6D contact wrench represented as (force, torque).    
    """
    ########## Your code starts here ##########
    # Hint: you may find cross_matrix(x) defined above helpful. This should be one line of code.
    w = np.concatenate((f, cross_matrix(p).dot(f)))
    ########## Your code ends here ##########

    return w

def cone_edges(f, mu):
    """
    Returns the edges of the specified friction cone. For 3D vectors, the
    friction cone is approximated by a pyramid whose vertices are circumscribed
    by the friction cone.

    In the case where the friction coefficient is 0, a list containing only the
    original contact force is returned.

    Args:
        f - 2D or 3D contact force.
        mu - friction coefficient.

    Return:
        edges - a list of forces whose convex hull approximates the friction cone.
    """
    # Edge case for frictionless contact
    if mu == 0.:
        return [f]

    # Planar wrenches
    D = f.shape[0]
    if D == 2:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 2
        theta = np.arctan(mu)
        edges[0] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) @ f
        edges[1] = np.array([[np.cos(-theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]]) @ f
        ########## Your code ends here ##########

    # Spatial wrenches
    elif D == 3:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 4
        normf = np.linalg.norm(f)
        n = np.random.randn(3)
        n -= n.dot(f) * f 
        n /= np.linalg.norm(n)
        n *= normf
        p = cross_matrix(n).dot(f)
        p /= np.linalg.norm(p)
        p *= normf
        theta = np.arctan(mu)
        edges[0] = mu*n + f
        edges[1] = -mu*n + f
        edges[2] = mu*p + f
        edges[3] = -mu*p + f

        # edges[0] = np.sin(theta)*n + np.cos(theta)*f
        # edges[1] = np.sin(-theta)*n + np.cos(theta)*f
        # edges[2] = np.sin(theta)*p + np.cos(theta)*f
        # edges[3] = np.sin(-theta)*p + np.cos(theta)*f

        ## axis-angle rotation matrix approach
        # n = np.random.randn(3)  # take a random vector
        # n -= n.dot(f) * f       # make it orthogonal to k
        # n /= np.linalg.norm(n)  # normalize it
        # nhat = cross_matrix(n)
        # theta = np.arctan(mu)
        # rotn = np.identity(3) + np.sin(theta)*nhat + (1-np.cos(theta))*(nhat@nhat)
        # fhat = cross_matrix(f)

        # edges[0] = rotn @ f 
        # edges[1] = (np.identity(3) + 1*fhat + 1*(fhat@fhat)) @ edges[0] #theta = pi/2
        # edges[2] = (np.identity(3) + 0*fhat + 2*(fhat@fhat)) @ edges[0]
        # edges[3] = (np.identity(3) - 1*fhat + 1*(fhat@fhat)) @ edges[0]
        ########## Your code ends here ##########

    else:
        raise RuntimeError("cone_edges(): f must be 3D or 6D. Received a {}D vector.".format(D))

    return edges

def form_closure_program(F):
    """
    Solves a linear program to determine whether the given contact wrenches
    are in form closure.

    Args:
        F - matrix whose columns are 3D or 6D contact wrenches.

    Return:
        True/False - whether the form closure condition is satisfied.
    """
    ########## Your code starts here ##########
    # Hint: you may find np.linalg.matrix_rank(F) helpful
    # TODO: Replace the following program (check the cvxpy documentation)

    # k = cp.Variable(1)
    # objective = cp.Minimize(k)
    # constraints = [k >= 0]
    n, j = F.shape
    # if np.linalg.matrix_rank(F) != n or np.linalg.matrix_rank(F) != j:
    k = cp.Variable(j)
    constraints = [k >= 1, F@k == 0]
    objective = cp.Minimize(cp.sum(k))
    if np.linalg.matrix_rank(F) < np.min(np.array([n,j])):
        return False



    ########## Your code ends here ##########

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.ECOS)
    print(f"Problem Value: {objective.value}")
    print(f"k: {k.value}")
    return prob.status not in ['infeasible', 'unbounded']

def is_in_form_closure(normals, points):
    """
    Calls form_closure_program() to determine whether the given contact normals
    are in form closure.

    Args:
        normals - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.

    Return:
        True/False - whether the forces are in form closure.
    """
    ########## Your code starts here ##########
    # TODO: Construct the F matrix (not necessarily 6 x 7)
    # F = np.zeros((6,7))
    j = len(normals) #number of forces
    n = np.size(wrench(normals[0], points[0])) #number of dimensions in wrench
    F = np.zeros((n, j)) #n x j matrix initialized
    for i in range(j):
        # print(wrench(normals[i], points[i])) #debugging
        F[:,i] = wrench(normals[i], points[i]) #each column of F is a wrench
    print(f"form closure matrix F: {F}")

    ########## Your code ends here ##########

    return form_closure_program(F)

def is_in_force_closure(forces, points, friction_coeffs):
    """
    Calls form_closure_program() to determine whether the given contact forces
    are in force closure.

    Args:
        forces - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.
        friction_coeffs - list of friction coefficients.

    Return:
        True/False - whether the forces are in force closure.
    """
    ########## Your code starts here ##########
    # TODO: Call cone_edges() to construct the F matrix (not necessarily 6 x 7)
    # F = np.zeros((6,7))    
    ###
    # j = len(forces) #number of forces
    # n = np.size(wrench(forces[0], points[0])) #number of dimensions in wrench
    # F = np.zeros((n, j)) #n x j matrix initialized
    # for i in range(j):
    #     # print(wrench(normals[i], points[i])) #debugging
    #     F[:,i] = wrench(normals[i], points[i]) #each column of F is a wrench
    # print(F)
    j = len(forces)
    n = np.size(wrench(forces[0], points[0]))
    # num_frictionless = sum([int(friction_coeffs[i] == 0) for i in range(j)])
    # F = np.zeros((n,int(2*n/3)*j - 3*num_frictionless))
    # for i in range(j):
    #     edges = cone_edges(forces[i], friction_coeffs[i])
    #     for k in range(len(edges)):
    #         F[:,i+k] = wrench(edges[k], points[i])
    # print(f"force closure matrix F: {F}")
    F = np.zeros((n, 0))
    
    for i in range(j):
        edges = cone_edges(forces[i], friction_coeffs[i])
        for k in range(len(edges)):
            F = np.append(F, np.expand_dims(wrench(edges[k], points[i]), axis=0).T, axis=1)
    print(f"force closure matrix F: {F}")
    ########## Your code ends here ##########

    return form_closure_program(F)
