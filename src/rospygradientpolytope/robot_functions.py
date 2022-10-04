import numpy as np
from scipy.special import logsumexp
import itertools
import warnings

def smooth_max(a):
    
    #print(a)
    return logsumexp(a)


def smooth_max_gradient(a):
    # http://eprints.maths.manchester.ac.uk/2728/1/paper.pdf
    return np.exp(a) / sum(np.exp(a))

def exp_sum(x,x_total):
    exp_sum_total = 0
    for i in range(len(x_total)):
        exp_sum_total += np.exp(x_total[i])
    
    return np.exp(x*1.0)/(exp_sum_total*1.0)
def exp_normalize(x):
    # a normalized version of smooth max gradient which can cope with large differences such as A=[1,10,1000]
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

    
def sigmoid(z, a=1.0):
    # Continuous Sigmoid function returns 0 if z is negative and 1 if positive.
    # The slope is determined by a
    
    
    ## To filter overflow error due to big values in input - z
    warnings.filterwarnings('ignore')
    z = np.asarray(z)
    
    
    scalar_input = False
    if z.ndim == 0:
        z = z[None]  # Makes x 1D
        scalar_input = True
    z = 1.0 / (1.0 + np.exp(-z * a))
    if scalar_input:
        return z.item()
    return z


def skew(u):
    # skew symmetric matrix performing the cross product
    uskew = np.zeros((3, 3))
    uskew[0, :] = [0.0, -u[2], u[1]]
    uskew[1, :] = [u[2], 0.0, -u[0]]
    uskew[2, :] = [-u[1], u[0], 0.0]
    return uskew


def screw_transform(L):
    lhat = skew(L)
    m1 = np.hstack([np.eye(3), -lhat])
    m2 = np.hstack([np.zeros([3, 3]), np.eye(3)])
    s = np.vstack([m1, m2])
    return s


def sigmoid_gradient(z, a=1.0):
    # Gradient of sigmoid function
    g = a * sigmoid(z, a) * (1 - sigmoid(z, a))
    return g


def cross_product_normalized(v1, v2):
    return np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))


def gradient_cross_product_normalized(v1, v2, dv1, dv2):
    # gradient of a cross product of two vector v1 v2 divided by its norm
    # dv1, dv2 are gradient of each vector
    # n = cross(v1,v2) / norm(cross(v1,v2))
    # n=   u/v
    # dn  =  du v - u dv
    # dq     dq       dq
    #       ---------------
    #             v'*v

    u = np.cross(v1, v2)
    v = np.linalg.norm(np.cross(v1, v2))

    dudq = gradient_cross_product(v1, v2, dv1, dv2)
    dvdq = gradient_vector_norm(u, dudq)

    dndq = ((dudq * v) - (u * dvdq)) / (v ** 2)
    return dndq


def gradient_vector_norm(v1, dv1):
    # gradient of the norm of a vector v1 with vector gradient dv1
    return np.dot(dv1, v1) / (np.dot(v1, v1) ** 0.5)


def gradient_cross_product(v1, v2, dv1, dv2):
    # gradient of the cross product of two vector v1 v2
    # dv1, dv2 are gradient of each vector
    return np.matmul(-skew(v2), dv1) + np.matmul(skew(v1), dv2)


def getHessian(J):
    # get kinematic Hessian matrix for a robot of revolute joints
    # Paper derivation http://dx.doi.org/10.1016/0094-114X(95)00069-B
    # Input Jacobian  J = [v_1 v_2....v_n]  matrix of unit twists  6  x n
    # Output Hessian Tensor (nxm) xn   H[:,:,j] = [dv_1 / dq_j dv_2 / dq_j dv_n / dq_j]
    # H contains n matrices H[:,:,1] is the second full matrix
    rows, cols = np.shape(J)
    H = np.zeros([rows, cols, cols])
    
    for i in range(cols):
        for j in range(cols):
            twist_i = J[:, i]
            twist_j = J[:, j]
            if i < j:
                omega_i_hat = skew(J[3:6, i])
                a_rows13 = np.hstack((omega_i_hat, np.zeros([3, 3])))
                a_rows36 = np.zeros([3, 6])
                a = np.vstack((a_rows13, a_rows36))
                H[:, i, j] = np.matmul(a, twist_j)
            elif i >= j:
                omega_j_hat = skew(J[3:6, j])
                a_rows13 = np.hstack((omega_j_hat, np.zeros([3, 3])))
                a_rows36 = np.hstack((np.zeros([3, 3]), omega_j_hat))
                a = np.vstack([a_rows13, a_rows36])
                H[:, i, j] = np.matmul(a, twist_i)
    return H

def getHessian_2(J):
    # get kinematic Hessian matrix for a robot of revolute joints
    # Paper derivation http://dx.doi.org/10.1016/0094-114X(95)00069-B
    # Input Jacobian  J = [v_1 v_2....v_n]  matrix of unit twists  6  x n
    # Output Hessian Tensor (nxm) xn   H[:,:,j] = [dv_1 / dq_j dv_2 / dq_j dv_n / dq_j]
    # H contains n matrices H[:,:,1] is the second full matrix
    rows, cols = np.shape(J)
    H = np.zeros([rows, cols, cols])
    for i in range(cols):
        for j in range(cols):
            twist_i = J[:, i]
            twist_j = J[:, j]
            if i < j:
                omega_i_hat = skew(J[3:6, i])
                a_rows13 = np.hstack((omega_i_hat, np.zeros([3, 3])))
                a_rows36 = np.zeros([3, 6])
                a = np.vstack((a_rows13, a_rows36))
                H[:, i, j] = np.matmul(a, twist_j)
            elif i >= j:
                omega_j_hat = skew(J[3:6, j])
                a_rows13 = np.hstack((omega_j_hat, np.zeros([3, 3])))
                a_rows36 = np.hstack((np.zeros([3, 3]), omega_j_hat))
                a = np.vstack([a_rows13, a_rows36])
                H[:, i, j] = np.matmul(a, twist_i)
    return H

def getDofCombinations(active_joints, m):
    # % We select m-1 independent twists that define m contained in N
    # %    the rest n-(m-1) are contained in N-1
    # % This is repeated for all combinations and pairs are corresponding rows in
    # % N and Nnot

    N = np.array(list(itertools.combinations(active_joints, m - 1)))
    Nnot = np.ones([np.shape(N)[0], len(active_joints) - (m - 1)], dtype=int)
    for i in range(np.shape(N)[0]):
        Nnot[i, :] = active_joints[
            ~np.isin(active_joints, N[i, :])]  # isin returns boolean array of intersection, ~ invert and then selects
    return N, Nnot

def get_jacobian_shifted(J):
    ## Input Jacobian angular velocity and linear velocity components are shifted
    ## Input -> J = [[w];[v]] Output -> J = [[v];[w]]
    #print('jacob_hessian')
    #print(J)
    return np.vstack((J[3:6,:],J[0:3,:]))


def getJ_pinv(J,coeff_damp):
    
    ## Input- Jacobian - square matrix if possible - 
    ## Non-square implement for parallel robots
    from numpy import transpose,matmul,eye,shape
    from numpy.linalg import inv
    
    I = eye(shape(J)[0])
    ## J^T(JJ^T + p^2*I)^-1
    J_T = transpose(J)
    #print('J_T is',J_T)
    
    J_pinv = matmul(J_T,inv(matmul(J,J_T) + ((coeff_damp**2)*I)))

    return J_pinv