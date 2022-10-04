import numpy as np
import polytope
import robot_functions
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import itertools


def get_Cartesian_polytope(jacobian, joint_space_vrep):
    Pv = np.zeros([np.shape(joint_space_vrep)[0], np.shape(jacobian)[0]])

    for row, i in zip(joint_space_vrep, range(np.shape(joint_space_vrep)[0])):
        Pv[i, :] = np.matmul(jacobian, row)
    return Pv


def plot_polytope_3d(poly):
    V = polytope.extreme(poly)
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.gca()
    hull = ConvexHull(V, qhull_options='Qs QJ')
    ax.plot(hull.points[hull.vertices, 0],
            hull.points[hull.vertices, 1],
            hull.points[hull.vertices, 2], 'ko', markersize=4)

    s = ax.plot_trisurf(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2], triangles=hull.simplices,
                        color='red', alpha=0.2, edgecolor='k')

    plt.show()
    return ax


def point_in_hull(point, hull, tolerance=1e-12):
    #    https: // stackoverflow.com / questions / 16750618 / whats - an - efficient - way - to - find - if -a - point - lies - in -the - convex - hull - of - a - point - cl / 42165596  # 42165596
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def get_hyperplane_parameters(JE, H, deltaq, sigmoid_slope=0):
    # %getHyperplanes Gets the hyperplanes parameters
    # %   gets the parameters that can be used to construct the cartesian
    # %   velocitity polytopes using hyperplane method as well as their gradients,

    m = np.shape(JE)[0]  # number of task space degrees of freedom
    number_of_joints = np.shape(JE)[1]  # number of joints
    active_joints = np.arange(number_of_joints)  # listing our the joints
    N, Nnot = robot_functions.getDofCombinations(active_joints, m)
    number_of_combinations = np.shape(N)[0]

    n = np.zeros([number_of_combinations, m])
    d_n_dq = np.zeros([number_of_combinations, m, number_of_joints])

    hplus = np.zeros([number_of_combinations, ])
    hminus = np.zeros([number_of_combinations, ])

    d_hplus_dq = np.zeros([number_of_combinations, number_of_joints])
    d_hminus_dq = np.zeros([number_of_combinations, number_of_joints])

    v1 = np.zeros([3, ])
    v2 = np.zeros([3, ])
    d_v1_dq = np.zeros([3, number_of_joints])
    d_v2_dq = np.zeros([3, number_of_joints])

    vk = np.zeros([3, ])
    d_vk_dq = np.zeros([3, number_of_joints])

    d_nt_dot_vk_dq = np.zeros([number_of_joints, ])

    for i in range(np.shape(N)[0]):
        v1 = JE[:, N[i, 0]]
        v2 = JE[:, N[i, 1]]
        n[i, :] = robot_functions.cross_product_normalized(v1, v2)

        for joint in range(number_of_joints):
            d_v1_dq[:, joint] = H[0:3, N[i, 0], joint]
            d_v2_dq[:, joint] = H[0:3, N[i, 1], joint]
            d_n_dq[i, :, joint] = robot_functions.gradient_cross_product_normalized(v1, v2, d_v1_dq[:, joint],
                                                                                    d_v2_dq[:, joint])
            d_hplus_dq[i, joint] = 0.0
            d_hminus_dq[i, joint] = 0.0

        hplus[i] = 0
        hminus[i] = 0

        for j in range(number_of_joints - (m - 1)):

            vk = JE[:, Nnot[i, j]]
            nt_dot_vk = np.matmul(vk, n[i, :])

            for joint in range(number_of_joints):
                d_vk_dq[:, joint] = H[0:3, Nnot[i, j], joint]
                d_nt_dot_vk_dq[joint] = np.matmul(d_n_dq[i, :, joint], vk) + np.matmul(n[i, :], d_vk_dq[:, joint])

                d_hplus_dq[i, joint] = d_hplus_dq[i, joint] + \
                                       (robot_functions.sigmoid_gradient(nt_dot_vk, sigmoid_slope)
                                        *
                                        d_nt_dot_vk_dq[joint] * deltaq[Nnot[i, j]] * nt_dot_vk) + \
                                       (robot_functions.sigmoid(nt_dot_vk, sigmoid_slope)
                                        *
                                        d_nt_dot_vk_dq[joint] * deltaq[Nnot[i, j]])

                d_hminus_dq[i, joint] = d_hminus_dq[i, joint] + \
                                        (robot_functions.sigmoid_gradient(nt_dot_vk, -sigmoid_slope)
                                         *
                                         d_nt_dot_vk_dq[joint] * deltaq[Nnot[i, j]] * nt_dot_vk) + \
                                        (robot_functions.sigmoid(nt_dot_vk, -sigmoid_slope)
                                         *
                                         d_nt_dot_vk_dq[joint] * deltaq[Nnot[i, j]])

            hplus[i] = hplus[i] + (robot_functions.sigmoid(nt_dot_vk, sigmoid_slope)) * (deltaq[Nnot[i, j]] * nt_dot_vk)
            hminus[i] = hminus[i] + (robot_functions.sigmoid(nt_dot_vk, -sigmoid_slope)) * (
                    deltaq[Nnot[i, j]] * nt_dot_vk)
    return n, hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq


def get_reduced_hyperplane_parameters(JE, deltaq, active_joints, sigmoid_slope=0):
    # %getHyperplanes Gets the hyperplanes parameters
    # %   gets the parameters that can be used to construct the cartesian
    # %   velocitity polytopes using hyperplane method as well as their gradients,
    print('Entering Reduced hyperplane parameters')
    m = np.shape(JE)[0]  # number of task space degrees of freedom
    number_of_joints = np.shape(JE)[1]  # number of joints
    active_joints = np.arange(number_of_joints)  # listing our the joints
    N, Nnot = robot_functions.getDofCombinations(active_joints, m)
    number_of_combinations = np.shape(N)[0]

    n = np.zeros([number_of_combinations, m])

    hplus = np.zeros([number_of_combinations, ])
    hminus = np.zeros([number_of_combinations, ])

    v1 = np.zeros([3, ])
    v2 = np.zeros([3, ])

    vk = np.zeros([3, ])

    for i in range(np.shape(N)[0]):
        v1 = JE[:, N[i, 0]]
        v2 = JE[:, N[i, 1]]
        n[i, :] = robot_functions.cross_product_normalized(v1, v2)

        hplus[i] = 0
        hminus[i] = 0

        for j in range(number_of_joints - (m - 1)):
            vk = JE[:, Nnot[i, j]]
            
            print('vk is as such')
            print(vk)
            nt_dot_vk = np.matmul(vk, n[i, :])

            print('nt_dot_vk')
            print(nt_dot_vk)
            print('h_plus before update')
            print(hplus[i])
            print('whole hplus before update')
            print(hplus)
            
            
            print('h_minus before update')
            print(hminus[i])
            print('whole hminus before update')
            print(hminus)
            
            
            
            hplus[i] = hplus[i] + (robot_functions.sigmoid(nt_dot_vk, sigmoid_slope)) * (deltaq[Nnot[i, j]] * nt_dot_vk)
            
            print('h_plus after update')
            print(hplus[i])
            print('whole hplus after update')
            print(hplus)
            hminus[i] = hminus[i] + (robot_functions.sigmoid(nt_dot_vk, -sigmoid_slope)) * (
                    deltaq[Nnot[i, j]] * nt_dot_vk)
            
            
                        
            print('h_minus afterupdate')
            print(hminus[i])
            print('whole hminus after update')
            print(hminus)
    print('Exit Reduced hyperplane parameters')
    return n, hplus, hminus



def get_gamma_hat(JE, H, qdot_max, qdot_min, vertices, sigmoid_slope=0):
    number_of_joints = np.shape(JE)[1]
    d_gamma_hat_dq = np.zeros([number_of_joints, ])

    Gamma_plus, Gamma_minus, d_Gamma_plus_dq, d_Gamma_minus_dq = get_gamma(JE, H, qdot_max, qdot_min, vertices,
                                                                           sigmoid_slope)
    # Switching the minus here since in capacity margin we get min(Gamma_plus,-Gamma_minus) so here we get max(-Gamma_plus,Gamma_minus)
    Gamma_all = np.vstack([Gamma_plus, -Gamma_minus]) #  Stacking
    
    #print('Gamma_plus',Gamma_plus)
    #print('Gamma_minus',Gamma_minus)
    #input('stop hereeeeeeeeeeeee')
    d_Gamma_all_dq = np.vstack([d_Gamma_plus_dq, -d_Gamma_minus_dq])

    index_of_min_value = np.unravel_index(np.argmax(Gamma_all, axis=None), Gamma_all.shape)  # Get's the index of the minimum value

    gamma_hat = -robot_functions.smooth_max(Gamma_all*sigmoid_slope)/sigmoid_slope
    #print('Gamma_all is', Gamma_all)
    #print('gamma_hat isssssssss')
    #print(gamma_hat)
    # Gradient of all gammas w.r.t gamma
    d_gamma_hat_d_gamma = robot_functions.exp_normalize(Gamma_all)

    for i in range(d_Gamma_all_dq.shape[-1]):
        # take the minimum value and multiple by it's gradient only
        d_gamma_hat_dq[i] = d_gamma_hat_d_gamma[index_of_min_value] * d_Gamma_all_dq[index_of_min_value[0], index_of_min_value[1], i]
    
    #print('Gamma_all_max isssssssss')
    #print(-np.max(Gamma_all))

    return gamma_hat, d_gamma_hat_dq, -np.max(Gamma_all)


def get_gamma(JE, H, qdot_max, qdot_min, vertices, sigmoid_slope=0):
    number_of_joints = np.shape(JE)[1]  # number of joints
    deltaq = qdot_max - qdot_min

    n, hplus, hminus, d_n_dq, d_hplus_dq, d_hminus_dq \
        = get_hyperplane_parameters(JE, H, deltaq, sigmoid_slope)
        
    
    number_of_joints = np.shape(JE)[1]  # number of joints
    Gamma_plus = np.zeros([np.shape(hplus)[0], np.shape(vertices)[0]])
    Gamma_minus = np.zeros([np.shape(hminus)[0], np.shape(vertices)[0]])
    d_Gamma_plus_dq = np.zeros([np.shape(hplus)[0], np.shape(vertices)[0], number_of_joints])
    d_Gamma_minus_dq = np.zeros([np.shape(hminus)[0], np.shape(vertices)[0], number_of_joints])

    for vertex in range(np.shape(vertices)[0]):
        Gamma_plus[:, vertex] = hplus + np.matmul(np.matmul(n, JE), qdot_min) - np.matmul(n, np.transpose(
            vertices[vertex, :]))
        Gamma_minus[:, vertex] = hminus + np.matmul(np.matmul(n, JE), qdot_min) + np.matmul(n, np.transpose(
            vertices[vertex, :]))

        for joint in range(number_of_joints):
            d_Gamma_plus_dq[:, vertex, joint] = d_hplus_dq[:, joint] \
                                                + np.matmul(np.matmul(d_n_dq[:, :, joint], JE), qdot_min) + \
                                                np.matmul(np.matmul(n, H[0:3, :, joint]), qdot_min) - (np.matmul(d_n_dq[:, :, joint], np.transpose(vertices[vertex, :])))
            d_Gamma_minus_dq[:, vertex, joint] = d_hminus_dq[:, joint] \
                                                 + np.matmul(np.matmul(d_n_dq[:, :, joint], JE), qdot_min) + \
                                                 np.matmul(np.matmul(n, H[0:3, :, joint]), qdot_min) + (
                                                     np.matmul(d_n_dq[:, :, joint], np.transpose(vertices[vertex, :])))

    return Gamma_plus, Gamma_minus, d_Gamma_plus_dq, d_Gamma_minus_dq


