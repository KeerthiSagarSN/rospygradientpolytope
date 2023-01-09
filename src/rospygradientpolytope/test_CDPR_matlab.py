from numpy import ones,array,shape,zeros,matmul,transpose,hstack,vstack,linspace
#from numpy.linalg import qr
from scipy.linalg import qr
from linearalgebra import V_unit,check_ndarray,skew_2D
import matplotlib.pyplot as plt


def boxLinearMappingFacets(Wm,t_bounds,combMatrix):


    n,m = shape(Wm)

    p = shape(combMatrix)[1]

    Am = zeros(shape=(2*p,n))
    bv = zeros(shape=(2*p,1))

    Wm = -Wm

    print('Wm as',Wm)
    print('Wm shape is',shape(Wm))
    counter = 0
    for kp in range(p):
        
        #if (kp%2 == 0):

        print('combMatrix',combMatrix)
        print('combMatrix[:,kp]',combMatrix[:,kp-1])
        Wmk = Wm[:,combMatrix[:,kp-1]]

        

        print('Wmk is',Wmk)
        [Qmk,Rmk] = qr(Wmk)
        print('Qmk',Qmk)
        print('Wmk',Wmk)
        avk = Qmk[:,-1]
        #avk = Wmk
        print('avk',avk)
        WmTavk = matmul(transpose(Wm),avk)

        print('lk matlab is',WmTavk)

        bku=0 # --> h+ parameter
        bkl=0 #; --> h- parameter



        for i in range(m):
            if WmTavk[i] >= 0:
                bku = bku + WmTavk[i]*t_bounds[i,1]
                bkl = bkl + WmTavk[i]*t_bounds[i,0]

            else:
                bku = bku + WmTavk[i]*t_bounds[i,0]
                bkl = bkl + WmTavk[i]*t_bounds[i,1]
            
        print('h+ parameter is',bku)
        print('h- parameter is',bkl)
        Am[counter,:] = transpose(avk)
        bv[counter] = bku ## --> nTp+ term

        counter += 1
        Am[counter,:] = -transpose(avk)
        bv[counter] = -bkl ## -- > nTp-term

        

        counter += 1
    print('nTp term is',bv)
    #Am = vstack((Am,-Am))
    #bv = vstack((bv,-bv))

        #Am[kp+1,:] = -transpose(avk)
        #bv[kp+1] = -bkl


    print('Am is',Am)

    #input('test componennts here')


    return Am, bv


def CapacityMargin(wvTvertices,Amt,bvt):

    p = shape(wvTvertices)[1]

    print('shape of bvt is',shape(bvt))
    print('p is',p)
    print('shape of Amt is',shape(Amt))
    print('shape of wvTvertices is',shape(wvTvertices))


    Sm = bvt*ones(shape=(1,p)) - matmul(Amt,wvTvertices)

    Sm_flat_array = check_ndarray(Sm)

    print('Gamma total is',Sm_flat_array)

    s = min(Sm_flat_array)

    return s

if __name__ == '__main__':
    mcab = 4 #; % number of cables
    n_dof = 2 # ;
    mass = 5.0
    tlb = 1 #; %[N]
    tub = 25 #; %[N]
    tmin = tlb*ones( shape = (mcab,1)) #;
    tmax = tub*ones(shape = (mcab,1)) #; Tension bounds are here
    t_bounds = hstack((tmin,tmax))

    # Wrench matrix to test here is
    #W = array([[-0.7071,-0.7071,-0.7071,-0.7071],[0.7071,0.7071,0.7071,0.7071]])
    #base_points = array([[-1,-1],[-1,1],[1,1],[1,-1]])
    base_points = array([[0,0],[0,1],[1,1],[1,0]])

    pos_bounds = array([[min(base_points[:,0]),max(base_points[:,0])],[min(base_points[:,1]),max(base_points[:,1])]])

    print('pos_bpunds',pos_bounds)
    q_actual = array([[-10000,-10000]])
    q_estimated = array([[-10000,-10000]])
        

    q_feasible = array([[-10000,-10000]])
    q_infeasible = array([[-10000,-10000]])
        
    step_size = 100
    q_in_x = linspace(pos_bounds[0,0],pos_bounds[0,1],step_size)
    print('q_in_x',q_in_x)
    q_in_y = linspace(pos_bounds[1,0],pos_bounds[1,1],step_size)
    print('q_in_y',q_in_y)
    for i in range(len(q_in_x)):
        x_in = q_in_x[i]

        x_in = 0.5
        
        for j in range(len(q_in_y)):
            y_in = q_in_y[j]

            y_in = 1.0
            q = array([x_in,y_in])

            

            #c1 = (ef - b1)/norm(ef - b1)

            Wm = zeros(shape=(2,4))
            for k in range(len(base_points)):
                cable_plt = array([[x_in,base_points[k,0]],[y_in,base_points[k,1]]])
                Wm[0,k] = base_points[k,0] - x_in 
                Wm[1,k] = base_points[k,1] - y_in

                print('self.base_points',base_points[k,:])
                

                #input('stop and check')

                #Wm[0,k] = W[0,k]*((norm(W[:,k]))**(-1))
                #Wm[1,k] = W[1,k]*((norm(W[:,k]))**(-1))

                Wm[:,k] = V_unit(Wm[:,k])
                test_arr = transpose(array([Wm[:,k]]))
                print('test_arr',test_arr)
                [Qmk,Rmk] = qr(test_arr )
                print('Qmk',Qmk)
                print('Wm',Wm)
                #input('Check Wrench matrix here')
        
            #W = array([[-0.7071,0.7071], [0.7071, 0.7071],[-0.7071,0.7071],[0.7071,0.7071]]) # wrench matrix for a fully suspended CDPR
            #Wm = array([[-0.4471,1.0,-1.0,-0.4472], [0.8944,0,0,0.8944]]) # wrench matrix for a fully suspended CDPR
            #w_e = array([[0],[-mass*9.8]])

            w_e = transpose(array([[-5,-5],[-5,5],[5,5],[5,-5]]))


            # Combination matrix are here

            combMatrix = array([[0,1,2,3]]) # Combinatino matrix is here


            Amt,bvt = boxLinearMappingFacets(Wm,t_bounds,combMatrix)

            CM = CapacityMargin(w_e,Amt,bvt)
            print('CM is',CM)
            #input('stop here')

            tol_value_abs = 5e-1
            tol_value = 5e-1

            

            if (CM) > tol_value :
                print('feasible point')
                q_feasible = vstack((q_feasible,q))       
                
                
                
            elif (CM) < -tol_value :

                print('infeasible point')
                q_infeasible = vstack((q_infeasible,q))
            
            if (abs(CM) < tol_value_abs):
                
                print('boundary point')
                #input('boundary point here')
                '''
                polytope_vertices, polytope_faces, facet_pair_idx, capacity_margin_faces, \
                    capacity_proj_vertex, polytope_vertices_est, polytope_faces_est, capacity_margin_faces_est, capacity_proj_vertex_est = \
                        force_polytope_2D(W,self.qdot_min, self.qdot_max,self.cartesian_desired_vertices,self.sigmoid_slope)
                '''
                #figure1.canvas.draw()
                #print('inside WFW - actual')
                
                #input('stop here')
                #print('Feasiblyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                
                #scaling_factor = 0.005
                #cartesian_desired_vertices_plt = (scaling_factor*self.cartesian_desired_vertices) + array([[x_in,y_in]])

                #plt.plot(cartesian_desired_vertices_plt[:,0], cartesian_desired_vertices_plt[:,1],color = 'green')

                #print('polytope_Vertices before',polytope_vertices)
                #polytope_vertices = vstack((polytope_vertices,polytope_vertices[0,:]))
                #polytope_vertices_plt = (scaling_factor*polytope_vertices) + array([[x_in,y_in]])

                #print('polytope_Vertices after',polytope_vertices)
                
                #plt.plot(polytope_vertices_plt[:,0], polytope_vertices_plt[:,1],color = 'k')

                #print('polytope_Vertices offset',polytope_vertices)
                #print('polytope_faces',polytope_faces)

                #print('facet_pair_idx',facet_pair_idx)
                #print('capacity_margin_faces',capacity_margin_faces)
                #print('capacity_proj_vertex',capacity_proj_vertex)
                #desired_vertex_set = plt.scatter(cartesian_desired_vertices_plt[:,0], cartesian_desired_vertices_plt[:,1],color = 'k')
                
                '''
                for i in range(len(n_k)):
                    
                    plt.plot([x_in,1*n_k[i,0]],[y_in,1*n_k[i,1]],color=color_arr[i])
                '''
                #plt.cla()

                #figure1.canvas.flush_events()


                q_actual = vstack((q_actual,q))
            
            
            
    q_actual = q_actual[1:,:]
    q_feasible = q_feasible[1:,:]
    q_infeasible = q_infeasible[1:,:]

    plt.plot(base_points [:,0], base_points [:,1],color = 'cyan')


    #plt_estimate = plt.scatter(q_estimated[:,0], q_estimated[:,1],color = 'cyan',s=2.5)

    #plt_estimate = plt_actual

    plt_feasible = plt.scatter(q_feasible[:,0], q_feasible[:,1],color = 'green',s=10.11)
    plt_infeasible = plt.scatter(q_infeasible[:,0], q_infeasible[:,1],color = 'red',s=10.11)
    plt_actual = plt.scatter(q_actual[:,0], q_actual[:,1],color = 'k',s=10.0)
    plt.show()


