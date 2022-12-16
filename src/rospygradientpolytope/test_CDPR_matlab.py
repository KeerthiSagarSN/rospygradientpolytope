from numpy import ones,array,shape,zeros,matmul,transpose,hstack,vstack
from numpy.linalg import qr
from linearalgebra import V_unit,check_ndarray,skew_2D


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
        test_skew = skew_2D(Wmk)
        print('test_skew is',test_skew)
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

    input('test componennts here')


    return Am, bv


def CapacityMargin(wvTvertices,Amt,bvt):

    p = shape(wvTvertices)[1]

    print('shape of bvt is',shape(bvt))
    print('p is',p)
    print('shape of Amt is',shape(Amt))
    print('shape of wvTvertices is',shape(wvTvertices))


    Sm = bvt*ones(shape=(1,p)) - matmul(Amt,wvTvertices)

    print('Gamma total is',Sm)

    s = min(min(Sm))

    return s

if __name__ == '__main__':
    mcab = 4 #; % number of cables
    n_dof = 2 # ;
    mass = 5 #[kg]
    tlb = 0 #; %[N]
    tub = 25 #; %[N]
    tmin = tlb*ones( shape = (mcab,1)) #;
    tmax = tub*ones(shape = (mcab,1)) #; Tension bounds are here
    t_bounds = hstack((tmin,tmax))

    # Wrench matrix to test here is
    W = array([[-0.7071,0.7071,-0.7071,0.7071],[0.7071,0.7071,0.7071,0.7071]])
    #W = array([[-0.7071,0.7071], [0.7071, 0.7071],[-0.7071,0.7071],[0.7071,0.7071]]) # wrench matrix for a fully suspended CDPR

    w_e = array([[0],[-mass*9.8]]) 


    # Combination matrix are here

    combMatrix = array([[0,1,2,3]]) # Combinatino matrix is here


    Amt,bvt = boxLinearMappingFacets(W,t_bounds,combMatrix)

    CM = CapacityMargin(w_e,Amt,bvt)

    print(CM)


