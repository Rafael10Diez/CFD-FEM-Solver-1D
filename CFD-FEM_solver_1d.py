#
# Programming solution for the final assignment in the course:
#
#     Computational Fluid Dynamics II, J.M. Burgerscentrum, Research School for Fluid Dynamics
#     
#     Professors:
#                   Prof.dr.ir. F.J. Vermolen
#                   Prof.dr.ir. C. Vuik
#
# Student name : Rafael Diez
# Date         : 20/01/2021
# Topic        : Exercise 17: One dimensional finite element code (CFD)
#
# Reference Links:
#     - http://homepage.tudelft.nl/d2b4e/burgers/burg.html
#     - http://homepage.tudelft.nl/d2b4e/burgers/exercises.pdf

import numpy as np

class Derive_S_elem:
    def __init__(self):
        
        # Derive FEM Formulation for the stiffness matrix and the force vector.
        
        from sympy import init_printing
        init_printing()
        #
        import sympy as sp
        symbols  =  sp.symbols
        sin      =  sp.sin
        
        # PDE:
        #     -D * d2U + lambda * U = f(x)
        
        # Standard Galerkin:
        #    Integral( D * grad_u * grad_v + Beta * u * v ) = Integral( f * v )
        
        # Force Vector has the form:
        #     f = a*sin(20x) + b,
        #     where the constants (a,b) can be adjusted to match the specified running cases [f=1, f=sin(20x)]
        
        D,beta,u1,u2,chi,x,x_0,x_1,dh,a,b,Jac =  map(symbols,'D,beta,u1,u2,chi,x,x_0,x_1,dh,a,b,Jac'.split(','))
        #
        diff_x       = lambda var: var.diff(chi)*Jac # dvar_dchi * dchi_dx
        
        chi_x        = (x-x_0)/dh
        
        f            = a*sin(20*x) + b
        
        N1,N2  =  N  =  [ 1-chi      , chi        ]
        
        grad_N       =  [ diff_x(N1) , diff_x(N2) ]
        
        u            = N1*u1 + N2*u2
        grad_u       = diff_x(u)
        
        S,B = [],[]
        
        for Ni,grad_Ni in zip(N, grad_N):
            #
            int_vol  = D*grad_u*grad_Ni + beta*u*Ni
            int_f    = (f*Ni).replace(chi, chi_x )
            #
            Ai    = sp.integrate(int_vol*dh,(chi,0  ,1  ))
            bi    = sp.integrate(int_f     ,(x  ,x_0,x_1))
            #
            si0   = Ai.subs(u1,1).subs(u2,0)
            si1   = Ai.subs(u1,0).subs(u2,1)
            S.append( [ si0, si1 ] )
            B.append( bi )
        
        # Sij:  [[D*Jac**2*dh + beta*dh/3, -D*Jac**2*dh + beta*dh/6], [-D*Jac**2*dh + beta*dh/6, D*Jac**2*dh + beta*dh/3]]
        # bi :  [a*cos(20*x_0)/20 - a*cos(20*x_1)/20 - a*x_0*cos(20*x_1)/(20*dh) + a*x_1*cos(20*x_1)/(20*dh) + a*sin(20*x_0)/(400*dh) - a*sin(20*x_1)/(400*dh) - b*x_0 + b*x_1 - b*x_0**2/(2*dh) + b*x_0*x_1/dh - b*x_1**2/(2*dh), -(a*sin(20*x_0)/400 - b*x_0**2/2)/dh + (a*x_0*cos(20*x_1)/20 - a*x_1*cos(20*x_1)/20 + a*sin(20*x_1)/400 - b*x_0*x_1 + b*x_1**2/2)/dh]
        print('Sij: ',S)
        print('bi : ',B)
#
class FemDiffusionReaction1D:
    # FEM Solver
    #     Diffusion-Reaction Equation:
    #         - D * d2U_dx2 + lambda * U = f
    #     BCs:
    #         - D*dU_dx = 0 at x=[0,1]
    def __init__(self,n,D,beta,f_ab):
        self.n    =  n
        self.D    =  D
        self.beta =  beta
        self.a    =  f_ab[0]
        self.b    =  f_ab[1]
        self.dh   =  1/(n-1)
    
    # Assignment 1:
    #     Integral( D * grad_u * grad_v + Beta * u * v ) = Integral( f * v )
    
    # Assignment 2:
    #     Sij: Integral( D * grad_Nj * grad_Ni + Beta * Nj * Ni )
    #     f  : Integral( f * Ni )
    
    def GenerateMesh(self):
        # Assignment 3
        n = self.n
        return np.linspace(0,1,n)
    
    def GenerateTopology(self):
        # Assignment 4
        n = self.n
        
        return [ [i,i+1] for i in range(n-1) ]
    
    def compute_S_elem(self):
        # Assignment 5
        print('Assignment 5: Derive S_elem')
        Derive_S_elem()
    
    def GenerateElementMatrix(self):
        # Assignment 6
        n        =  self.n   
        D        =  self.D   
        beta     =  self.beta
        a        =  self.a   
        b        =  self.b   
        dh       =  self.dh
        Jac      =  1./dh
        #
        sin,cos = np.sin, np.cos
        
        # Note: copy-pasted expression from the output above
        self.Sij = [[D*Jac**2*dh + beta*dh/3, -D*Jac**2*dh + beta*dh/6], [-D*Jac**2*dh + beta*dh/6, D*Jac**2*dh + beta*dh/3]]
        self.Bi  = lambda x_0,x_1: [a*cos(20*x_0)/20 - a*cos(20*x_1)/20 - a*x_0*cos(20*x_1)/(20*dh) + a*x_1*cos(20*x_1)/(20*dh) + a*sin(20*x_0)/(400*dh) - a*sin(20*x_1)/(400*dh) - b*x_0 + b*x_1 - b*x_0**2/(2*dh) + b*x_0*x_1/dh - b*x_1**2/(2*dh), -(a*sin(20*x_0)/400 - b*x_0**2/2)/dh + (a*x_0*cos(20*x_1)/20 - a*x_1*cos(20*x_1)/20 + a*sin(20*x_1)/400 - b*x_0*x_1 + b*x_1**2/2)/dh]
        
        return self.Sij
    
    def AssembleMatrix(self):
        # Assignment 7
        n     = self.n
        S     = np.zeros((n,n))
        
        Sij  = self.GenerateElementMatrix()
        emat = self.GenerateTopology()
        
        for i in range(n-1):
            for j in range(2):
                for k in range(2):
                    S[emat[i][j],emat[i][k]] += Sij[j][k]
        
        return S
    
    def GenerateElementVector(self):
        # Assignment 8
        #     - vector Bi was already generated before
        return self.Bi
    
    def AssembleVector(self):
        # Assignment 9
        n   = self.n
        dh  = self.dh
        B   = np.zeros(n)
        
        emat = self.GenerateTopology()
        
        bij = self.GenerateElementVector()
        
        for i in range(n-1):
            x_0,x_1 = i*dh, (i+1)*dh
            for j in range(2):
                B[emat[i][j]] += bij(x_0,x_1)[j]
        
        return B
    
    def get_S_b(self):
        # Assignment 10
        S = self.AssembleMatrix()
        B = self.AssembleVector()
        return S,B
    def femsolve1d(self):
        # Assignment 11
        
        S,B = self.get_S_b()
        n   = self.n
        
        x   = np.linspace    (0,1,n)
        u   = np.linalg.solve(S,B)
        
        return x,u

class ExactSolution:
    # Assignment 13: Derive Analytical Solution
    def derivation(self):
        
        import sympy as sp
        
        symbols             =  sp.symbols
        integrate           =  sp.integrate
        Derivative          =  sp.Derivative
        Function            =  sp.Function
        dsolve              =  sp.dsolve
        sin                 =  sp.sin
        
        # - D*Lapla(U) + beta*U = f
        
        beta,x,a,b,D        =  map(symbols,'beta,x,a,b,D'.split(','))
        U                   =  Function('U')(x)
        #
        f                   =  a*sin(20*x) + b
        d2U                 =  Derivative(U,x,x)
        R2                  =  -D * d2U + beta * U - f
        
        R0                  = dsolve( R2 , U)
        print(R0)
        # Eq(U(x), C1*exp(-x*sqrt(beta/D)) + C2*exp(x*sqrt(beta/D)) + a*sin(20*x)/(400*D + beta) + b/beta)
        
    def find_constants(self):
        
        import sympy as sp
        
        symbols             =  sp.symbols
        sqrt                =  sp.sqrt
        sin                 =  sp.sin
        exp                 =  sp.exp
        beta,x,a,b,D,C1,C2  =  map(symbols,'beta,x,a,b,D,C1,C2'.split(','))
        U_sol               =  C1*exp(-x*sqrt(beta/D)) + C2*exp(x*sqrt(beta/D)) + a*sin(20*x)/(400*D + beta) + b/beta
        #
        Eqs                 =  []
        Eqs                +=  [ U_sol.diff(x).subs(x,0) ]
        Eqs                +=  [ U_sol.diff(x).subs(x,1) ]
        C_sols              =  sp.solve(Eqs,C1,C2)
        
        # C1: 20*a*exp(2*sqrt(beta/D))/(400*D*sqrt(beta/D)*exp(2*sqrt(beta/D)) - 400*D*sqrt(beta/D) + beta*sqrt(beta/D)*exp(2*sqrt(beta/D)) - beta*sqrt(beta/D)) - 20*a*exp(sqrt(beta/D))*cos(20)/(400*D*sqrt(beta/D)*exp(2*sqrt(beta/D)) - 400*D*sqrt(beta/D) + beta*sqrt(beta/D)*exp(2*sqrt(beta/D)) - beta*sqrt(beta/D))
        # C2: -20*a*exp(sqrt(beta/D))*cos(20)/(400*D*sqrt(beta/D)*exp(2*sqrt(beta/D)) - 400*D*sqrt(beta/D) + beta*sqrt(beta/D)*exp(2*sqrt(beta/D)) - beta*sqrt(beta/D)) + 20*a/(400*D*sqrt(beta/D)*exp(2*sqrt(beta/D)) - 400*D*sqrt(beta/D) + beta*sqrt(beta/D)*exp(2*sqrt(beta/D)) - beta*sqrt(beta/D))
        print(C_sols)
    
    def get_u_exact(self,x,beta,D,a,b):
        exp, sqrt, sin, cos = np.exp, np.sqrt, np.sin, np.cos
        C1      = 20*a*exp(2*sqrt(beta/D))/(400*D*sqrt(beta/D)*exp(2*sqrt(beta/D)) - 400*D*sqrt(beta/D) + beta*sqrt(beta/D)*exp(2*sqrt(beta/D)) - beta*sqrt(beta/D)) - 20*a*exp(sqrt(beta/D))*cos(20)/(400*D*sqrt(beta/D)*exp(2*sqrt(beta/D)) - 400*D*sqrt(beta/D) + beta*sqrt(beta/D)*exp(2*sqrt(beta/D)) - beta*sqrt(beta/D))
        C2      = -20*a*exp(sqrt(beta/D))*cos(20)/(400*D*sqrt(beta/D)*exp(2*sqrt(beta/D)) - 400*D*sqrt(beta/D) + beta*sqrt(beta/D)*exp(2*sqrt(beta/D)) - beta*sqrt(beta/D)) + 20*a/(400*D*sqrt(beta/D)*exp(2*sqrt(beta/D)) - 400*D*sqrt(beta/D) + beta*sqrt(beta/D)*exp(2*sqrt(beta/D)) - beta*sqrt(beta/D))
        U_exact =  C1*exp(-x*sqrt(beta/D)) + C2*exp(x*sqrt(beta/D)) + a*sin(20*x)/(400*D + beta) + b/beta
        return U_exact

def plot_center(n,D,beta,f_ab,title=''):
    
    # 1) call FEM solver
    
    x,u           = FemDiffusionReaction1D(n,D,beta,f_ab).femsolve1d()
    
    # 2) call analytical solution
    
    get_u_exact   = ExactSolution().get_u_exact
    a,b           = f_ab
    x_ref         = np.linspace(0,1,5*n)
    U_ref         = get_u_exact(x_ref,beta,D,a,b)
    
    # 3) plot everything
    
    plt.plot(x    , u                        ,linewidth = 5 ,label = 'FEM')
    plt.plot(x_ref, U_ref,'r',linestyle = ':',linewidth = 5 ,label = 'Exact Sol.')
    
    d = 0.005
    
    plt.xlim(0,1)
    plt.ylim(min(U_ref)-d, max(U_ref)+d)
    plt.xlabel('x')
    plt.ylabel('U')
    plt.title(title, y=1.02)
    plt.legend()
    plt.grid()
    plt.show()

if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18})
    
    # Assignment 12:
    # f = 1
    # f = 0*sin(20x) + 1
    n,D,beta,f_ab = 100, 1, 1, (0,1)
    plot_center(n,D,beta,f_ab,'Case 1\n f(x) = 1, n = %s' % n)
    
    # Assignment 13:
    # f = sin(20x)
    # f = 1*sin(20x) + 0
    n,D,beta,f_ab = 100, 1, 1, (1,0)
    plot_center(n,D,beta,f_ab,'Case 2\n f(x) = sin(20x), n = %s' % n)