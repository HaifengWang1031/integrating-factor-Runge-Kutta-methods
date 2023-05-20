from typing import Tuple
import numpy.typing as tnp

import numpy as np
import scipy.linalg as linalg
from scipy import sparse 
import scipy.sparse.linalg as slinalg
import scipy.fft as fft

import tqdm

class Allen_Cahn_Solver_1D():
    def __init__(
        self,
        eps:float,
        t_period:float,
        s_domain:Tuple[float,float],
        time_step:float,
        discrete_num: int,
        initial_condition: tnp.NDArray,
        boundary_condition: str,
        potential: str,
        step_method: str) -> None:

        self.eps = eps

        self.T = t_period
        self.tau = time_step
        self.tn = np.array([0])

        self.xa,self.xb = s_domain
        self.N = discrete_num
        self.h = (self.xb-self.xa)/self.N
        self.xh = np.linspace(self.xa,self.xb,self.N+1)

        u0 = initial_condition

        self.bc = boundary_condition
        if self.bc == "periodic":
            self.L = self.eps**2*self.laplace_periodic()

            self.G = self.gradient_periodic()
            self.U = u0[:-1][np.newaxis,:]

        elif self.bc == "neumann":
            self.L = self.eps**2*self.laplace_neumann()

            self.G = self.gradient_neumann()
            self.U = u0[np.newaxis,:]

        if potential == "flory_huggins":
            self.F,self.f = self.fh_potential()
        elif potential == "ginzburg_landau":
            self.F,self.f = self.gl_potential()
        
        self.Energys = np.array(self.energy(self.U[0]))

        if step_method == "IFRK1":
            self.step = self.IFRK1
            self.expM = linalg.expm(self.tau*self.L)

        elif step_method == "IFRK2":
            self.step = self.IFRK2
            self.expM = linalg.expm(self.tau*self.L)

        elif step_method == "IFRK3":
            self.step = self.IFRK3
            self.expM = linalg.expm(self.tau*self.L)
            self.expM13 = linalg.expm(1/3*self.tau*self.L)
            self.expM23 = linalg.expm(2/3*self.tau*self.L)

        elif step_method == "Shu_Osher3":
            self.step = self.Shu_Osher3
            self.expM = linalg.expm(self.tau*self.L)
            self.expM12 = linalg.expm(1/2*self.tau*self.L)
            self.expmM12 = linalg.expm(-1/2*self.tau*self.L)

        elif step_method == "IFRK4":
            self.step = self.IFRK4
            self.expM = linalg.expm(self.tau*self.L)
            self.expM12 = linalg.expm(1/2*self.tau*self.L)
        
        elif step_method == "IFRK54":
            self.step = self.IFRK54 
            c0 = 0; c1 = 0.4549; c2 = c3 = 0.5165; c4 = 0.9903; c5 = 1
            self.expM10 = linalg.expm((c1-c0)*self.tau*self.L) 

            self.expM20 = linalg.expm((c2-c0)*self.tau*self.L) 
            self.expM21 = linalg.expm((c2-c1)*self.tau*self.L) 

            self.expM30 = linalg.expm((c3-c0)*self.tau*self.L)
            self.expM32 = linalg.expm((c3-c2)*self.tau*self.L)
            
            self.expM40 = linalg.expm((c4-c0)*self.tau*self.L)
            self.expM43 = linalg.expm((c4-c3)*self.tau*self.L)

            self.expM50 = linalg.expm((c5-c0)*self.tau*self.L)
            self.expM51 = linalg.expm((c5-c1)*self.tau*self.L)
            self.expM53 = linalg.expm((c5-c3)*self.tau*self.L)
            self.expM54 = linalg.expm((c5-c4)*self.tau*self.L)
        
        elif step_method == "EFRK1":
            self.step = self.EFRK1
            self.phi0 = np.eye(self.N)

            self.phi1 = self.phi0@(np.eye(self.N) - self.tau*self.L)
            self.phiM10 = np.linalg.solve(self.phi1,self.phi0)
        
        elif step_method == "EFRK2":
            self.step = self.EFRK2
            self.phi0 = np.eye(self.N)

            self.phi1 = self.phi0@(np.eye(self.N) - self.tau*self.L)
            self.phiM10 = np.linalg.solve(self.phi1,self.phi0)

            self.phi2 = 1/2*self.phi0 + 1/2*self.phi1@(np.eye(self.N) - self.tau*self.L) 
            self.phiM20 = np.linalg.solve(self.phi2,self.phi0)
            self.phiM21 = np.linalg.solve(self.phi2,self.phi1)
        
        elif step_method == "EFRK3":
            self.step = self.EFRK3

            self.phi0 = np.eye(self.N)
            self.phi1 = self.phi0@(np.eye(self.N) - 2/3*self.tau*self.L)
            self.phi2 = 2/3*self.phi0 +\
                        1/3*self.phi1@(np.eye(self.N) - 4/3*self.tau*self.L)
            self.phi3 = 59/128*self.phi0 +\
                        15/128*self.phi0@(np.eye(self.N) - 4/3*self.tau*self.L) +\
                        27/64*self.phi2@(np.eye(self.N) - 4/3*self.tau*self.L)
            
            self.phiM10 = np.linalg.solve(self.phi1,self.phi0)

            self.phiM20 = np.linalg.solve(self.phi2,self.phi0)
            self.phiM21 = np.linalg.solve(self.phi2,self.phi1)

            self.phiM30 = np.linalg.solve(self.phi3,self.phi0)
            self.phiM31 = np.linalg.solve(self.phi3,self.phi1)
            self.phiM32 = np.linalg.solve(self.phi3,self.phi2)
        
        elif step_method == "EFRK4":
            self.step = self.EFRK4
            self.phi0 = np.eye(self.N)

            self.phi1 = self.phi0@(np.eye(self.N) - 1/2*self.tau*self.L)
            self.phiM10 = np.linalg.solve(self.phi1,self.phi0)

            self.phi2 = 1/2*self.phi0@(np.eye(self.N) + 1/2*self.tau*self.L) + 1/2*self.phi1@(np.eye(self.N) - self.tau*self.L)
            self.phiM20 = np.linalg.solve(self.phi2,self.phi0)
            self.phiM21 = np.linalg.solve(self.phi2,self.phi1)

            self.phi3 = 1/9*self.phi0@(np.eye(self.N) + self.tau*self.L) + 2/9*self.phi1@(np.eye(self.N) + 3/2*self.tau*self.L) + 2/3*self.phi2@(np.eye(self.N) - 3/2*self.tau*self.L)
            self.phiM30 = np.linalg.solve(self.phi3,self.phi0)
            self.phiM31 = np.linalg.solve(self.phi3,self.phi1)
            self.phiM32 = np.linalg.solve(self.phi3,self.phi2)

            self.phi4 = 1/3*self.phi1@(np.eye(self.N) - 1/2*self.tau*self.L) + 1/3*self.phi2@np.eye(self.N) + 1/3*self.phi3@(np.eye(self.N) - 1/2*self.tau*self.L)
            self.phiM41 = np.linalg.solve(self.phi4,self.phi1)
            self.phiM42 = np.linalg.solve(self.phi4,self.phi2)
            self.phiM43 = np.linalg.solve(self.phi4,self.phi3)

        elif step_method == "EFRK54":
            self.step = self.IFRK54 
            c0 = 0; c1 = 0.4549; c2 = c3 = 0.5165; c4 = 0.9903; c5 = 1
            r = 1.346586417284006

            self.phi0 = np.eye(self.N)
            self.phi1 = 0.387392167970373*self.phi0 + 0.612607832029627*self.phi0@(np.eye(self.N) - self.tau/r*self.L)
            self.phi2 = 0.568702484115635*self.phi0 + 0.431297515884365*self.phi1@(np.eye(self.N) - self.tau/r*self.L)
            self.phi3 = 0.589791736452092*self.phi0 + 0.410208263547908*self.phi2@(np.eye(self.N) - self.tau/r*self.L)
            self.phi4 = 0.213474206786188*self.phi0 + 0.786525793213812*self.phi3@(np.eye(self.N) - self.tau/r*self.L) 

            self.phi5 = 0.270147144537063*self.phi0 + 0.029337521506634*self.phi0@(np.eye(self.N) - self.tau/r*self.L) +\
                                                      0.239419175840559*self.phi1@(np.eye(self.N) - self.tau/r*self.L) +\
                                                      0.227000995504038*self.phi3@(np.eye(self.N) - self.tau/r*self.L) +\
                                                      0.234095162611706*self.phi4@(np.eye(self.N) - self.tau/r*self.L)

            self.phiM10 = np.linalg.solve(self.phi1,self.phi0)

            self.phiM20 = np.linalg.solve(self.phi2,self.phi0)
            self.phiM21 = np.linalg.solve(self.phi2,self.phi1)          
            
            self.phiM30 = np.linalg.solve(self.phi3,self.phi0)          
            self.phiM32 = np.linalg.solve(self.phi3,self.phi2)          
            
            self.phiM40 = np.linalg.solve(self.phi4,self.phi0)          
            self.phiM43 = np.linalg.solve(self.phi4,self.phi3)          
            
            self.phiM50 = np.linalg.solve(self.phi5,self.phi0)          
            self.phiM51 = np.linalg.solve(self.phi5,self.phi1)          
            self.phiM53 = np.linalg.solve(self.phi5,self.phi3)           
            self.phiM54 = np.linalg.solve(self.phi5,self.phi4)           

    # Linear operators 
    def laplace_periodic(self):
        L = np.diag(-2*np.ones(self.N)) +\
            np.diag(np.ones(self.N-1),1) +\
            np.diag(np.ones(self.N-1),-1)
        L[0,-1] = L[-1,0] = 1; L /= self.h**2
        return L

    def laplace_neumann(self):
        L = np.diag(-2*np.ones(self.N+1)) +\
            np.diag(np.ones(self.N),1) +\
            np.diag(np.ones(self.N),-1)
        L[0,1] += 1; L[-1,-2] += 1; L /= self.h**2
        return L

    def gradient_periodic(self):
        G = np.diag(np.ones(self.N-1),1) - np.diag(np.ones(self.N-1),-1)
        G[0,-1] = -1 ;G[-1,0] = 1; G /= self.h
        return G

    def gradient_neumann(self):
        G = np.diag(np.ones(self.N),1); G -= np.diag(np.ones(self.N),-1)
        G[0,1] -= 1; G[-1,-2] += 1; G /= self.h
        return G

    # Potential and correspond nonlinear term
    # f(u) = -F'(u)
    def gl_potential(self):
        F = lambda u: 1/4*(1-u**2)**2
        f = lambda u: (u-u**3)
        return F,f

    def fh_potential(self,theta = 0.8,theta_c = 1.6):
        F =lambda u: theta/2*((1+u)*np.log(1+u) + (1-u)*np.log(1-u)) - theta_c/2*u**2
        f =lambda u: theta/2*np.log((1-u)/(1+u)) + theta_c*u
        return F,f

    # Time step method
        # IFRK
    def IFRK1(self,Un):
        return self.expM@(Un + self.tau*self.f(Un))

    def IFRK2(self,Un):
        U1 = self.expM@(Un + self.tau*self.f(Un))
        return 1/2*self.expM@Un +\
               1/2*(U1 + self.tau*self.f(U1))

    def IFRK3(self,Un):
        U1 = self.expM23@(Un + 2*self.tau/3*self.f(Un))
        U2 = 2/3*self.expM23@Un + 1/3*(U1 + 4*self.tau/3*self.f(U1))
        return 59/128*self.expM@Un +\
               15/128*self.expM@(Un + 4/3*self.tau*self.f(Un)) +\
               54/128*self.expM13@(U2 + 4/3*self.tau*self.f(U2))
            
    def Shu_Osher3(self,Un):
        U1 = self.expM@(Un + self.tau*self.f(Un))
        U2 = 3/4*self.expM12@Un + 1/4*self.expMm12@(U1 + self.tau*self.f(U1))
        return 1/3*self.expM@Un +\
               2/3*self.expM12@(U2 + self.tau*self.f(U2))

    def IFRK4(self,Un):
        U1 = self.expM12@(Un + self.tau/2*self.f(Un))
        U2 = self.expM12@Un + self.tau/2*self.f(U1)
        U3 = self.expM@Un + self.tau*self.expM12@self.f(U2)

        return 1/3*self.expM12@(U1 + self.tau/2*self.f(U1)) +\
               1/3*self.expM12@U2 +\
               1/3*(U3 + self.tau/2*self.f(U3))
    
    def IFRK54(self,Un):
        r = 1.346586417284006
        U0 = Un
        U1 = 0.387392167970373*self.expM10@U0 + 0.612607832029627*self.expM10@(U0 + self.tau/r*self.f(U0))
        U2 = 0.568702484115635*self.expM20@U0 + 0.431297515884365*self.expM21@(U1 + self.tau/r*self.f(U1))
        U3 = 0.589791736452092*self.expM30@U0 + 0.410208263547908*self.expM32@(U2 + self.tau/r*self.f(U2))
        U4 = 0.213474206786188*self.expM40@U0 + 0.786525793213812*self.expM43@(U3 + self.tau/r*self.f(U3)) 

        return 0.270147144537063*self.expM50@U0 + 0.029337521506634*self.expM50@(U0 + self.tau/r*self.f(U0)) +\
               0.239419175840559*self.expM51@(U1 + self.tau/r*self.f(U1)) +\
               0.227000995504038*self.expM53@(U3 + self.tau/r*self.f(U3)) +\
               0.234095162611706*self.expM54@(U4 + self.tau/r*self.f(U4))

        # EFRK
    def EFRK1(self,Un):
        return self.phiM10@(Un + self.tau*self.f(Un))
    
    def EFRK2(self,Un):
        U1 = self.phiM10@(Un + self.tau*self.f(Un))
        return 1/2*self.phiM20@Un +\
               1/2*self.phiM21@(U1 + self.tau*self.f(U1))

    def EFRK3(self,Un):
        U1 = self.phiM10@(Un + 2*self.tau/3*self.f(Un))
        U2 = 2/3*self.phiM20@Un +\
             1/3*self.phiM21@(U1 + 4*self.tau/3*self.f(U1))
        return 59/128*self.phiM30@Un +\
               15/128*self.phiM30@(Un + 4/3*self.tau*self.f(Un)) +\
               54/128*self.phiM32@(U2 + 4/3*self.tau*self.f(U2))
    
    def EFRK4(self,Un):
        U0 = Un
        U1 = self.phiM10@(U0 + self.tau/2*self.f(U0))
        U2 = 1/2*self.phiM20@(U0 - self.tau/2*self.f(U0)) +\
             1/2*self.phiM21@(U1 + self.tau*self.f(U1))
        U3 = 1/9*self.phiM30@(U0 - self.tau*self.f(U0)) +\
             2/9*self.phiM31@(U1 - 3/2*self.tau*self.f(U1)) +\
             2/3*self.phiM32@(U2 + 3/2*self.tau*self.f(U2))
        return 1/3*self.phiM41@(U1 + self.tau/2*self.f(U1)) +\
               1/3*self.phiM42@U2 +\
               1/3*self.phiM43@(U3 + self.tau/2*self.f(U3))

    def EFRK54(self,Un):
        r = 1.346586417284006
        U0 = Un
        U1 = 0.387392167970373*self.phiM10@U0 + 0.612607832029627*self.phiM10@(U0 + self.tau/r*self.f(U0))
        U2 = 0.568702484115635*self.phiM20@U0 + 0.431297515884365*self.phiM21@(U1 + self.tau/r*self.f(U1))
        U3 = 0.589791736452092*self.phiM30@U0 + 0.410208263547908*self.phiM32@(U2 + self.tau/r*self.f(U2))
        U4 = 0.213474206786188*self.phiM40@U0 + 0.786525793213812*self.phiM43@(U3 + self.tau/r*self.f(U3)) 

        return 0.270147144537063*self.phiM50@U0 + 0.029337521506634*self.phiM50@(U0 + self.tau/r*self.f(U0)) +\
               0.239419175840559*self.phiM51@(U1 + self.tau/r*self.f(U1)) +\
               0.227000995504038*self.phiM53@(U3 + self.tau/r*self.f(U3)) +\
               0.234095162611706*self.phiM54@(U4 + self.tau/r*self.f(U4))
    
    def energy(self,Un):
        energy = np.nan
        if self.bc == "periodic":
           energy = self.h*(-1/2*Un@self.L@Un + self.F(Un)@np.ones(self.N))
        elif self.bc == "neumann":
           energy = self.h*np.sum((self.eps**2/2*(self.G@Un)**2 + self.F(Un))[:-1])
        return energy

    def solve(self):
        pbar = tqdm.tqdm(total=self.T+self.tau,
                         bar_format="{desc}: {percentage:.2f}% |{bar}| {n:.2f}/{total:.2f}[{elapsed}<{remaining}] {postfix}",
                         mininterval=0.1)

        while self.tn[-1] < self.T:
            u = self.step(self.U[-1])

            self.tn = np.append(self.tn,np.round(self.tn[-1]+self.tau,12))
            self.U = np.concatenate([self.U,u[np.newaxis,:]],axis=0) 
            self.Energys = np.append(self.Energys,self.energy(u))
            
            pbar.set_postfix({"time_step":self.tau,
                              "energy":self.Energys[-1],
                              "maximum_val":np.max(np.abs(u))})
            pbar.update(self.tau)
        pbar.close()

class Allen_Cahn_fSolver_1D():
    def __init__(
        self,
        eps:float,
        t_period:float,
        s_domain:Tuple[float,float],
        time_step:float,
        discrete_num: int,
        initial_condition: tnp.NDArray,
        boundary_condition: str,
        potential: str,
        step_method: str) -> None:

        self.eps = eps

        self.T = t_period
        self.tau = time_step
        self.tn = np.arange(0,self.T+self.tau,self.tau)

        self.xa,self.xb = s_domain
        self.N = discrete_num
        self.h = (self.xb-self.xa)/self.N
        self.xh = np.linspace(self.xa,self.xb,self.N+1)

        u0 = initial_condition

        self.bc = boundary_condition
        if self.bc == "periodic":
            self.L = self.eps**2*self.laplace_periodic()
            self.D = np.array([-4*self.eps**2*np.sin(k*np.pi/self.N)**2/self.h**2 for k in range(self.N)])

            self.U = np.empty((self.tn.shape[0],self.N)); self.U[0] = u0[:-1]
            self.fU = np.empty((self.tn.shape[0],self.N),dtype = np.complex128); self.fU[0] = fft.fft(u0[:-1])
        elif self.bc == "neumann":
            raise

        if potential == "flory_huggins":
            self.F,self.f = self.fh_potential()
        elif potential == "ginzburg_landau":
            self.F,self.f = self.gl_potential()
        elif potential == "ginzburg_landau_nonlocal":
            self.F,self.f = self.gl_potential_nl()
        
        self.Energys = np.empty_like(self.tn); self.Energys[0] = self.energy(self.U[0])


        if step_method == "IFRK1":
            self.step = self.IFRK1
            self.expM = np.exp(self.tau*self.D)

        elif step_method == "IFRK2":
            self.step = self.IFRK2
            self.expM = np.exp(self.tau*self.D)

        elif step_method == "IFRK3":
            self.step = self.IFRK3
            self.expM = np.exp(self.tau*self.D)
            self.expM13 = np.exp(1/3*self.tau*self.D)
            self.expM23 = np.exp(2/3*self.tau*self.D)

        elif step_method == "Shu_Osher3":
            self.step = self.Shu_Osher3
            self.expM = np.exp(self.tau*self.D)
            self.expM12 = np.exp(1/2*self.tau*self.D)
            self.expMm12 = np.exp(-1/2*self.tau*self.D)

        elif step_method == "IFRK4":
            self.step = self.IFRK4
            self.expM = np.exp(self.tau*self.D)
            self.expM12 = np.exp(1/2*self.tau*self.D)

        elif step_method == "IFRK54":
            self.step = self.IFRK54 
            c0 = 0; c1 = 0.4549; c2 = c3 = 0.5165; c4 = 0.9903; c5 = 1
            self.expM10 = np.exp((c1-c0)*self.tau*self.D) 

            self.expM20 = np.exp((c2-c0)*self.tau*self.D) 
            self.expM21 = np.exp((c2-c1)*self.tau*self.D) 

            self.expM30 = np.exp((c3-c0)*self.tau*self.D)
            self.expM32 = np.exp((c3-c2)*self.tau*self.D)

            self.expM40 = np.exp((c4-c0)*self.tau*self.D)
            self.expM43 = np.exp((c4-c3)*self.tau*self.D)

            self.expM50 = np.exp((c5-c0)*self.tau*self.D)
            self.expM51 = np.exp((c5-c1)*self.tau*self.D)
            self.expM53 = np.exp((c5-c3)*self.tau*self.D)
            self.expM54 = np.exp((c5-c4)*self.tau*self.D)

        elif step_method == "EFRK1":
            self.step = self.EFRK1
            self.phi0 = np.ones(self.N)

            self.phi1 = self.phi0*(np.ones(self.N) - self.tau*self.D)
            self.phiM10 = self.phi0/self.phi1
        
        elif step_method == "EFRK2":
            self.step = self.EFRK2
            self.phi0 = np.ones(self.N)

            self.phi1 = self.phi0*(np.ones(self.N) - self.tau*self.D)
            self.phi2 = 1/2*self.phi0 + 1/2*self.phi1*(np.ones(self.N) - self.tau*self.D) 

            self.phiM10 = self.phi0/self.phi1
            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2
        
        elif step_method == "EFRK3":
            self.step = self.EFRK3

            self.phi0 = np.ones(self.N)
            self.phi1 =     self.phi0*(np.ones(self.N) - 2/3*self.tau*self.D)
            self.phi2 = 2/3*self.phi0 +\
                        1/3*self.phi1*(np.ones(self.N) - 4/3*self.tau*self.D)
            self.phi3 = 59/128*self.phi0 +\
                        15/128*self.phi0*(np.ones(self.N) - 4/3*self.tau*self.D) +\
                        27/64 *self.phi2*(np.ones(self.N) - 4/3*self.tau*self.D)

            self.phiM10 = self.phi0/self.phi1

            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2

            self.phiM30 = self.phi0/self.phi3
            self.phiM31 = self.phi1/self.phi3
            self.phiM32 = self.phi2/self.phi3
        
        elif step_method == "EFRK4":
            self.step = self.EFRK4
            self.phi0 = np.ones(self.N)
            self.phi1 = self.phi0*(np.ones(self.N) - 1/2*self.tau*self.D)
            self.phi2 = 1/2*self.phi0*(np.ones(self.N) + 1/2*self.tau*self.D) +\
                        1/2*self.phi1*(np.ones(self.N) - self.tau*self.D)
            self.phi3 = 1/9*self.phi0*(np.ones(self.N) + self.tau*self.D) +\
                        2/9*self.phi1*(np.ones(self.N) + 3/2*self.tau*self.D) +\
                        2/3*self.phi2*(np.ones(self.N) - 3/2*self.tau*self.D)
            self.phi4 = 1/3*self.phi1*(np.ones(self.N) - 1/2*self.tau*self.D) +\
                        1/3*self.phi2*np.ones(self.N) +\
                        1/3*self.phi3*(np.ones(self.N) - 1/2*self.tau*self.D)

            self.phiM10 = self.phi0/self.phi1
            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2
            self.phiM30 = self.phi0/self.phi3
            self.phiM31 = self.phi1/self.phi3
            self.phiM32 = self.phi2/self.phi3
            self.phiM41 = self.phi1/self.phi4
            self.phiM42 = self.phi2/self.phi4
            self.phiM43 = self.phi3/self.phi4

        elif step_method == "EFRK54":
            self.step = self.EFRK54 
            c0 = 0; c1 = 0.4549; c2 = c3 = 0.5165; c4 = 0.9903; c5 = 1
            r = 1.346586417284006

            self.phi0 = np.ones(self.N)
            self.phi1 = 0.387392167970373*self.phi0 + 0.612607832029627*self.phi0*(np.ones(self.N) - self.tau/r*self.D)
            self.phi2 = 0.568702484115635*self.phi0 + 0.431297515884365*self.phi1*(np.ones(self.N) - self.tau/r*self.D)
            self.phi3 = 0.589791736452092*self.phi0 + 0.410208263547908*self.phi2*(np.ones(self.N) - self.tau/r*self.D)
            self.phi4 = 0.213474206786188*self.phi0 + 0.786525793213812*self.phi3*(np.ones(self.N) - self.tau/r*self.D) 

            self.phi5 = 0.270147144537063*self.phi0 + 0.029337521506634*self.phi0*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.239419175840559*self.phi1*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.227000995504038*self.phi3*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.234095162611706*self.phi4*(np.ones(self.N) - self.tau/r*self.D)

            self.phiM10 = self.phi0/self.phi1

            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2          

            self.phiM30 = self.phi0/self.phi3          
            self.phiM32 = self.phi2/self.phi3          

            self.phiM40 = self.phi0/self.phi4          
            self.phiM43 = self.phi3/self.phi4          

            self.phiM50 = self.phi0/self.phi5          
            self.phiM51 = self.phi1/self.phi5          
            self.phiM53 = self.phi3/self.phi5           
            self.phiM54 = self.phi4/self.phi5           

    # Linear operators 
    def laplace_periodic(self):
        L = np.diag(-2*np.ones(self.N)) +\
            np.diag(np.ones(self.N-1),1) +\
            np.diag(np.ones(self.N-1),-1)
        L[0,-1] = L[-1,0] = 1; L /= self.h**2
        return L

    # Potential and correspond nonlinear term
    # f(u) = -F'(u)
    def gl_potential(self):
        F = lambda u: 1/4*(1-u**2)**2
        f = lambda u: (u-u**3)
        return F,f

    def gl_potential_nl(self):
        F = lambda u: 1/4*(1-u**2)**2
        f = lambda u: (u-u**3) - np.mean(u-u**3)
        return F,f

    def fh_potential(self,theta = 0.8,theta_c = 1.6):
        F =lambda u: theta/2*((1+u)*np.log(1+u) + (1-u)*np.log(1-u)) - theta_c/2*u**2
        f =lambda u: theta/2*np.log((1-u)/(1+u)) + theta_c*u
        return F,f

    # Time step method
    # Forward Euler
    def residual(self,Un):
        return fft.ifft(fft.fft(Un)*self.D).real + self.f(Un)
    def FE(self,Un):
        return Un + self.tau*self.residual(Un)

        # IFRK
    def IFRK1(self,fUn,Un):
        return self.expM*(fUn + self.tau*fft.fft(self.f(Un)))

    def IFRK2(self,fUn,Un):
        fU1 = self.expM*(fUn + self.tau*fft.fft(self.f(Un)))
        return 1/2*self.expM*fUn +\
               1/2*(fU1 + self.tau*fft.fft(self.f(np.real(fft.ifft(fU1)))))

    def IFRK3(self,fUn,Un):
        fU1 = self.expM23*(fUn + 2*self.tau/3*fft.fft(self.f(Un)))
        fU2 = 2/3*self.expM23*fUn + 1/3*(fU1 + 4*self.tau/3*fft.fft(self.f(np.real(fft.ifft(fU1)))))
        return 59/128*self.expM*fUn +\
               15/128*self.expM*(fUn + 4/3*self.tau*fft.fft(self.f(Un))) +\
               54/128*self.expM13*(fU2 + 4/3*self.tau*fft.fft(self.f(fft.ifft(fU2).real)))
            
    def Shu_Osher3(self,fUn,Un):
        fU1 = self.expM*(fUn + self.tau*fft.fft(self.f(Un)))
        fU2 = 3/4*self.expM12*fUn + 1/4*self.expMm12*(fU1 + self.tau*fft.fft(self.f(fft.ifft(fU1).real)))
        return 1/3*self.expM*fUn +\
               2/3*self.expM12*(fU2 + self.tau*fft.fft(self.f(fft.ifft(fU2).real)))

    def IFRK4(self,fUn,Un):
        fU1 = self.expM12*(fUn + self.tau/2*fft.fft(self.f(Un)))
        fU2 = self.expM12*fUn + self.tau/2*fft.fft(self.f(fft.ifft(fU1).real))
        fU3 = self.expM*fUn + self.tau*self.expM12*fft.fft(self.f(fft.ifft(fU2).real))

        return 1/3*self.expM12*(fU1 + self.tau/2*fft.fft(self.f(fft.ifft(fU1).real))) +\
               1/3*self.expM12*fU2 +\
               1/3*(fU3 + self.tau/2*fft.fft(self.f(fft.ifft(fU3).real)))
    
    def IFRK54(self,fUn,Un):
        r = 1.346586417284006
        fU0 = fUn
        fU1 = 0.387392167970373*self.expM10*fU0 + 0.612607832029627*self.expM10*(fU0 + self.tau/r*fft.fft(self.f(fft.ifft(fU0).real)))
        fU2 = 0.568702484115635*self.expM20*fU0 + 0.431297515884365*self.expM21*(fU1 + self.tau/r*fft.fft(self.f(fft.ifft(fU1).real)))
        fU3 = 0.589791736452092*self.expM30*fU0 + 0.410208263547908*self.expM32*(fU2 + self.tau/r*fft.fft(self.f(fft.ifft(fU2).real)))
        fU4 = 0.213474206786188*self.expM40*fU0 + 0.786525793213812*self.expM43*(fU3 + self.tau/r*fft.fft(self.f(fft.ifft(fU3).real))) 

        return 0.270147144537063*self.expM50*fU0 + 0.029337521506634*self.expM50*(fU0 + self.tau/r*fft.fft(self.f(fft.ifft(fU0).real))) +\
                                                   0.239419175840559*self.expM51*(fU1 + self.tau/r*fft.fft(self.f(fft.ifft(fU1).real))) +\
                                                   0.227000995504038*self.expM53*(fU3 + self.tau/r*fft.fft(self.f(fft.ifft(fU3).real))) +\
                                                   0.234095162611706*self.expM54*(fU4 + self.tau/r*fft.fft(self.f(fft.ifft(fU4).real)))

    #   EFRK
    def EFRK1(self,fUn,Un):
        return self.phiM10*(fUn + self.tau*fft.fft(self.f(Un)))
    
    def EFRK2(self,fUn,Un):
        fU1 = self.phiM10*(fUn + self.tau*fft.fft(self.f(Un)))
        return 1/2*self.phiM20*fUn +\
               1/2*self.phiM21*(fU1 + self.tau*fft.fft(self.f(fft.ifft(fU1).real)))

    def EFRK3(self,fUn,Un):
        fU1 = self.phiM10*(fUn + 2*self.tau/3*fft.fft(self.f(Un)))
        fU2 = 2/3*self.phiM20*fUn +\
              1/3*self.phiM21*(fU1 + 4*self.tau/3*fft.fft(self.f(fft.ifft(fU1).real)))
        return 59/128*self.phiM30*fUn +\
               15/128*self.phiM30*(fUn + 4/3*self.tau*fft.fft(self.f(Un))) +\
               54/128*self.phiM32*(fU2 + 4/3*self.tau*fft.fft(self.f(fft.ifft(fU2).real)))
    
    def EFRK4(self,fUn,Un):
        fU1 = self.phiM10*(fUn + self.tau/2*fft.fft(self.f(Un)))

        fU2 = 1/2*self.phiM20*(fUn - self.tau/2*fft.fft(self.f(fft.ifft(fUn).real))) +\
              1/2*self.phiM21*(fU1 + self.tau*fft.fft(self.f(fft.ifft(fU1).real)))

        fU3 = 1/9*self.phiM30*(fUn - self.tau*fft.fft(self.f(fft.ifft(fUn).real))) +\
              2/9*self.phiM31*(fU1 - 3*self.tau/2*fft.fft(self.f(fft.ifft(fU1).real))) +\
              2/3*self.phiM32*(fU2 + 3*self.tau/2*fft.fft(self.f(fft.ifft(fU2).real)))

        return 1/3*self.phiM41*(fU1 + self.tau/2*fft.fft(self.f(fft.ifft(fU1).real))) +\
               1/3*self.phiM42* fU2 +\
               1/3*self.phiM43*(fU3 + self.tau/2*fft.fft(self.f(fft.ifft(fU3).real)))

    def EFRK54(self,fUn,Un):
        r = 1.346586417284006
        fU0 = fUn
        fU1 = 0.387392167970373*self.phiM10*fU0 + 0.612607832029627*self.phiM10*(fU0 + self.tau/r*fft.fft(self.f(fft.ifft(fU0).real)))
        fU2 = 0.568702484115635*self.phiM20*fU0 + 0.431297515884365*self.phiM21*(fU1 + self.tau/r*fft.fft(self.f(fft.ifft(fU1).real)))
        fU3 = 0.589791736452092*self.phiM30*fU0 + 0.410208263547908*self.phiM32*(fU2 + self.tau/r*fft.fft(self.f(fft.ifft(fU2).real)))
        fU4 = 0.213474206786188*self.phiM40*fU0 + 0.786525793213812*self.phiM43*(fU3 + self.tau/r*fft.fft(self.f(fft.ifft(fU3).real))) 

        return 0.270147144537063*self.phiM50*fU0 + 0.029337521506634*self.phiM50*(fU0 + self.tau/r*fft.fft(self.f(fft.ifft(fU0).real))) +\
                                                   0.239419175840559*self.phiM51*(fU1 + self.tau/r*fft.fft(self.f(fft.ifft(fU1).real))) +\
                                                   0.227000995504038*self.phiM53*(fU3 + self.tau/r*fft.fft(self.f(fft.ifft(fU3).real))) +\
                                                   0.234095162611706*self.phiM54*(fU4 + self.tau/r*fft.fft(self.f(fft.ifft(fU4).real)))
    
    def energy(self,Un):
        energy = np.nan
        if self.bc == "periodic":
           energy = self.h*(-1/2*Un@self.L@Un + self.F(Un)@np.ones(self.N))
        elif self.bc == "neumann":
            pass
        return energy

    def solve(self):
        pbar = tqdm.tqdm(total=self.T+self.tau,
                         bar_format="{desc}: {percentage:.2f}% |{bar}| {n:.2f}/{total:.2f}[{elapsed}<{remaining}] {postfix}",
                         mininterval=0.1)

        cur_index = 0
        while cur_index < self.tn.shape[0]-1:
            fu = self.step(self.fU[cur_index],self.U[cur_index])

            cur_index += 1

            self.fU[cur_index] = fu
            self.U[cur_index] = fft.ifft(fu).real

            self.Energys[cur_index] = self.energy(self.U[cur_index])
            
            pbar.set_postfix({"time_step":self.tau,
                              "energy":self.Energys[cur_index],
                              "maximum_val":np.max(np.abs(self.U[cur_index]))})
            pbar.update(self.tau)
        pbar.close()

class Allen_Cahn_Solver_2D():
    def __init__(
        self,
        eps:float,
        t_period:float,
        s_domain:Tuple[float,float,float,float],
        time_step:float,
        discrete_num: Tuple[int,int],
        initial_condition: tnp.NDArray,
        boundary_condition: str,
        potential: str,
        step_method: str) -> None:

        self.eps = eps

        self.T = t_period
        self.tau = time_step
        self.tn = np.array([0])

        self.xa,self.ya,self.xb,self.yb = s_domain
        self.N,self.M = discrete_num
        self.hx = (self.xb-self.xa)/self.N
        self.hy = (self.yb-self.ya)/self.M

        self.xh = np.linspace(self.xa,self.xb,self.N+1)
        self.yh = np.linspace(self.ya,self.yb,self.M+1)
        self.Xh,self.Yh = np.meshgrid(self.xh,self.yh)

        u0 = initial_condition

        self.bc = boundary_condition
        if self.bc == "periodic":
            self.L = self.eps**2*self.laplace_periodic()
            self.D,self.Q = linalg.eigh(self.L)
            self.Q_inv = self.Q.T
            
            self.Gx,self.Gy = self.gradient_periodic()

            self.U = u0[:-1,:-1].flatten()[np.newaxis,:]

        elif self.bc == "neumann":
            self.L = self.eps**2*self.laplace_neumann()
            self.D,self.Q = np.linalg.eig(self.L)
            self.Q_inv = np.linalg.inv(self.Q)

            self.Gx,self.Gy = self.gradient_neumann()

            self.U = u0.flatten()[np.newaxis,:]

        if potential == "flory_huggins":
            self.F,self.f = self.fh_potential()
        elif potential == "ginzburg_landau":
            self.F,self.f = self.gl_potential()

        self.Energys = np.array(self.energy(self.U[0]))
        
        if step_method == "IFRK1":
            self.step = self.IFRK1
            # self.expM = slinalg.expm(self.tau*self.L)
            self.expM = self.Q@np.diag(np.exp(self.tau*self.D))@self.Q_inv

        elif step_method == "IFRK2":
            self.step = self.IFRK2
            # self.expM = slinalg.expm(self.tau*self.L)
            self.expM = self.Q@np.diag(np.exp(self.tau*self.D))@self.Q_inv

        elif step_method == "IFRK3":
            self.step = self.IFRK3
            # self.expM = slinalg.expm(self.tau*self.L)
            # self.expM13 = slinalg.expm(1/3*self.tau*self.L)
            # self.expM23 = slinalg.expm(2/3*self.tau*self.L)
            self.expM = self.Q@np.diag(np.exp(self.tau*self.D))@self.Q_inv
            self.expM13 = self.Q@np.diag(np.exp(1/3*self.tau*self.D))@self.Q_inv
            self.expM23 = self.Q@np.diag(np.exp(2/3*self.tau*self.D))@self.Q_inv

        elif step_method == "IFRK4":
            self.step = self.IFRK4
            # self.expM = slinalg.expm(self.tau*self.L)
            # self.expM12 = slinalg.expm(1/2*self.tau*self.L)
            self.expM = self.Q@np.diag(np.exp(self.tau*self.D))@self.Q_inv
            self.expM12 = self.Q@np.diag(np.exp(1/2*self.tau*self.D))@self.Q_inv

    def laplace_periodic(self):
        Lx = np.diag(-2*np.ones(self.N)) +\
             np.diag(np.ones(self.N-1),1) +\
             np.diag(np.ones(self.N-1),-1)
        Lx[0,-1] = Lx[-1,0] = 1; Lx /= self.hx**2
        Ix = np.eye(self.N)
        
        Ly = np.diag(-2*np.ones(self.M)) +\
             np.diag(np.ones(self.M-1),1) +\
             np.diag(np.ones(self.M-1),-1)
        Ly[0,-1] = Ly[-1,0] = 1; Ly /= self.hy**2
        Iy = np.eye(self.M)

        L = sparse.kron(Iy,Lx) +\
            sparse.kron(Ly,Ix)
        return L.toarray()

    def gradient_periodic(self):
        Gx = np.diag(np.ones(self.N-1),1) - np.diag(np.ones(self.N-1),-1)
        Gx[0,-1] = -1 ;Gx[-1,0] = 1; Gx /= self.hx
        Ix = np.eye(self.N)

        Gy = np.diag(np.ones(self.M-1),1) - np.diag(np.ones(self.M-1),-1)
        Gy[0,-1] = -1 ;Gy[-1,0] = 1; Gy /= self.hy
        Iy = np.eye(self.M)

        G = sparse.kron(Iy,Gx).toarray(),\
            sparse.kron(Gy,Ix).toarray()
        return G

    def laplace_neumann(self):
        Lx = np.diag(-2*np.ones(self.N+1)) +\
             np.diag(np.ones(self.N),1) +\
             np.diag(np.ones(self.N),-1)
        Lx[0,1] += 1; Lx[-1,-2] += 1; Lx /= self.hx**2
        Ix = np.eye(self.N+1)

        Ly = np.diag(-2*np.ones(self.M+1)) +\
             np.diag(np.ones(self.M),1) +\
             np.diag(np.ones(self.M),-1)
        Ly[0,1] += 1; Ly[-1,-2] += 1; Ly /= self.hy**2
        Iy = np.eye(self.M+1)

        L = sparse.kron(Iy,Lx) +\
            sparse.kron(Ly,Ix)
        return L.toarray()

    def gradient_neumann(self):
        Gx = np.diag(np.ones(self.N),1) - np.diag(np.ones(self.N),-1)
        Gx[0,1] -= 1; Gx[-1,-2] += 1; Gx /= self.hx
        Ix = np.eye(self.N+1)

        Gy = np.diag(np.ones(self.M),1) - np.diag(np.ones(self.M),-1)
        Gy[0,1] -= 1 ;Gy[-1,-2] += 1; Gy /= self.hy
        Iy = np.eye(self.M+1)

        G = sparse.kron(Iy,Gx).toarray(),\
            sparse.kron(Gy,Ix).toarray()
        return G

    def gl_potential(self):
        F = lambda u: 1/4*(1-u**2)**2
        # f(u) = -F'(u)
        f = lambda u: (u-u**3)
        return F,f

    def fh_potential(self,theta = 0.8,theta_c = 1.6):
        F =lambda u: theta/2*((1+u)*np.log(1+u) + (1-u)*np.log(1-u)) - theta_c/2*u**2
        # f(u) = -F'(u)
        f =lambda u: theta/2*np.log((1-u)/(1+u)) + theta_c*u
        return F,f

    def IFRK1(self,Un):
        return self.expM@(Un + self.tau*self.f(Un))

    def IFRK2(self,Un):
        U1 = self.expM@(Un + self.tau*self.f(Un))
        return 1/2*self.expM@Un +\
               1/2*(U1 + self.tau*self.f(U1))

    def IFRK3(self,Un):
        U1 = self.expM23@(Un + 2*self.tau/3*self.f(Un))
        U2 = 2/3*self.expM23@Un + 1/3*(U1 + 4*self.tau/3*self.f(U1))
        return 59/128*self.expM@Un +\
               15/128*self.expM@(Un + 4/3*self.tau*self.f(Un)) +\
               54/128*self.expM13@(U2 + 4/3*self.tau*self.f(U2))

    def IFRK4(self,Un):
        U1 = self.expM12@(Un + self.tau/2*self.f(Un))
        U2 = self.expM12@Un + self.tau/2*self.f(U1)
        U3 = self.expM@Un + self.tau*self.expM12@self.f(U2)

        return 1/3*self.expM12@(U1 + self.tau/2*self.f(U1)) +\
               1/3*self.expM12@U2 +\
               1/3*(U3 + self.tau/2*self.f(U3))
    
    def energy(self,Un):
        energy = np.nan
        if self.bc == "periodic":
        #    energy = self.hx*self.hy*np.sum(self.eps**2/2*((self.Gx@Un)**2 + (self.Gy@Un)**2) + self.F(Un))
           energy = self.hx*self.hy*(-1/2*(Un[np.newaxis,:])@(self.L@(Un[:,np.newaxis])) + np.sum(self.F(Un)))
        elif self.bc == "neumann":
           energy = self.hx*self.hy*np.sum((self.eps**2/2*((self.Gx@Un)**2 + (self.Gy@Un)**2) + self.F(Un)).reshape((self.M+1,self.N+1))[:-1,:-1].flatten())
        return energy

    def solve(self):
        pbar = tqdm.tqdm(total=self.T+self.tau,
                         bar_format="{desc}: {percentage:.2f}% |{bar}| {n:.2f}/{total:.2f}[{elapsed}<{remaining}] {postfix}",
                         mininterval=0.1)

        while self.tn[-1] < self.T:
            u = self.step(self.U[-1])

            self.tn = np.append(self.tn,np.round(self.tn[-1]+self.tau,12)) 
            self.U = np.concatenate([self.U,u[np.newaxis,:]],axis=0) 
            self.Energys = np.append(self.Energys,self.energy(u))
            
            pbar.set_postfix({"time_step":self.tau,
                              "energy":self.Energys[-1],
                              "maximum_val":np.max(u)})
            pbar.update(self.tau)
        pbar.close()

class Allen_Cahn_fSolver_2D():
    def __init__(
        self,
        eps:float,
        t_period:float,
        s_domain:Tuple[float,float,float,float],
        time_step:float,
        discrete_num: Tuple[int,int],
        initial_condition: tnp.NDArray,
        boundary_condition: str,
        potential: str,
        step_method: str) -> None:

        self.eps = eps

        self.T = t_period
        self.tau = time_step
        self.tn = np.arange(0,self.T+self.tau,self.tau)

        self.xa,self.ya,self.xb,self.yb = s_domain
        self.N,self.M = discrete_num
        self.hx = (self.xb-self.xa)/self.N
        self.hy = (self.yb-self.ya)/self.M

        self.xh = np.linspace(self.xa,self.xb,self.N+1)
        self.yh = np.linspace(self.ya,self.yb,self.M+1)
        self.Xh,self.Yh = np.meshgrid(self.xh,self.yh)

        u0 = initial_condition

        self.bc = boundary_condition
        if self.bc == "periodic":
            self.Lx,self.Ly = self.laplace_periodic()
            self.Lx = self.eps**2*self.Lx
            self.Ly = self.eps**2*self.Ly

            self.Dx = np.array([-4*self.eps**2*np.sin(k*np.pi/self.N)**2/self.hx**2 for k in range(self.N)])[np.newaxis,:]
            self.Dy = np.array([-4*self.eps**2*np.sin(k*np.pi/self.M)**2/self.hy**2 for k in range(self.M)])[:,np.newaxis]
            self.D = self.Dx + self.Dy

            self.U = np.empty((self.tn.shape[0],self.M,self.N)); self.U[0] = u0[:-1,:-1]
            self.fU = np.empty((self.tn.shape[0],self.M,self.N),dtype = np.complex128); self.fU[0] = fft.fft2(u0[:-1,:-1])

        elif self.bc == "neumann":
            raise

        if potential == "flory_huggins":
            self.F,self.f = self.fh_potential()
        elif potential == "ginzburg_landau":
            self.F,self.f = self.gl_potential()

        self.Energys = np.empty_like(self.tn); self.Energys[0] = self.energy(self.U[0])
        
        if step_method == "IFRK1":
            self.step = self.IFRK1
            self.expM = np.exp(self.tau*self.D)

        elif step_method == "IFRK2":
            self.step = self.IFRK2
            self.expM = np.exp(self.tau*self.D)

        elif step_method == "IFRK3":
            self.step = self.IFRK3
            self.expM = np.exp(self.tau*self.D)
            self.expM13 = np.exp(1/3*self.tau*self.D)
            self.expM23 = np.exp(2/3*self.tau*self.D)

        elif step_method == "Shu_Osher3":
            self.step = self.Shu_Osher3
            self.expM = np.exp(self.tau*self.D)
            self.expM12 = np.exp(1/2*self.tau*self.D)
            self.expMm12 = np.exp(-1/2*self.tau*self.D)

        elif step_method == "IFRK4":
            self.step = self.IFRK4
            self.expM = np.exp(self.tau*self.D)
            self.expM12 = np.exp(1/2*self.tau*self.D)

        elif step_method == "IFRK54":
            self.step = self.IFRK54 
            c0 = 0; c1 = 0.4549; c2 = c3 = 0.5165; c4 = 0.9903; c5 = 1
            self.expM10 = np.exp((c1-c0)*self.tau*self.D) 

            self.expM20 = np.exp((c2-c0)*self.tau*self.D) 
            self.expM21 = np.exp((c2-c1)*self.tau*self.D) 

            self.expM30 = np.exp((c3-c0)*self.tau*self.D)
            self.expM32 = np.exp((c3-c2)*self.tau*self.D)

            self.expM40 = np.exp((c4-c0)*self.tau*self.D)
            self.expM43 = np.exp((c4-c3)*self.tau*self.D)

            self.expM50 = np.exp((c5-c0)*self.tau*self.D)
            self.expM51 = np.exp((c5-c1)*self.tau*self.D)
            self.expM53 = np.exp((c5-c3)*self.tau*self.D)
            self.expM54 = np.exp((c5-c4)*self.tau*self.D)

        elif step_method == "EFRK1":
            self.step = self.EFRK1
            self.phi0 = np.ones(self.N)

            self.phi1 = self.phi0*(np.ones(self.N) - self.tau*self.D)
            self.phiM10 = self.phi0/self.phi1
        
        elif step_method == "EFRK2":
            self.step = self.EFRK2
            self.phi0 = np.ones(self.N)

            self.phi1 = self.phi0*(np.ones(self.N) - self.tau*self.D)
            self.phi2 = 1/2*self.phi0 + 1/2*self.phi1*(np.ones(self.N) - self.tau*self.D) 

            self.phiM10 = self.phi0/self.phi1
            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2
        
        elif step_method == "EFRK3":
            self.step = self.EFRK3

            self.phi0 = np.ones(self.N)
            self.phi1 = self.phi0*(np.ones(self.N) - 2/3*self.tau*self.D)
            self.phi2 = 2/3*self.phi0 +\
                        1/3*self.phi1*(np.ones(self.N) - 4/3*self.tau*self.D)
            self.phi3 = 59/128*self.phi0 +\
                        15/128*self.phi0*(np.ones(self.N) - 4/3*self.tau*self.D) +\
                        27/64*self.phi2*(np.ones(self.N) - 4/3*self.tau*self.D)

            self.phiM10 = self.phi0/self.phi1

            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2

            self.phiM30 = self.phi0/self.phi3
            self.phiM31 = self.phi1/self.phi3
            self.phiM32 = self.phi2/self.phi3
        
        elif step_method == "EFRK4":
            self.step = self.EFRK4
            self.phi0 = np.ones(self.N)
            self.phi1 = self.phi0*(np.ones(self.N) - 1/2*self.tau*self.D)
            self.phi2 = 1/2*self.phi0*(np.ones(self.N) + 1/2*self.tau*self.D) +\
                        1/2*self.phi1*(np.ones(self.N) - self.tau*self.D)
            self.phi3 = 1/9*self.phi0*(np.ones(self.N) + self.tau*self.D) +\
                        2/9*self.phi1*(np.ones(self.N) + 3/2*self.tau*self.D) +\
                        2/3*self.phi2*(np.ones(self.N) - 3/2*self.tau*self.D)
            self.phi4 = 1/3*self.phi1*(np.ones(self.N) - 1/2*self.tau*self.D) +\
                        1/3*self.phi2*np.ones(self.N) +\
                        1/3*self.phi3*(np.ones(self.N) - 1/2*self.tau*self.D)

            self.phiM10 = self.phi0/self.phi1
            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2
            self.phiM30 = self.phi0/self.phi3
            self.phiM31 = self.phi1/self.phi3
            self.phiM32 = self.phi2/self.phi3
            self.phiM41 = self.phi1/self.phi4
            self.phiM42 = self.phi2/self.phi4
            self.phiM43 = self.phi3/self.phi4

        elif step_method == "EFRK54":
            self.step = self.EFRK54 
            c0 = 0; c1 = 0.4549; c2 = c3 = 0.5165; c4 = 0.9903; c5 = 1
            r = 1.346586417284006

            self.phi0 = np.ones(self.N)
            self.phi1 = 0.387392167970373*self.phi0 + 0.612607832029627*self.phi0*(np.ones(self.N) - self.tau/r*self.D)
            self.phi2 = 0.568702484115635*self.phi0 + 0.431297515884365*self.phi1*(np.ones(self.N) - self.tau/r*self.D)
            self.phi3 = 0.589791736452092*self.phi0 + 0.410208263547908*self.phi2*(np.ones(self.N) - self.tau/r*self.D)
            self.phi4 = 0.213474206786188*self.phi0 + 0.786525793213812*self.phi3*(np.ones(self.N) - self.tau/r*self.D) 

            self.phi5 = 0.270147144537063*self.phi0 + 0.029337521506634*self.phi0*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.239419175840559*self.phi1*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.227000995504038*self.phi3*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.234095162611706*self.phi4*(np.ones(self.N) - self.tau/r*self.D)

            self.phiM10 = self.phi0/self.phi1

            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2          

            self.phiM30 = self.phi0/self.phi3          
            self.phiM32 = self.phi2/self.phi3          

            self.phiM40 = self.phi0/self.phi4          
            self.phiM43 = self.phi3/self.phi4          

            self.phiM50 = self.phi0/self.phi5          
            self.phiM51 = self.phi1/self.phi5          
            self.phiM53 = self.phi3/self.phi5           
            self.phiM54 = self.phi4/self.phi5           


    def laplace_periodic(self):
        Lx = np.diag(-2*np.ones(self.N)) +\
             np.diag(np.ones(self.N-1),1) +\
             np.diag(np.ones(self.N-1),-1)
        Lx[0,-1] = Lx[-1,0] = 1; Lx /= self.hx**2
        
        Ly = np.diag(-2*np.ones(self.M)) +\
             np.diag(np.ones(self.M-1),1) +\
             np.diag(np.ones(self.M-1),-1)
        Ly[0,-1] = Ly[-1,0] = 1; Ly /= self.hy**2

        return Lx,Ly

    def gl_potential(self):
        F = lambda u: 1/4*(1-u**2)**2
        # f(u) = -F'(u)
        f = lambda u: (u-u**3)
        return F,f

    def fh_potential(self,theta = 0.8,theta_c = 1.6):
        F =lambda u: theta/2*((1+u)*np.log(1+u) + (1-u)*np.log(1-u)) - theta_c/2*u**2
        # f(u) = -F'(u)
        f =lambda u: theta/2*np.log((1-u)/(1+u)) + theta_c*u
        return F,f

    #   IFRK
    def IFRK1(self,fUn,Un):
        return self.expM*(fUn + self.tau*fft.fft2(self.f(Un)))

    def IFRK2(self,fUn,Un):
        fU1 = self.expM*(fUn + self.tau*fft.fft2(self.f(Un)))
        return 1/2*self.expM*fUn +\
               1/2*(fU1 + self.tau*fft.fft2(self.f(fft.ifft2(fU1).real)))

    def IFRK3(self,fUn,Un):
        fU1 = self.expM23*(fUn + 2*self.tau/3*fft.fft2(self.f(Un)))
        fU2 = 2/3*self.expM23*fUn + 1/3*(fU1 + 4*self.tau/3*fft.fft2(self.f(fft.ifft2(fU1).real)))
        return 59/128*self.expM*fUn +\
               15/128*self.expM*(fUn + 4/3*self.tau*fft.fft2(self.f(Un))) +\
               54/128*self.expM13*(fU2 + 4/3*self.tau*fft.fft2(self.f(fft.ifft2(fU2).real)))
            
    def Shu_Osher3(self,fUn,Un):
        fU1 = self.expM*(fUn + self.tau*fft.fft2(self.f(Un)))
        fU2 = 3/4*self.expM12*fUn + 1/4*self.expMm12*(fU1 + self.tau*fft.fft2(self.f(fft.ifft2(fU1).real)))
        return 1/3*self.expM*fUn +\
               2/3*self.expM12*(fU2 + self.tau*fft.fft(self.f(fft.ifft2(fU2).real)))

    def IFRK4(self,fUn,Un):
        fU1 = self.expM12*(fUn + self.tau/2*fft.fft2(self.f(Un)))
        fU2 = self.expM12*fUn + self.tau/2*fft.fft2(self.f(fft.ifft2(fU1).real))
        fU3 = self.expM*fUn + self.tau*self.expM12*fft.fft2(self.f(fft.ifft2(fU2).real))

        return 1/3*self.expM12*(fU1 + self.tau/2*fft.fft2(self.f(fft.ifft2(fU1).real))) +\
               1/3*self.expM12*fU2 +\
               1/3*(fU3 + self.tau/2*fft.fft2(self.f(fft.ifft2(fU3).real)))
    
    def IFRK54(self,fUn,Un):
        r = 1.346586417284006
        fU0 = fUn
        fU1 = 0.387392167970373*self.expM10*fU0 + 0.612607832029627*self.expM10*(fU0 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU0).real)))
        fU2 = 0.568702484115635*self.expM20*fU0 + 0.431297515884365*self.expM21*(fU1 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU1).real)))
        fU3 = 0.589791736452092*self.expM30*fU0 + 0.410208263547908*self.expM32*(fU2 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU2).real)))
        fU4 = 0.213474206786188*self.expM40*fU0 + 0.786525793213812*self.expM43*(fU3 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU3).real))) 

        return 0.270147144537063*self.expM50*fU0 + 0.029337521506634*self.expM50*(fU0 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU0).real))) +\
                                                   0.239419175840559*self.expM51*(fU1 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU1).real))) +\
                                                   0.227000995504038*self.expM53*(fU3 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU3).real))) +\
                                                   0.234095162611706*self.expM54*(fU4 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU4).real)))
    
    #   EFRK
    def EFRK1(self,fUn,Un):
        return self.phiM10*(fUn + self.tau*fft.fft2(self.f(Un)))
    
    def EFRK2(self,fUn,Un):
        fU1 = self.phiM10*(fUn + self.tau*fft.fft2(self.f(Un)))
        return 1/2*self.phiM20*fUn +\
               1/2*self.phiM21*(fU1 + self.tau*fft.fft2(self.f(fft.ifft2(fU1).real)))

    def EFRK3(self,fUn,Un):
        fU1 = self.phiM10*(fUn + 2*self.tau/3*fft.fft2(self.f(Un)))
        fU2 = 2/3*self.phiM20*fUn +\
              1/3*self.phiM21*(fU1 + 4*self.tau/3*fft.fft2(self.f(fft.ifft2(fU1).real)))
        return 59/128*self.phiM30*fUn +\
               15/128*self.phiM30*(fUn + 4/3*self.tau*fft.fft2(self.f(Un))) +\
               54/128*self.phiM32*(fU2 + 4/3*self.tau*fft.fft2(self.f(fft.ifft2(fU2).real)))
    
    def EFRK4(self,fUn,Un):
        fU1 = self.phiM10*(fUn + self.tau/2*fft.fft2(self.f(Un)))
        fU2 = 1/2*self.phiM20*(fUn - self.tau/2*fft.fft2(self.f(fft.ifft2(fUn).real))) +\
              1/2*self.phiM21*(fU1 + self.tau*fft.fft2(self.f(fft.ifft2(fU1).real)))
        fU3 = 1/9*self.phiM30*(fUn - self.tau*fft.fft2(self.f(fft.ifft2(fUn).real))) +\
              2/9*self.phiM31*(fU1 - 3*self.tau/2*fft.fft2(self.f(fft.ifft2(fU1).real))) +\
              2/3*self.phiM32*(fU2 + 3*self.tau/2*fft.fft2(self.f(fft.ifft2(fU2).real)))

        return 1/3*self.phiM41*(fU1 + self.tau/2*fft.fft2(self.f(fft.ifft2(fU1).real))) +\
               1/3*self.phiM42*fU2 +\
               1/3*self.phiM43*(fU3 + self.tau/2*fft.fft2(self.f(fft.ifft2(fU3).real)))

    def EFRK54(self,fUn,Un):
        r = 1.346586417284006
        fU0 = fUn
        fU1 = 0.387392167970373*self.phiM10*fU0 + 0.612607832029627*self.phiM10*(fU0 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU0).real)))
        fU2 = 0.568702484115635*self.phiM20*fU0 + 0.431297515884365*self.phiM21*(fU1 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU1).real)))
        fU3 = 0.589791736452092*self.phiM30*fU0 + 0.410208263547908*self.phiM32*(fU2 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU2).real)))
        fU4 = 0.213474206786188*self.phiM40*fU0 + 0.786525793213812*self.phiM43*(fU3 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU3).real))) 

        return 0.270147144537063*self.phiM50*fU0 + 0.029337521506634*self.phiM50*(fU0 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU0).real))) +\
                                                   0.239419175840559*self.phiM51*(fU1 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU1).real))) +\
                                                   0.227000995504038*self.phiM53*(fU3 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU3).real))) +\
                                                   0.234095162611706*self.phiM54*(fU4 + self.tau/r*fft.fft2(self.f(fft.ifft2(fU4).real)))
    def energy(self,Un):
        energy = np.nan
        if self.bc == "periodic":
           energy = self.hx*self.hy*(-1/2*Un.flatten()@(self.Ly@Un + Un@self.Lx).flatten() + self.F(Un.flatten())@np.ones(self.N*self.M))
        elif self.bc == "neumann":
           pass
        return energy

    def solve(self):
        pbar = tqdm.tqdm(total=self.T+self.tau,
                         bar_format="{desc}: {percentage:.2f}% |{bar}| {n:.2f}/{total:.2f}[{elapsed}<{remaining}] {postfix}",
                         mininterval=0.1)

        cur_index = 0
        while cur_index < self.tn.shape[0]-1:
            fu = self.step(self.fU[cur_index],self.U[cur_index])

            cur_index += 1

            self.fU[cur_index] = fu
            self.U[cur_index] = fft.ifft2(fu).real

            self.Energys[cur_index] = self.energy(self.U[cur_index])
            
            pbar.set_postfix({"time_step":self.tau,
                              "energy":self.Energys[cur_index],
                              "maximum_val":np.max(np.abs(self.U[cur_index]))})
            pbar.update(self.tau)
        pbar.close()

class Allen_Cahn_Solver_3D():
    def __init__(
        self,
        eps:float,
        t_period:float,
        s_domain:Tuple[float,float,float,float,float,float],
        time_step:float,
        discrete_num: Tuple[int,int,int],
        initial_condition: tnp.NDArray,
        boundary_condition: str,
        potential: str,
        step_method: str) -> None:

        self.eps = eps

        self.T = t_period
        self.tau = time_step
        self.tn = np.array([0])

        self.xa,self.ya,self.za,self.xb,self.yb,self.zb = s_domain
        self.N,self.M,self.K = discrete_num
        self.hx = (self.xb-self.xa)/self.N
        self.hy = (self.yb-self.ya)/self.M
        self.hz = (self.zb-self.za)/self.K

        self.xh = np.linspace(self.xa,self.xb,self.N+1)
        self.yh = np.linspace(self.ya,self.yb,self.M+1)
        self.zh = np.linspace(self.za,self.zb,self.K+1)
        self.Xh,self.Yh,self.Zh = np.meshgrid(self.xh,self.yh,self.zh)

        u0 = initial_condition
        self.bc = boundary_condition

        if self.bc == "periodic":
            self.L = self.eps**2*self.laplace_periodic()
            self.D,self.Q = linalg.eigh(self.L)
            self.Q_inv = self.Q.T

            self.Gx,self.Gy,self.Gz = self.gradient_periodic()
            self.U = u0[:-1,:-1,:-1].flatten()[np.newaxis,:]

        elif self.bc == "neumann":
            self.L = self.eps**2*self.laplace_neumann()
            self.D,self.Q = np.linalg.eig(self.L)
            self.Q_inv = np.linalg.inv(self.Q)

            self.Gx,self.Gy,self.Gz = self.gradient_neumann()
            self.U = u0.flatten()[np.newaxis,:]

        if potential == "flory_huggins":
            self.F,self.f = self.fh_potential()
        elif potential == "ginzburg_landau":
            self.F,self.f = self.gl_potential()

        self.Energys = np.array(self.energy(self.U[0]))
        
        if step_method == "IFRK1":
            self.step = self.IFRK1
            self.expM = self.Q@np.diag(np.exp(self.tau*self.D))@self.Q_inv

        elif step_method == "IFRK2":
            self.step = self.IFRK2
            self.expM = self.Q@np.diag(np.exp(self.tau*self.D))@self.Q_inv

        elif step_method == "IFRK3":
            self.step = self.IFRK3
            self.expM = self.Q@np.diag(np.exp(self.tau*self.D))@self.Q_inv
            self.expM13 = self.Q@np.diag(np.exp(1/3*self.tau*self.D))@self.Q_inv
            self.expM23 = self.Q@np.diag(np.exp(2/3*self.tau*self.D))@self.Q_inv

        elif step_method == "IFRK4":
            self.step = self.IFRK4
            self.expM = self.Q@np.diag(np.exp(self.tau*self.D))@self.Q_inv
            self.expM12 = self.Q@np.diag(np.exp(1/2*self.tau*self.D))@self.Q_inv

    def laplace_periodic(self):
        Lx = np.diag(-2*np.ones(self.N)) +\
             np.diag(np.ones(self.N-1),1) +\
             np.diag(np.ones(self.N-1),-1)
        Lx[0,-1] = Lx[-1,0] = 1; Lx /= self.hx**2
        Ix = np.eye(self.N)
        
        Ly = np.diag(-2*np.ones(self.M)) +\
             np.diag(np.ones(self.M-1),1) +\
             np.diag(np.ones(self.M-1),-1)
        Ly[0,-1] = Ly[-1,0] = 1; Ly /= self.hy**2
        Iy = np.eye(self.M)

        Lz = np.diag(-2*np.ones(self.K)) +\
             np.diag(np.ones(self.K-1),1) +\
             np.diag(np.ones(self.K-1),-1)
        Lz[0,-1] = Lz[-1,0] = 1; Lz /= self.hz**2
        Iz = np.eye(self.K)

        L = np.kron(Iz,np.kron(Iy,Lx)) +\
            np.kron(Iz,np.kron(Ly,Ix)) +\
            np.kron(Lz,np.kron(Iy,Ix))
        return L

    def gradient_periodic(self):
        Gx = np.diag(np.ones(self.N-1),1) - np.diag(np.ones(self.N-1),-1)
        Gx[0,-1] = -1 ; Gx[-1,0] = 1; Gx /= self.hx
        Ix = np.eye(self.N)

        Gy = np.diag(np.ones(self.M-1),1) - np.diag(np.ones(self.M-1),-1)
        Gy[0,-1] = -1 ; Gy[-1,0] = 1; Gy /= self.hy
        Iy = np.eye(self.M)

        Gz = np.diag(np.ones(self.K-1),1) - np.diag(np.ones(self.K-1),-1)
        Gz[0,-1] = -1 ; Gz[-1,0] = 1; Gz /= self.hz
        Iz = np.eye(self.K)

        G = np.kron(Iz,np.kron(Iy,Gx)),\
            np.kron(Iz,np.kron(Gy,Ix)),\
            np.kron(Gz,np.kron(Iy,Ix))
        return G

    def laplace_neumann(self):
        Lx = np.diag(-2*np.ones(self.N+1)) +\
             np.diag(np.ones(self.N),1) +\
             np.diag(np.ones(self.N),-1)
        Lx[0,1] += 1; Lx[-1,-2] += 1; Lx /= self.hx**2
        Ix = np.eye(self.N+1)

        Ly = np.diag(-2*np.ones(self.M+1)) +\
             np.diag(np.ones(self.M),1) +\
             np.diag(np.ones(self.M),-1)
        Ly[0,1] += 1; Ly[-1,-2] += 1; Ly /= self.hy**2
        Iy = np.eye(self.M+1)

        Lz = np.diag(-2*np.ones(self.K+1)) +\
             np.diag(np.ones(self.K),1) +\
             np.diag(np.ones(self.K),-1)
        Lz[0,1] += 1; Lz[-1,-2] += 1; Lz /= self.hz**2
        Iz = np.eye(self.K+1)

        L = sparse.kron(Iz,np.kron(Iy,Lx)) +\
            sparse.kron(Iz,np.kron(Ly,Ix)) +\
            sparse.kron(Lz,np.kron(Iy,Ix))
        return L

    def gradient_neumann(self):
        Gx = np.diag(np.ones(self.N),1) - np.diag(np.ones(self.N),-1)
        Gx[0,1] -= 1; Gx[-1,-2] += 1; Gx /= self.hx
        Ix = np.eye(self.N+1)

        Gy = np.diag(np.ones(self.M),1) - np.diag(np.ones(self.M),-1)
        Gy[0,1] -= 1 ;Gy[-1,-2] += 1; Gy /= self.hy
        Iy = np.eye(self.M+1)

        Gz = np.diag(np.ones(self.K),1) - np.diag(np.ones(self.K),-1)
        Gz[0,1] -= 1 ;Gz[-1,-2] += 1; Gz /= self.hz
        Iz = np.eye(self.K)

        G = sparse.kron(Iz,np.kron(Iy,Gx)),\
            sparse.kron(Iz,np.kron(Gy,Ix)),\
            sparse.kron(Gz,np.kron(Iy,Ix))
        return G

    def gl_potential(self):
        F = lambda u: 1/4*(1-u**2)**2
        f = lambda u: (u-u**3)
        return F,f

    def fh_potential(self,theta = 0.8,theta_c = 1.6):
        F =lambda u: theta/2*((1+u)*np.log(1+u) + (1-u)*np.log(1-u)) - theta_c/2*u**2
        f =lambda u: theta/2*np.log((1-u)/(1+u)) + theta_c*u
        return F,f

    def IFRK1(self,Un):
        return self.expM@(Un + self.tau*self.f(Un))

    def IFRK2(self,Un):
        U1 = self.expM@(Un + self.tau*self.f(Un))
        return 1/2*self.expM@Un +\
               1/2*(U1 + self.tau*self.f(U1))

    def IFRK3(self,Un):
        U1 = self.expM23@(Un + 2*self.tau/3*self.f(Un))
        U2 = 2/3*self.expM23@Un + 1/3*(U1 + 4*self.tau/3*self.f(U1))
        return 59/128*self.expM@Un +\
               15/128*self.expM@(Un + 4/3*self.tau*self.f(Un)) +\
               54/128*self.expM13@(U2 + 4/3*self.tau*self.f(U2))

    def IFRK4(self,Un):
        U1 = self.expM12@(Un + self.tau/2*self.f(Un))
        U2 = self.expM12@Un + self.tau/2*self.f(U1)
        U3 = self.expM@Un + self.tau*self.expM12@self.f(U2)

        return 1/3*self.expM12@(U1 + self.tau/2*self.f(U1)) +\
               1/3*self.expM12@U2 +\
               1/3*(U3 + self.tau/2*self.f(U3))
    
    def energy(self,Un):
        energy = np.nan
        if self.bc == "periodic":
           energy = self.hx*self.hy*self.hz*(-1/2*(Un[np.newaxis,:])@(self.L@(Un[:,np.newaxis])) + np.sum(self.F(Un)))
        elif self.bc == "neumann":
           energy = self.hx*self.hy*np.sum((self.eps**2/2*((self.Gx@Un)**2 + (self.Gy@Un)**2 + (self.Gy@Un)**2) + self.F(Un)).reshape((self.M+1,self.N+1,self.K+1))[:-1,:-1,:-1].flatten())
        return energy

    def solve(self):
        pbar = tqdm.tqdm(total=self.T+self.tau,
                         bar_format="{desc}: {percentage:.2f}%|{bar}| {n:.2f}/{total:.2f}[{elapsed}<{remaining}] {postfix}",
                         mininterval=0.1)

        while self.tn[-1] < self.T:
            u = self.step(self.U[-1])

            self.tn = np.append(self.tn,np.round(self.tn[-1]+self.tau,12)) 
            self.U = np.concatenate([self.U,u[np.newaxis,:]],axis=0) 
            self.Energys = np.append(self.Energys,self.energy(u))
            
            pbar.set_postfix({"time_step":self.tau,
                              "energy":self.Energys[-1],
                              "maximum_val":np.max(u)})
            pbar.update(self.tau)
        pbar.close()

class Allen_Cahn_fSolver_3D():
    def __init__(
        self,
        eps:float,
        t_period:float,
        s_domain:Tuple[float,float,float,float,float,float],
        time_step:float,
        discrete_num: Tuple[int,int,int],
        initial_condition: tnp.NDArray,
        boundary_condition: str,
        potential: str,
        step_method: str) -> None:

        self.eps = eps

        self.T = t_period
        self.tau = time_step
        self.tn = np.arange(0,self.T+self.tau,self.tau)

        self.xa,self.ya,self.za,self.xb,self.yb,self.zb = s_domain
        self.N,self.M,self.K = discrete_num
        self.hx = (self.xb-self.xa)/self.N
        self.hy = (self.yb-self.ya)/self.M
        self.hz = (self.zb-self.za)/self.K

        self.xh = np.linspace(self.xa,self.xb,self.N+1)
        self.yh = np.linspace(self.ya,self.yb,self.M+1)
        self.zh = np.linspace(self.za,self.zb,self.K+1)
        self.Xh,self.Yh,self.Zh = np.meshgrid(self.xh,self.yh,self.zh)

        u0 = initial_condition
        self.bc = boundary_condition

        if self.bc == "periodic":
            self.Lx,self.Ly,self.Lz = self.laplace_periodic()
            self.Lx = self.eps**2*self.Lx
            self.Ly = self.eps**2*self.Ly
            self.Lz = self.eps**2*self.Lz

            self.Dx = np.array([-4*self.eps**2*np.sin(k*np.pi/self.N)**2/self.hx**2 for k in range(self.N)])[np.newaxis,:,np.newaxis]
            self.Dy = np.array([-4*self.eps**2*np.sin(k*np.pi/self.M)**2/self.hy**2 for k in range(self.M)])[:,np.newaxis,np.newaxis]
            self.Dz = np.array([-4*self.eps**2*np.sin(k*np.pi/self.K)**2/self.hz**2 for k in range(self.K)])[np.newaxis,np.newaxis,:]

            self.D = self.Dx + self.Dy + self.Dz

            self.U = np.empty((self.tn.shape[0],self.M,self.N,self.K)); self.U[0] = u0[:-1,:-1,:-1]
            self.fU = np.empty((self.tn.shape[0],self.M,self.N,self.K),dtype = np.complex128); self.fU[0] = fft.fftn(u0[:-1,:-1,:-1])
        elif self.bc == "neumann":
            raise

        if potential == "flory_huggins":
            self.F,self.f = self.fh_potential()
        elif potential == "ginzburg_landau":
            self.F,self.f = self.gl_potential()

        self.Energys = np.empty_like(self.tn); self.Energys[0] = self.energy(self.U[0])
        
        if step_method == "IFRK1":
            self.step = self.IFRK1
            self.expM = np.exp(self.tau*self.D)

        elif step_method == "IFRK2":
            self.step = self.IFRK2
            self.expM = np.exp(self.tau*self.D)

        elif step_method == "IFRK3":
            self.step = self.IFRK3
            self.expM = np.exp(self.tau*self.D)
            self.expM13 = np.exp(1/3*self.tau*self.D)
            self.expM23 = np.exp(2/3*self.tau*self.D)

        elif step_method == "Shu_Osher3":
            self.step = self.Shu_Osher3
            self.expM = np.exp(self.tau*self.D)
            self.expM12 = np.exp(1/2*self.tau*self.D)
            self.expMm12 = np.exp(-1/2*self.tau*self.D)

        elif step_method == "IFRK4":
            self.step = self.IFRK4
            self.expM = np.exp(self.tau*self.D)
            self.expM12 = np.exp(1/2*self.tau*self.D)

        elif step_method == "IFRK54":
            self.step = self.IFRK54 
            c0 = 0; c1 = 0.4549; c2 = c3 = 0.5165; c4 = 0.9903; c5 = 1
            self.expM10 = np.exp((c1-c0)*self.tau*self.D) 

            self.expM20 = np.exp((c2-c0)*self.tau*self.D) 
            self.expM21 = np.exp((c2-c1)*self.tau*self.D) 

            self.expM30 = np.exp((c3-c0)*self.tau*self.D)
            self.expM32 = np.exp((c3-c2)*self.tau*self.D)

            self.expM40 = np.exp((c4-c0)*self.tau*self.D)
            self.expM43 = np.exp((c4-c3)*self.tau*self.D)

            self.expM50 = np.exp((c5-c0)*self.tau*self.D)
            self.expM51 = np.exp((c5-c1)*self.tau*self.D)
            self.expM53 = np.exp((c5-c3)*self.tau*self.D)
            self.expM54 = np.exp((c5-c4)*self.tau*self.D)

        elif step_method == "EFRK1":
            self.step = self.EFRK1
            self.phi0 = np.ones(self.N)

            self.phi1 = self.phi0*(np.ones(self.N) - self.tau*self.D)
            self.phiM10 = self.phi0/self.phi1
        
        elif step_method == "EFRK2":
            self.step = self.EFRK2
            self.phi0 = np.ones(self.N)

            self.phi1 = self.phi0*(np.ones(self.N) - self.tau*self.D)
            self.phi2 = 1/2*self.phi0 + 1/2*self.phi1*(np.ones(self.N) - self.tau*self.D) 

            self.phiM10 = self.phi0/self.phi1
            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2
        
        elif step_method == "EFRK3":
            self.step = self.EFRK3

            self.phi0 = np.ones(self.N)
            self.phi1 = self.phi0*(np.ones(self.N) - 2/3*self.tau*self.D)
            self.phi2 = 2/3*self.phi0 +\
                        1/3*self.phi1*(np.ones(self.N) - 4/3*self.tau*self.D)
            self.phi3 = 59/128*self.phi0 +\
                        15/128*self.phi0*(np.ones(self.N) - 4/3*self.tau*self.D) +\
                        27/64*self.phi2*(np.ones(self.N) - 4/3*self.tau*self.D)

            self.phiM10 = self.phi0/self.phi1

            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2

            self.phiM30 = self.phi0/self.phi3
            self.phiM31 = self.phi1/self.phi3
            self.phiM32 = self.phi2/self.phi3
        
        elif step_method == "EFRK4":
            self.step = self.EFRK4
            self.phi0 = np.ones(self.N)
            self.phi1 = self.phi0*(np.ones(self.N) - 1/2*self.tau*self.D)
            self.phi2 = 1/2*self.phi0*(np.ones(self.N) + 1/2*self.tau*self.D) +\
                        1/2*self.phi1*(np.ones(self.N) - self.tau*self.D)
            self.phi3 = 1/9*self.phi0*(np.ones(self.N) + self.tau*self.D) +\
                        2/9*self.phi1*(np.ones(self.N) + 3/2*self.tau*self.D) +\
                        2/3*self.phi2*(np.ones(self.N) - 3/2*self.tau*self.D)
            self.phi4 = 1/3*self.phi1*(np.ones(self.N) - 1/2*self.tau*self.D) +\
                        1/3*self.phi2*np.ones(self.N) +\
                        1/3*self.phi3*(np.ones(self.N) - 1/2*self.tau*self.D)

            self.phiM10 = self.phi0/self.phi1
            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2
            self.phiM30 = self.phi0/self.phi3
            self.phiM31 = self.phi1/self.phi3
            self.phiM32 = self.phi2/self.phi3
            self.phiM41 = self.phi1/self.phi4
            self.phiM42 = self.phi2/self.phi4
            self.phiM43 = self.phi3/self.phi4

        elif step_method == "EFRK54":
            self.step = self.EFRK54 
            c0 = 0; c1 = 0.4549; c2 = c3 = 0.5165; c4 = 0.9903; c5 = 1
            r = 1.346586417284006

            self.phi0 = np.ones(self.N)
            self.phi1 = 0.387392167970373*self.phi0 + 0.612607832029627*self.phi0*(np.ones(self.N) - self.tau/r*self.D)
            self.phi2 = 0.568702484115635*self.phi0 + 0.431297515884365*self.phi1*(np.ones(self.N) - self.tau/r*self.D)
            self.phi3 = 0.589791736452092*self.phi0 + 0.410208263547908*self.phi2*(np.ones(self.N) - self.tau/r*self.D)
            self.phi4 = 0.213474206786188*self.phi0 + 0.786525793213812*self.phi3*(np.ones(self.N) - self.tau/r*self.D) 

            self.phi5 = 0.270147144537063*self.phi0 + 0.029337521506634*self.phi0*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.239419175840559*self.phi1*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.227000995504038*self.phi3*(np.ones(self.N) - self.tau/r*self.D) +\
                                                      0.234095162611706*self.phi4*(np.ones(self.N) - self.tau/r*self.D)

            self.phiM10 = self.phi0/self.phi1

            self.phiM20 = self.phi0/self.phi2
            self.phiM21 = self.phi1/self.phi2          

            self.phiM30 = self.phi0/self.phi3          
            self.phiM32 = self.phi2/self.phi3          

            self.phiM40 = self.phi0/self.phi4          
            self.phiM43 = self.phi3/self.phi4          

            self.phiM50 = self.phi0/self.phi5          
            self.phiM51 = self.phi1/self.phi5          
            self.phiM53 = self.phi3/self.phi5           
            self.phiM54 = self.phi4/self.phi5           


    def laplace_periodic(self):
        Lx = np.diag(-2*np.ones(self.N)) +\
             np.diag(np.ones(self.N-1),1) +\
             np.diag(np.ones(self.N-1),-1)
        Lx[0,-1] = Lx[-1,0] = 1; Lx /= self.hx**2
        
        Ly = np.diag(-2*np.ones(self.M)) +\
             np.diag(np.ones(self.M-1),1) +\
             np.diag(np.ones(self.M-1),-1)
        Ly[0,-1] = Ly[-1,0] = 1; Ly /= self.hy**2

        Lz = np.diag(-2*np.ones(self.K)) +\
             np.diag(np.ones(self.K-1),1) +\
             np.diag(np.ones(self.K-1),-1)
        Lz[0,-1] = Lz[-1,0] = 1; Lz /= self.hz**2

        return Lx,Ly,Lz

    def gl_potential(self):
        F = lambda u: 1/4*(1-u**2)**2
        f = lambda u: (u-u**3)
        return F,f

    def fh_potential(self,theta = 0.8,theta_c = 1.6):
        F =lambda u: theta/2*((1+u)*np.log(1+u) + (1-u)*np.log(1-u)) - theta_c/2*u**2
        f =lambda u: theta/2*np.log((1-u)/(1+u)) + theta_c*u
        return F,f

    #   IFRK
    def IFRK1(self,fUn,Un):
        return self.expM*(fUn + self.tau*fft.fftn(self.f(Un)))

    def IFRK2(self,fUn,Un):
        fU1 = self.expM*(fUn + self.tau*fft.fftn(self.f(Un)))
        return 1/2*self.expM*fUn +\
               1/2*(fU1 + self.tau*fft.fftn(self.f(fft.ifftn(fU1).real)))

    def IFRK3(self,fUn,Un):
        fU1 = self.expM23*(fUn + 2*self.tau/3*fft.fftn(self.f(Un)))
        fU2 = 2/3*self.expM23*fUn + 1/3*(fU1 + 4*self.tau/3*fft.fftn(self.f(fft.ifftn(fU1).real)))
        return 59/128*self.expM*fUn +\
               15/128*self.expM*(fUn + 4/3*self.tau*fft.fftn(self.f(Un))) +\
               54/128*self.expM13*(fU2 + 4/3*self.tau*fft.fftn(self.f(fft.ifftn(fU2).real)))
            
    def Shu_Osher3(self,fUn,Un):
        fU1 = self.expM*(fUn + self.tau*fft.fftn(self.f(Un)))
        fU2 = 3/4*self.expM12*fUn + 1/4*self.expMm12*(fU1 + self.tau*fft.fftn(self.f(fft.ifftn(fU1).real)))
        return 1/3*self.expM*fUn +\
               2/3*self.expM12*(fU2 + self.tau*fft.fft(self.f(fft.ifftn(fU2).real)))

    def IFRK4(self,fUn,Un):
        fU1 = self.expM12*(fUn + self.tau/2*fft.fftn(self.f(Un)))
        fU2 = self.expM12*fUn + self.tau/2*fft.fftn(self.f(fft.ifftn(fU1).real))
        fU3 = self.expM*fUn + self.tau*self.expM12*fft.fftn(self.f(fft.ifftn(fU2).real))

        return 1/3*self.expM12*(fU1 + self.tau/2*fft.fftn(self.f(fft.ifftn(fU1).real))) +\
               1/3*self.expM12*fU2 +\
               1/3*(fU3 + self.tau/2*fft.fftn(self.f(fft.ifftn(fU3).real)))
    
    def IFRK54(self,fUn,Un):
        r = 1.346586417284006
        fU0 = fUn
        fU1 = 0.387392167970373*self.expM10*fU0 + 0.612607832029627*self.expM10*(fU0 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU0).real)))
        fU2 = 0.568702484115635*self.expM20*fU0 + 0.431297515884365*self.expM21*(fU1 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU1).real)))
        fU3 = 0.589791736452092*self.expM30*fU0 + 0.410208263547908*self.expM32*(fU2 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU2).real)))
        fU4 = 0.213474206786188*self.expM40*fU0 + 0.786525793213812*self.expM43*(fU3 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU3).real))) 

        return 0.270147144537063*self.expM50*fU0 + 0.029337521506634*self.expM50*(fU0 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU0).real))) +\
                                                   0.239419175840559*self.expM51*(fU1 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU1).real))) +\
                                                   0.227000995504038*self.expM53*(fU3 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU3).real))) +\
                                                   0.234095162611706*self.expM54*(fU4 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU4).real)))
    
    #   EFRK
    def EFRK1(self,fUn,Un):
        return self.phiM10*(fUn + self.tau*fft.fftn(self.f(Un)))
    
    def EFRK2(self,fUn,Un):
        fU1 = self.phiM10*(fUn + self.tau*fft.fftn(self.f(Un)))
        return 1/2*self.phiM20*fUn +\
               1/2*self.phiM21*(fU1 + self.tau*fft.fftn(self.f(fft.ifftn(fU1).real)))

    def EFRK3(self,fUn,Un):
        fU1 = self.phiM10*(fUn + 2*self.tau/3*fft.fftn(self.f(Un)))
        fU2 = 2/3*self.phiM20*fUn +\
              1/3*self.phiM21*(fU1 + 4*self.tau/3*fft.fftn(self.f(fft.ifftn(fU1).real)))
        return 59/128*self.phiM30*fUn +\
               15/128*self.phiM30*(fUn + 4/3*self.tau*fft.fftn(self.f(Un))) +\
               54/128*self.phiM32*(fU2 + 4/3*self.tau*fft.fftn(self.f(fft.ifftn(fU2).real)))
    
    def EFRK4(self,fUn,Un):
        fU1 = self.phiM10*(fUn + self.tau/2*fft.fftn(self.f(Un)))
        fU2 = 1/2*self.phiM20*(fUn - self.tau/2*fft.fftn(self.f(fft.ifftn(fUn).real))) +\
              1/2*self.phiM21*(fU1 + self.tau*fft.fftn(self.f(fft.ifftn(fU1).real)))
        fU3 = 1/9*self.phiM30*(fUn - self.tau*fft.fftn(self.f(fft.ifftn(fUn).real))) +\
              2/9*self.phiM31*(fU1 - 3*self.tau/2*fft.fftn(self.f(fft.ifftn(fU1).real))) +\
              2/3*self.phiM32*(fU2 + 3*self.tau/2*fft.fftn(self.f(fft.ifftn(fU2).real)))

        return 1/3*self.phiM41*(fU1 + self.tau/2*fft.fftn(self.f(fft.ifftn(fU1).real))) +\
               1/3*self.phiM42*fU2 +\
               1/3*self.phiM43*(fU3 + self.tau/2*fft.fftn(self.f(fft.ifftn(fU3).real)))

    def EFRK54(self,fUn,Un):
        r = 1.346586417284006
        fU0 = fUn
        fU1 = 0.387392167970373*self.phiM10*fU0 + 0.612607832029627*self.phiM10*(fU0 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU0).real)))
        fU2 = 0.568702484115635*self.phiM20*fU0 + 0.431297515884365*self.phiM21*(fU1 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU1).real)))
        fU3 = 0.589791736452092*self.phiM30*fU0 + 0.410208263547908*self.phiM32*(fU2 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU2).real)))
        fU4 = 0.213474206786188*self.phiM40*fU0 + 0.786525793213812*self.phiM43*(fU3 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU3).real))) 

        return 0.270147144537063*self.phiM50*fU0 + 0.029337521506634*self.phiM50*(fU0 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU0).real))) +\
                                                   0.239419175840559*self.phiM51*(fU1 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU1).real))) +\
                                                   0.227000995504038*self.phiM53*(fU3 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU3).real))) +\
                                                   0.234095162611706*self.phiM54*(fU4 + self.tau/r*fft.fftn(self.f(fft.ifftn(fU4).real)))
    
    def energy(self,Un):
        energy = np.nan
        if self.bc == "periodic":
           energy = self.hx*self.hy*self.hz*(-1/2*Un.flatten()@(fft.ifftn(fft.fftn(Un)*self.D).real).flatten() + self.F(Un.flatten())@np.ones(self.N*self.M*self.K))
        elif self.bc == "neumann":
            pass
        return energy

    def solve(self):
        pbar = tqdm.tqdm(total=self.T+self.tau,
                         bar_format="{desc}: {percentage:.2f}%|{bar}| {n:.2f}/{total:.2f}[{elapsed}<{remaining}] {postfix}",
                         mininterval=0.1)

        cur_index = 0
        while cur_index < self.tn.shape[0]-1:
            fu = self.step(self.fU[cur_index],self.U[cur_index])

            cur_index += 1

            self.fU[cur_index] = fu
            self.U[cur_index] = fft.ifftn(fu).real
            self.Energys[cur_index] = self.energy(self.U[cur_index])
            
            pbar.set_postfix({"time_step":self.tau,
                              "energy":self.Energys[cur_index],
                              "maximum_val":np.max(self.U[cur_index])})
            pbar.update(self.tau)
        pbar.close()


if __name__ == "__main__":
    np.set_printoptions(linewidth=100,precision=4)
    # eps = 0.01
    # t_period = 1
    # s_domain = (0,1)
    # N = 16
    # discrete_num = N
    # xn = np.linspace(s_domain[0],s_domain[1],N+1)

    # initial_condition = np.random.uniform(-0.01,0.01,N+1)
    # boundary_condition = "periodic"
    # potential = "flory_huggins" 
    # solver = Allen_Cahn_Solver_1D(eps,t_period,s_domain,2**-6,discrete_num,initial_condition,boundary_condition,potential,"IFRK3")

    # L_v,L_m = linalg.eigh(solver.L)
    # print(L_m)
    # print(np.linalg.norm(-solver.L,np.inf))
    # print(np.linalg.norm(L_m@np.diag(-L_v)@L_m.T@initial_condition[:-1],np.inf))
    # print(np.linalg.norm(L_m@(np.max(-L_v)*np.eye(N))@L_m.T@initial_condition[:-1],np.inf))

    # print(np.linalg.norm(L_m@(1*np.eye(N))@L_m.T@initial_condition[:-1],np.inf))
    # print(np.linalg.norm(L_m@(2*np.eye(N))@L_m.T@initial_condition[:-1],np.inf))


    # print(np.real(np.fft.ifft(np.fft.fft(initial_condition[:-1])*[-4*eps**2*np.sin(k*np.pi/N)**2/solver.h**2 for k in range(N)])))

    eps = 0.01
    t_period = 1
    s_domain = (0,0,1,1)
    N = 8
    discrete_num = N,N
    xn = np.linspace(s_domain[0],s_domain[1],N+1)

    initial_condition = np.random.uniform(-0.8,0.8,(N+1,N+1))
    boundary_condition = "periodic"
    potential = "flory_huggins" 
    solver = Allen_Cahn_fSolver_2D(eps,t_period,s_domain,2**-6,discrete_num,initial_condition,boundary_condition,potential,"IFRK4")
    # print(solver.Energys)
    # solver.solve()

    # print(solver.Ly@solver.U[0] + solver.U[0]@solver.Lx)
    # print(fft.ifft2(fft.fft2(solver.U[0])*solver.D).real)

    # import timeit
    # print(timeit.timeit(lambda :solver.Ly@solver.U[0] + solver.U[0]@solver.Lx,number=1000))
    # print(timeit.timeit(lambda :np.real(np.fft.ifft2(np.fft.fft2(solver.U[0])*solver.D).real),number=1000))
    # print(timeit.timeit(lambda :np.real(fft.ifft2(fft.fft2(solver.U[0])*solver.D).real),number=1000))

    # discrete_num = N,N,N
    # initial_condition = np.random.uniform(-0.8,0.8,(N,N,N))

    # print(fft.ifftn(fft.fftn(initial_condition,axes=[0,1,2]),axes=[0,1,2]).real - initial_condition)