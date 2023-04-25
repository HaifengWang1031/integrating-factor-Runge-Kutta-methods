from typing import Tuple
import numpy as np
import numpy.typing as tnp

import scipy.linalg as linalg
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
        self.a,self.b = s_domain
        self.N = discrete_num
        self.h = (self.b-self.a)/self.N
        self.xh = np.linspace(self.a,self.b,self.N+1)
        self.tau = time_step

        u0 = initial_condition

        self.bc = boundary_condition
        if self.bc == "periodic":
            self.L = self.eps**2*self.laplace_periodic()
            self.D,self.Q = linalg.eigh(self.L)
            self.Q_inv = self.Q.T
            self.G = self.gradient_periodic()
            self.U = u0[:-1][np.newaxis,:]

        elif self.bc == "neumann":
            self.L = self.eps**2*self.laplace_neumann()
            self.D,self.Q = np.linalg.eig(self.L)
            self.Q_inv = np.linalg.inv(self.Q)
            self.G = self.gradient_neumann()
            self.U = u0[np.newaxis,:]
        
        if potential == "flory_huggins":
            self.F,self.f = self.fh_potential()
        else:
            self.F,self.f = self.gl_potential()
        
        if step_method == "IFRK1":
            self.step = self.IFRK1
        elif step_method == "IFRK2":
            self.step = self.IFRK2
        elif step_method == "IFRK3":
            self.step = self.IFRK3
        elif step_method == "IFRK4":
            self.step = self.IFRK4
        elif step_method == "Euler":
            self.step = self.Euler

        self.tn = np.array([0])
        self.Energys = np.array(self.energy(self.U[0]))

    def laplace_periodic(self):
        L = np.diag(-2*np.ones(self.N))
        L += np.diag(np.ones(self.N-1),1) + np.diag(np.ones(self.N-1),-1)
        L[0,-1] = L[-1,0] = 1; L /= self.h**2
        return L

    def gradient_periodic(self):
        G = np.diag(np.ones(self.N-1),1) - np.diag(np.ones(self.N-1),-1)
        G[0,-1] = -1 ;G[-1,0] = 1; G /= self.h
        return G

    def laplace_neumann(self):
        L = np.diag(-2*np.ones(self.N+1))
        L += np.diag(np.ones(self.N),1) + np.diag(np.ones(self.N),-1)
        L[0,1] += 1; L[-1,-2] += 1; L /= self.h**2
        return L

    def gradient_neumann(self):
        L = np.diag(np.ones(self.N),1); L -= np.diag(np.ones(self.N),-1)
        L[0,1] -= 1; L[-1,-2] += 1; L /= self.h
        return L

    def gl_potential(self):
        F = lambda u: (u**2-1)**2/4
        f = lambda u: (u - u**3)
        return F,f

    def fh_potential(self,theta = 0.8,theta_c = 1.6):
        F =lambda u: theta/2*((1+u)*np.log(1+u) + (1-u)*np.log(1-u)) - theta_c/2*u**2
        f =lambda u: theta/2*np.log((1-u)/(1+u)) + theta_c*u
        return F,f

    def Euler(self,Un,tau):
        return np.linalg.solve(np.eye(self.N)-tau*self.L,Un + tau*self.f(Un))

    def IFRK1(self,Un,tau):
        expM = self.Q@np.diag(np.exp(tau*self.D))@self.Q_inv

        return expM@(Un + tau*self.f(Un))

    def IFRK2(self,Un,tau):
        expM = self.Q@np.diag(np.exp(tau*self.D))@self.Q_inv

        U1 = expM@(Un + tau*self.f(Un))
        return 1/2*expM@Un +\
               1/2*(U1 + tau*self.f(U1))

    def IFRK3(self,Un,tau):
        expM = self.Q@np.diag(np.exp(tau*self.D))@self.Q_inv
        expM13 = self.Q@np.diag(np.exp(1/3*tau*self.D))@self.Q_inv
        expM23 = self.Q@np.diag(np.exp(2/3*tau*self.D))@self.Q_inv

        U1 = expM23@(Un + 2*tau/3*self.f(Un))
        U2 = 2/3*expM23@Un + 1/3*(U1 + 4*tau/3*self.f(U1))
        return 59/128*expM@Un +\
               15/128*expM@(Un + 4/3*tau*self.f(Un)) +\
               54/128*expM13@(U2 + 4/3*tau*self.f(U2))

    def IFRK4(self,Un,tau):
        expM = self.Q@np.diag(np.exp(tau*self.D))@self.Q_inv
        expM12 = self.Q@np.diag(np.exp(1/2*tau*self.D))@self.Q_inv

        U1 = expM12@(Un + tau/2*self.f(Un))
        U2 = expM12@Un + tau/2*self.f(U1)
        U3 = expM@Un + tau*expM12@self.f(U2)

        return 1/3*expM12@(U1 + tau/2*self.f(U1)) +\
               1/3*expM12@U2 +\
               1/3*(U3 + tau/2*self.f(U3))
    
    def energy(self,Un):
        energy = np.nan
        if self.bc == "periodic":
           energy = self.h*np.sum(self.eps**2/2*(self.G@Un)**2 + self.F(Un))
        elif self.bc == "neumann":
           energy = self.h*np.sum((self.eps**2/2*(self.G@Un)**2 + self.F(Un))[:-1])
        return energy

    def solve(self):
        pbar = tqdm.tqdm(total=self.T+1e-5,
                         bar_format="{desc}: {percentage:.2f}%|{bar}| {n:.2f}/{total:.2f}[{elapsed}<{remaining}] {postfix}",
                         mininterval=0.1)

        while self.tn[-1] < self.T:
            if self.tn[-1] + self.tau <= self.T:
                tau = self.tau
            else: 
                tau = self.T - self.tn[-1]
            u = self.step(self.U[-1],tau)

            self.tn = np.append(self.tn,np.round(self.tn[-1]+tau,5)) 
            self.U = np.concatenate([self.U,u[np.newaxis,:]],axis=0) 
            self.Energys = np.append(self.Energys,self.energy(u))
            
            pbar.set_postfix({"time_step":tau,
                              "energy":self.Energys[-1],
                              "maximum_val":np.max(u)})
            pbar.update(self.tau)
        pbar.close()


if __name__ == "__main__":
    eps = 0.001
    t_period = 10
    s_domain = (0,1)
    time_step = 1e-3
    discrete_num = 1024
    initial_condition = np.random.rand(1025)
    boundary_condition = "periodic"
    potential = "flory_huggins"
    step_method = "IFRK1"
    
    solver = Allen_Cahn_Solver_1D(eps,
                                  t_period,
                                  s_domain,
                                  time_step,
                                  discrete_num,
                                  initial_condition,
                                  boundary_condition,
                                  potential,
                                  step_method)
    
    print(solver.laplace_neumann())
    # solver.solve()
