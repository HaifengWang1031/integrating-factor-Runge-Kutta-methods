from typing import Tuple
import numpy.typing as tnp

import numpy as np
import scipy.linalg as linalg
from scipy import sparse 
import scipy.sparse.linalg as slinalg

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
            self.phiM10 = np.linalg.solve(self.phi1,self.phi0)

            self.phi2 = 2/3*self.phi0 + 1/3*self.phi1@(np.eye(self.N) - 4/3*self.tau*self.L)
            self.phiM20 = np.linalg.solve(self.phi2,self.phi0)
            self.phiM21 = np.linalg.solve(self.phi2,self.phi1)

            self.phi3 = 59/128*self.phi0 + 15/128*self.phi0@(np.eye(self.N) - 4/3*self.tau*self.L) + 27/64*self.phi2@(np.eye(self.N) - 4/3*self.tau*self.L)
            self.phiM30 = np.linalg.solve(self.phi3,self.phi0)
            self.phiM31 = np.linalg.solve(self.phi3,self.phi1)
            self.phiM32 = np.linalg.solve(self.phi3,self.phi2)

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

    def IFRK4(self,Un):
        U1 = self.expM12@(Un + self.tau/2*self.f(Un))
        U2 = self.expM12@Un + self.tau/2*self.f(U1)
        U3 = self.expM@Un + self.tau*self.expM12@self.f(U2)

        return 1/3*self.expM12@(U1 + self.tau/2*self.f(U1)) +\
               1/3*self.expM12@U2 +\
               1/3*(U3 + self.tau/2*self.f(U3))

        # EFRK
    def EFRK1(self,Un):
        return self.phiM10@(Un + self.tau*self.f(Un))
    
    def EFRK2(self,Un):
        U1 = self.phiM10@(Un + self.tau*self.f(Un))
        return 1/2*self.phiM20@Un + 1/2*self.phiM21@(U1 + self.tau*self.f(U1))

    def EFRK3(self,Un):
        U1 = self.phiM10@(Un + 2*self.tau/3*self.f(Un))
        U2 = 2/3*self.phiM20@Un + 1/3*self.phiM21@(U1 + 4*self.tau/3*self.f(U1))
        return 59/128*self.phiM30@Un +\
               15/128*self.phiM30@(Un + 4/3*self.tau*self.f(Un)) +\
               54/128*self.phiM32@(U2 + 4/3*self.tau*self.f(U2))
    
    def energy(self,Un):
        energy = np.nan
        if self.bc == "periodic":
        #    energy = self.h*np.sum(self.eps**2/2*(self.G@Un)**2 + self.F(Un))
           energy = self.h*(-1/2*(Un[np.newaxis,:])@(self.L@(Un[:,np.newaxis])) + np.sum(self.F(Un)))
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
                              "maximum_val":np.max(u)})
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
        #    energy = self.hx*self.hy*self.hz*np.sum(self.eps**2/2*((self.Gx@Un)**2 + (self.Gy@Un)**2 + (self.Gz@Un)**2) + self.F(Un))
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


if __name__ == "__main__":
    eps = 0.01
    t_period = 1
    s_domain = (0,0,1,1)
    N = 8
    discrete_num = N,N
    xn = np.linspace(s_domain[0],s_domain[1],N+1)

    initial_condition = np.random.rand(N+1,N+1)
    boundary_condition = "periodic"
    potential = "flory_huggins" 
    solver = Allen_Cahn_Solver_2D(eps,t_period,s_domain,2**-6,discrete_num,initial_condition,boundary_condition,potential,"IFRK3")
    # solver.solve()

    # print(solver.L@solver.U[-1] - np.real(np.fft.ifft(np.fft.fft(solver.L[:,0])*np.fft.fft(solver.U[-1]))))

    # import timeit
    # print(timeit.timeit(lambda :solver.L@solver.U[-1]))
    # print(timeit.timeit(lambda :np.real(np.fft.ifft(np.fft.fft(solver.L[:,0])*np.fft.fft(solver.U[-1])))))

