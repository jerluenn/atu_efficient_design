import sys
import os
import shutil

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
from scipy.integrate import odeint
from scipy.optimize._lsq.least_squares import least_squares

class atu: 

    def __init__(self, elastic_mod, tether_length, 
                 tube_inner_radius, tube_outer_radius,
                 shear_mod, mass_distribution): 

        ### DO ALL THE INITIALISATION PROPERLY. 

        self._or = tube_outer_radius
        self._ir = tube_inner_radius
        self._tl = tether_length
        self._E = elastic_mod
        self._shear_mod = shear_mod
        self._mass_distribution = mass_distribution

        self._area = np.pi * self._or**2 - np.pi * self._ir**2
        self._I = ((np.pi * self._or**4) / 4) - \
            ((np.pi * self._ir**4) / 4)
        self._J = 2 * self._I

    def _initialiseStates(self):

        # Initialise all ODE states.

        ### ADD ALL THE STATES REQUIRED, AND ADD KBT, KSE. 

        self._p = SX.sym('p', 3)
        self._eta = SX.sym('self._eta', 4) 
        # self._R = SX.sym('R', 9)
        self._n = SX.sym('n', 3)
        self._m = SX.sym('m', 3)
        self._tau = SX.sym('tau', 3)
        self._alpha = SX.sym('alpha', 1)
        self._curvature = SX.sym('u', 1)
        self._E_sym = SX.sym('E', 1)
        self._mass_distribution_sym = SX.sym('md', 1)

        self._p_d = SX.sym('p_dot', 3)
        self._eta_d = SX.sym('eta_dot', 4)
        self._n_d = SX.sym('n_dot', 3)
        self._m_d = SX.sym('m_dot', 3)
        self._tau_d = SX.sym('tau_dot', 3)
        self._alpha_d = SX.sym('alpha_dot', 1)
        self._Kappa_d = SX.sym('Kappa_d_dot', 1)
        self._curvature_d = SX.sym('u_dot', 1)
        self._E_sym_d = SX.sym('E_dot', 1)
        self._mass_distribution_sym_d = SX.sym('md_dot', 1)

        # Initialise constants

        self._g = SX([9.81, 0, 0])
        self._f_ext = self._mass_distribution_sym * self._g
        self._Kappa = SX.sym('Kappa', 1)

        # Setting R 

        self._R = SX(3,3)
        self._R[0,0] = 2*(self._eta[0]**2 + self._eta[1]**2) - 1
        self._R[0,1] = 2*(self._eta[1]*self._eta[2] - self._eta[0]*self._eta[3])
        self._R[0,2] = 2*(self._eta[1]*self._eta[3] + self._eta[0]*self._eta[2])
        self._R[1,0] = 2*(self._eta[1]*self._eta[2] + self._eta[0]*self._eta[3])
        self._R[1,1] = 2*(self._eta[0]**2 + self._eta[2]**2) - 1
        self._R[1,2] = 2*(self._eta[2]*self._eta[3] - self._eta[0]*self._eta[1])
        self._R[2,0] = 2*(self._eta[1]*self._eta[3] - self._eta[0]*self._eta[2])
        self._R[2,1] = 2*(self._eta[2]*self._eta[3] + self._eta[0]*self._eta[1])
        self._R[2,2] = 2*(self._eta[0]**2 + self._eta[3]**2) - 1

        self.Kse = diag([self._shear_mod * self._area,
                        self._shear_mod * self._area, self._E_sym * self._area])
        self.Kbt = diag([self._E_sym * self._I, self._E_sym *
                        self._I, self._shear_mod * self._J])

        # Intermediate states

        self._u = inv(self._Kbt)@transpose(reshape(self._R, 3, 3))@self._m
        self._v = inv(self._Kse)@transpose(reshape(self._R, 3, 3))@self._n + SX([0, 0, 1])
        self._k = 0.1

    def _createIntegrator(self):

        model_name = 'tetherunit_stepIntegrator'

        c = self._k*(1-transpose(self._eta)@self._eta)

        p_dot = reshape(self._R, 3, 3) @ self._v
        eta_dot = vertcat(
            0.5*(-self._u[0]*self._eta[1] - self._u[1]*self._eta[2] - self._u[2]*self._eta[3]),
            0.5*(self._u[0]*self._eta[0] + self._u[2]*self._eta[2] - self._u[1]*self._eta[3]),
            0.5*(self._u[1]*self._eta[0] - self._u[2]*self._eta[1] + self._u[0]*self._eta[3]),
            0.5*(self._u[2]*self._eta[0] + self._u[1]*self._eta[1] - self._u[0]*self._eta[2])
        ) + c * self._eta 
        n_dot = - (self._f_ext) - self.get_external_distributed_forces()
        m_dot = - cross(p_dot, self._n) 
        tau_dot = SX.zeros(3)
        alpha_dot = 1
        kappa_dot = 0
        # u_dot = 0
        # u_dot = norm_2(jacobian(self._u, self._m) @ m_dot + jacobian(self._u, self._eta) @ eta_dot)
        u_dot = norm_2(inv(-self._Kbt)@((skew(self._u)@self._Kbt@self._u) + skew(self._v)@transpose(self._R)@self._n))

        x = vertcat(self._p, self._eta, self._n, self._m, self._tau, self._alpha, self._Kappa, self._curvature)
        xdot = vertcat(p_dot, eta_dot,
                       n_dot, m_dot, tau_dot, alpha_dot, kappa_dot, u_dot)
        x_dot_impl = vertcat(self._p_d, self._eta_d, self._n_d, self._m_d, self._tau_d, self._alpha_d, self._Kappa_d, self._curvature_d)

        self.model = AcadosModel()
        self.model.name = model_name
        self.model.x = x 
        self.model.f_expl_expr = xdot 
        self.model.f_impl_expr = xdot - x_dot_impl
        self.model.u = SX([])
        self.model.z = SX([])
        self.model.xdot = x_dot_impl

        sim = AcadosSim()
        sim.model = self.model 

        Sf = self._tether_length/self._integrationSteps

        # os.chdir(os.path.expanduser(
        #     self._workspace + "/atu_acados_python/scripts/"))

        # sim.code_export_directory = self._dir_name + '/' + model_name

        # for exporting data to library folder afterwards.
        # self._dir_list.append(sim.code_export_directory)
        # self._integrator_names.append(model_name)

        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 1
        acados_integrator = AcadosSimSolver(sim)

        self._stepIntegrator = acados_integrator

        return acados_integrator
    
    def createSolver(self):

        self.ocp = AcadosOcp()
        self.ocp.model = self.tetherObject.model
        nx = self.tetherObject.model.x.size()[0]
        nu = self.tetherObject.model.u.size()[0]
        ny = nx + nu

        self.ocp.dims.N = self.integration_steps
        # self.ocp.cost.cost_type_0 = 'LINEAR_LS'
        # self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.W_e = np.identity(nx)
        self.ocp.cost.W = np.zeros((ny, ny))
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx_e = np.zeros((nx, nx))
        self.ocp.cost.Vx_e[7:13, 7:13] = np.identity(6)
        self.ocp.cost.yref  = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((nx))
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.solver_options.qp_solver_iter_max = 400
        # self.ocp.solver_options.sim_method_num_steps = self.integration_steps
        self.ocp.solver_options.qp_solver_warm_start = 2

        self.ocp.solver_options.levenberg_marquardt = 10.0

        # self.ocp.solver_options.levenberg_marquardt = 1.0

        # self.ocp.solver_options.levenberg_marquardt = 1.0
        self.ocp.solver_options.regularize_method = 'CONVEXIFY'

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self.boundary_length

        wrench_lb = -5
        wrench_ub = 5

        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        self.ocp.constraints.lbx_0 = np.array([0, 0, 0, 1, 0, 0, 0, #pose at start.
            wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb,
            0, 0, 0])  # tension, alpha, kappa, curvature

        self.ocp.constraints.ubx_0 = np.array([0, 0, 0, 1, 0, 0, 0, #pose at start.
            wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub,
            0, 0, 0]) 

        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        self.ocp.constraints.lbx = np.array([-5, -5, -5, -1.05, -1.05, -1.05, -1.05, #pose at start.
            wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb, 
            0, 0, 0])

        self.ocp.constraints.ubx = np.array([5, 5, 5, 1.05, 1.05, 1.05, 1.05, #pose at start.
            wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub, 
            0, 0, 0])

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 1

        solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')
        integrator = AcadosSimSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')

        return solver, integrator