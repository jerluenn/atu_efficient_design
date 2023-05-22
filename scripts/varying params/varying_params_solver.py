from atu_varying_params import * 
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *
from matplotlib import pyplot as plt 
import pandas as pd

def main(min_, max_, num_it, num_solver_it, param_num, local_val, num_it_local): 

    tether_length = 3.1 
    E = 1.0e9 
    ir = 0.0006
    or_ = 0.002 
    shear_mod = 0.75e9 
    md = 0.035 

    atu_obj = atu(E, tether_length, ir, or_, shear_mod, md) 
    initial_solution = np.zeros((16, ))
    initial_solution[3] = 1.0 
    initial_solution[7] = tether_length*md*9.81
    initial_solution[11] = 0.1611493627
    initial_solution[14] = E
    initial_solution[15] = md
    atu_obj.initialise_WrenchBVP_solution(initial_solution, 1)

    array_ = np.linspace(min_, max_, num_it) 
    correct_sol = np.zeros((6, num_it))   
    states = np.zeros((18, num_it, num_solver_it))

    for k in range(num_it):

        for i in range(num_solver_it):

            atu_obj.getWrenchBVPsolver().solve()

            if atu_obj.getWrenchBVPsolver().get_cost() < 1e-5: 
                
                initial_solution[7:13] = atu_obj.getWrenchBVPsolver().get(0, 'x')[7:13]
                correct_sol[:, k] = initial_solution[7:13]
                solved = 1
                break

            else: 

                solved = 0 

        if solved == 0: 

            print("Could not solve!")

        if param_num == 0: 

            atu_obj.update_rod_parameters(array_[k], md, tether_length, param_num) 
            atu_obj.initialise_WrenchBVP_solution(initial_solution, 0)
            initial_solution[14] = array_[k]

        elif param_num == 1: 

            atu_obj.update_rod_parameters(E, array_[k], tether_length, param_num)  
            atu_obj.initialise_WrenchBVP_solution(initial_solution, 0)
            initial_solution[15] = array_[k]

        elif param_num == 2: 

            atu_obj.update_rod_parameters(E, md, array_[k], param_num)   
            atu_obj.initialise_WrenchBVP_solution(initial_solution, 0)

        sol_around_My = np.linspace(initial_solution[11]-local_val, initial_solution[11]+local_val, num_it_local+1)

        for n in range(num_it_local+1): 

            initial_solution[11] = sol_around_My[n]
            atu_obj.getOneStepIntegrator().set('x', initial_solution)
            atu_obj.getOneStepIntegrator().solve()            
            #change this!
            states[0:16, k, n] = atu_obj.getOneStepIntegrator().get('x')
            states[16, k, n] = np.linalg.norm(atu_obj.getOneStepIntegrator().get('x')[7:13])
            sens = atu_obj.getOneStepIntegrator().get('S_forw')
            states[17, k, n] = np.linalg.norm(sens)/np.linalg.norm(np.linalg.pinv(sens))


if __name__ == "__main__":

    main(1.0e9, 1.0000000005e9, 10, 1000, 0, 0.00005, 2) 