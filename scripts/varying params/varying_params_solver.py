from atu_varying_params import * 
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *
from matplotlib import pyplot as plt 

def main(): 

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
    atu_obj.getWrenchBVPsolver().solve()
    print(atu_obj.getWrenchBVPsolver().get_cost())
    E = 1.5e9
    initial_solution[14] = E
    atu_obj.initialise_WrenchBVP_solution(initial_solution, 1)
    atu_obj.getWrenchBVPsolver().solve()
    print(atu_obj.getWrenchBVPsolver().get_cost())


if __name__ == "__main__":

    main() 