import matplotlib.pyplot as plt
import numpy as np
from classy import Class

delta_msq_solar = 0.002458
delta_msq_atm = 0.000074

def v_masses(m_small, normal):
    delta_m21_sq = delta_msq_atm
    delta_m31_sq = delta_msq_solar
    if normal:
        m1 = m_small
        m2_sq = delta_m21_sq + m1**2
        m3_sq = delta_m31_sq +m1**2
        m2 = np.sqrt(m2_sq)
        m3 = np.sqrt(m3_sq)
    else:
        m3 = m_small
        m1_sq = m3**2 + delta_m31_sq
        m2_sq = delta_m21_sq + m1_sq
        m1 = np.sqrt(m1_sq)
        m2 = np.sqrt(m2_sq)
        
    return m1,m2,m3

LambdaCDM_settings = {'omega_b':0.0223828,
                     #'omega_cdm':0.1201075,
                     'h':0.67810,
                     'A_s':2.100549e-09,
                     'n_s':0.9660499,
                     'tau_reio':0.05430842,
                     'output':'mPk',
                     'P_k_max_1/Mpc':3.0,
                      'Omega_m':0.309883043,
                     # The next line should be uncommented for higher precision (but significantly slower running)
                     'ncdm_fluid_approximation':3,
                     # You may uncomment this line to get more info on the ncdm sector from Class:
                     'background_verbose':1
                    }
    
k_array_default = np.logspace(-4,np.log10(LambdaCDM_settings['P_k_max_1/Mpc']),1000)
    
def dict_results(CLASS_solver):
    Dict_S = {'age': CLASS_solver.age(),
            'Neff': CLASS_solver.Neff(),
            'omega_b': CLASS_solver.omega_b(),
            'Omega0_cdm':CLASS_solver.Omega0_cdm(),
            'h':CLASS_solver.h(),
            'Omega0_k':CLASS_solver.Omega0_k(),
            'Omega0_m': CLASS_solver.Omega0_m(),
            'Omega_b': CLASS_solver.Omega_b(),
            'Omega_g': CLASS_solver.Omega_g(),
            'Omega_lambda': CLASS_solver.Omega_Lambda(),
            'Omega_m': CLASS_solver.Omega_m(),
            'Omega_r': CLASS_solver.Omega_r(),
            'rs_drag': CLASS_solver.rs_drag(),
            'Sigma8': CLASS_solver.sigma8(),
            'Sigma8_cb': CLASS_solver.sigma8_cb(),
            'T_cmb': CLASS_solver.T_cmb(),
            'theta_s_100': CLASS_solver.theta_s_100(),
            'theta_star_100': CLASS_solver.theta_star_100(),  
            'n_s':CLASS_solver.n_s(),
            'tau_reio':CLASS_solver.tau_reio()
             }
    return Dict_S    

def v_masses_std(m_small, normal, filename, print_dict=False):
    
    v_masses(m_small,normal)
    
    m1,m2,m3 = v_masses(m_small,normal)
    
    
    neutrino_mass_settings = {'N_ur':0.00441,
                              'N_ncdm':3,
                              'm_ncdm':'{},{},{}'.format(m1,m2,m3)     
                             }

    neutrino = Class()
    neutrino.set(LambdaCDM_settings)
    neutrino.set(neutrino_mass_settings)
    neutrino.compute()

    kk = np.copy(k_array_default)
    Pk_neutrino = np.zeros(len(kk))

    h = neutrino.h()
    
    for i,k in enumerate(kk):
        Pk_neutrino[i] = neutrino.pk(k*h,0.)*h**3 # function .pk(k,z)

    dict_n = dict_results(neutrino)
    if(print_dict):
        print(dict_n)
    
    
    np.savez(filename, results_dictionary = dict_n, k_n_array = kk, Pk_n_array = Pk_neutrino, normal_hierarchy = normal, m1 = m1, m2 = m2, m3 = m3, m_small = m_small, sum_nu = m1+m2+m3)
    
    
def make_LambdaCDM_Pk(filename):
    LambdaCDM = Class()
    LambdaCDM.set(LambdaCDM_settings)
    LambdaCDM.compute()
    
    
    
    kk = np.copy(k_array_default)
    Pk_LambdaCDM = np.zeros(len(kk)) 
    h = LambdaCDM.h()
    
    for i,k in enumerate(kk):
        Pk_LambdaCDM[i] = LambdaCDM.pk(k*h,0.)*h**3 # 
        
    np.savez(filename, results_dictionary = dict_results(LambdaCDM), k_array = kk, Pk_LambdaCDM = Pk_LambdaCDM)
    
    
    
