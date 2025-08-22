import numpy as np
from classy import Class
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import zeta

delta_msq_solar = 0.002458
delta_msq_atm = 0.000074
zeta3 = zeta(3)

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
    
    settings_with_sterile = {'use_ncdm_psd_files': "0,0,0",
                             'ncdm_psd_parameters': "1,1,0,0,0,0,-100"}

    neutrino = Class()
    neutrino.set(LambdaCDM_settings)
    neutrino.set(neutrino_mass_settings)
    neutrino.set(settings_with_sterile)
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
    
def finale(e_array,f_array,poly_degree):
    e,f = cdf_array(e_array,f_array)
    
    T,N,poly_coefficients = everything_poly(e,f,poly_degree)
    
    return T,N,poly_coefficients

t_range = np.linspace(1, 1.6, 200)
b_range = np.linspace(0, 2, 200)
d_range = np.linspace(2.5/3, 4/3, 200)
    

def norm(T, b_prime):
    result = (T**3 * (3/2) * zeta3) + b_prime
    return result


def cdf_fit(x, T, b_prime):

    def thermal(eps, T):
        return eps**2 / (np.exp(eps/T) + 1) 
        
    num, err = quad(lambda eps: thermal(eps, T), 0, x)
    denom = norm(T, b_prime)

    return num/denom
    

def find_coeff(eps, cdf_data, T_values, B_values, t_range = None):
    ks = np.zeros((len(T_values), len(B_values)))
    
    for i, t in enumerate(T_values):
        for j, b in enumerate(B_values):
            cdf_fit_model = [cdf_fit(x, t, b) for x in eps]
            ks_val  = np.max(np.abs(cdf_fit_model - cdf_data))
            ks[i, j] = ks_val

    min_index = np.unravel_index(np.argmin(ks), ks.shape)
    best_t = T_values[min_index[0]]
    best_b = B_values[min_index[1]]

    return best_t, best_b


def bfunc_fit(x, frac, d):
    return (frac * (1/6) * (6 - ((x/d)**3 + 3 * (x/d)**2 + 6 * (x/d) + 6) * np.exp(-x/d)))


def find_d(eps, frac, bfunct, d_vals):
    best_d = np.zeros(len(d_vals))
                  
    for i, d in enumerate(d_vals):
        bfunc_fitt = [bfunc_fit(x, frac, d) for x in eps]
        ks_val  = np.max(np.abs(bfunc_fitt - bfunct))
        best_d[i] = ks_val

    min_index = np.unravel_index(np.argmin(best_d), best_d.shape)
    best_D = d_vals[min_index]
        
    return best_D

def find_N(e, f, T, B, D):
    integrand = f * e**3
    integral = np.trapz(integrand, e)
    dividend = ((7 * np.pi**4 * T**4)/120) + (B * 4 * D)
    N = integral/dividend

    return N

def new_finale(e_array,f_array,t = None, b = None, d = None):
    e,f = cdf_array(e_array,f_array)
    cdf = cdf_faster(e,f)

    if t is not None:
        t_val = t
    else:
        t_val = t_range

    if b is not None:
        b_val = b
    else:
        b_val = b_range

    if d is not None:
        d_val = d
    else:
        d_val = d_range
        
    T, B = find_coeff(e, cdf, t_val, b_val)
    
    fraction = B/norm(T, B) 
    delta_cdf = cdf - (cdf*norm(T,0)/norm(T, B))
    
    D = find_d(e, fraction, delta_cdf, d_val)
    N = find_N(e, f, T, B, D)
    arr = [B, 1/D, 0, 0, 0]
    
    return T,N,arr


def cdf_array(e_array,f_array):
    high = np.where(cdf_faster(e_array,e_array**2*f_array)>1-10**-4)[0][0]
    
    k = len(e_array)-high
    e_array_shorter = np.delete(e_array,np.s_[-k:])
    f_array_shorter = np.delete(f_array,np.s_[-k:])
    
    return e_array_shorter,f_array_shorter

def cdf_faster(e,f):
    r = np.zeros(len(e))
    y = e**2 * f
    for i in range(1, len(r)):
        r[i] = r[i-1] + 0.5 * (y[i-1] + y[i]) * (e[i]-e[i-1])
        
    r /= np.trapz(y,e)
    return r

def everything_poly(e_array,f_array,poly_degree):
    diff_still_reverse,T_best,N_best = everything1(e_array,f_array)
    
    diff_smaller_reverse = []
    for i in diff_still_reverse: 
        if i < 0:
            break
        if i > 0:
            diff_smaller_reverse.append(i)
    diff_smaller_correct = diff_smaller_reverse[::-1]
    E_new = e_array[len(e_array)-len(diff_smaller_correct):]
    F_new = f_array[len(e_array)-len(diff_smaller_correct):]
    
    np.polyfit(E_new,np.log(diff_smaller_correct),poly_degree) 
    coefficients = np.polyfit(E_new,np.log(diff_smaller_correct),poly_degree) 
    
    if coefficients[0]>0:
        coefficients[0] = 0 
        
    return T_best,N_best,coefficients

def everything1(e_array,f_array):
    Tbest,Nbest = fit3(e_array,f_array)
    e_array_reverse = e_array[::-1]
    f_array_reverse = f_array[::-1]
    diff_reverse = np.zeros(len(e_array_reverse))
    
    for i in range(len(e_array_reverse)):
        diff_reverse[i] = (f_array_reverse[i])-((Nbest)/(np.exp(e_array_reverse[i]/Tbest)+1))
    return diff_reverse,Tbest,Nbest


@nb.jit(nopython=True)
def fit2(e_array, f_array):
    e_max = e_array[np.where(e_array**2*f_array == np.max(e_array**2*f_array))[0]][0]
    f_max = f_array[np.where(e_array**2*f_array == np.max(e_array**2*f_array))[0]][0]
    
    Del_e = e_array[1] - e_array[0]
    Del_T = (1/2.301)*Del_e
    
    T = e_max/2.301
    N = (((np.exp(e_max/T)+1)*np.max(e_array**2*f_array))/e_max**2)
    Del_N = (((np.exp(e_max/T+Del_T)+1)*np.max(e_array**2*f_array))/e_max**2)
    
    return T,N,Del_T,Del_N

@nb.jit(nopython=True)
def least_sum(e_array,f_array,T, N):
    Sum = 0 
    for i in range(len(e_array)):
        Sum = Sum + ((e_array[i]**2*f_array[i]-((N)*(e_array[i]**2)/(np.exp(e_array[i]/T)+1)))**2)
    return Sum


@nb.jit(nopython=True)
def fit3(e_array,f_array):
    T_0, N_0, T_error, N_error = fit2(e_array,f_array)
    
    T = np.linspace(T_0-T_error,T_0+T_error,100)
    N = np.linspace(N_0-N_error,N_0+N_error,100)
    M = np.zeros((len(T),len(N)))
    for i in range(len(T)):
        for j in range(len(N)):
            M[i,j] = least_sum(e_array,f_array,T[i],N[j])
    
    w = np.where(M==np.amin(M))
    T_best = T[w[0][0]]
    N_best = N[w[1][0]]
    
    return T_best,N_best
    
    
def v_masses_sterile(m_small, normal, in_filename, save_filename, make_plot = False, sum_mnu_filename = False, coeffs = False, parameter = None, bprime = None, d = None):
    actual_data = np.load(in_filename, allow_pickle=True)
    f_array = actual_data['fe'][-1]
    e_array = actual_data['e'][-1]
    
    mass1, mass2, mass3 = v_masses(m_small, normal)

    if parameter is None:
        T_best, N_best, coefficients = new_finale(e_array, f_array)

    else:
        ts = np.linspace(parameter-(parameter*0.0001), parameter, 200)
        T_best, N_best, coefficients = new_finale(e_array, f_array, ts)

    if bprime is None:
        T_best, N_best, coefficients = new_finale(e_array, f_array)

    else:
        bs = np.linspace(bprime-(bprime*0.0001), bprime, 200)
        T_best, N_best, coefficients = new_finale(e_array, f_array, t = None, b = bs)

    if d is None:
        T_best, N_best, coefficients = new_finale(e_array, f_array)

    else:
        ds = np.linspace(d-(d*0.0001), d, 200)
        T_best, N_best, coefficients = new_finale(e_array, f_array, t = None, b = None, d = d)

    
    params = '{},{},{},{},{},{},{}'.format(T_best, N_best, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4])
    
    af = actual_data['scalefactors'][-1]
    tf = actual_data['temp'][-1]
    
    value = 1/(af * tf)
    
    neutrino_mass_settings = {'N_ncdm':3,
                              'use_ncdm_psd_files': "0,0,0",
                              'm_ncdm': '{},{},{}'.format(mass1, mass2, mass3),
                              'T_ncdm': '{},{},{}'.format(value, value, value),
                              'ncdm_psd_parameters': params,
                              'N_ur': 0.0}
    
    neutrino = Class()
    neutrino.set(LambdaCDM_settings)
    neutrino.set(neutrino_mass_settings)
    neutrino.compute()
    
    neutrino_results = dict_results(neutrino)
    N_eff = neutrino_results['Neff']
    
    LambdaCDM = Class()
    LambdaCDM.set(LambdaCDM_settings)
    LambdaCDM.set({'N_ur': '{},{},{}'.format(N_eff, N_eff, N_eff)})
    LambdaCDM.compute()
    
    LambdaCDM_results = dict_results(LambdaCDM)
    
    h = LambdaCDM_results['h']
    
    Pk_Lambda = np.zeros(len(k_array_default))
    Pk_neutrino = np.zeros(len(k_array_default))
    
    for i, k in enumerate(k_array_default):
        Pk_Lambda[i] = LambdaCDM.pk(k*h, 0.) * h**3
        Pk_neutrino[i] = neutrino.pk(k*h, 0.) * h**3
        
    if sum_mnu_filename:
        save_filename += '{}'.format(np.round((mass1+mass2+mass3)*1000,0))[:-2]
        
    np.savez(save_filename, n_results = neutrino_results, LCDM_results = LambdaCDM_results, k_array = k_array_default, Pk_n_array = Pk_neutrino, Pk_L_array = Pk_Lambda, normal_hierarchy = normal, m1 = mass1, m2 = mass2, m3 = mass3, m_small = m_small, sum_nu = mass1+mass2+mass3)

    if(make_plot):
        plt.figure()
        plt.semilogx(k_array_default, (Pk_neutrino - Pk_Lambda)/Pk_Lambda)
        plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
        plt.ylabel(r'$\Delta P / P$')
        plt.show()

    if(coeffs):
        return T_best, N_best, coefficients[0], coefficients[1]

    
    
    
