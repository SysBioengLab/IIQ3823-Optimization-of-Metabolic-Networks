import numpy as np
from scipy.integrate import solve_ivp
import warnings
import contextlib
import io

def dFBA(model_original, limited_substrate, o2_index, init_concentrations, t_span = (0, 10), t_eval = np.linspace(0, 10, 100), params = (1, 0.1)):
    """
    Simulates cell growth and substrate consumption using dynamic Flux Balance Analysis (dFBA).
    This function integrates a system of ODEs where biomass and metabolite concentrations are updated over time,
    and metabolic fluxes are determined by FBA optimization at each time step.

    Note:
        The first element of the state vector corresponds to biomass. Substrate and metabolite indices must be offset accordingly.
        Ensure that the indices for limited_substrate and o2_index refer to the correct positions in model.exchanges.

    Args:
        model_original: The original metabolic model (e.g., cobra.Model).
        limited_substrate (int): Index of the limiting substrate in the model's exchange list.
        o2_index (int): Index of the oxygen exchange in the model's exchange list.
        init_concentrations (array-like): Initial concentrations [biomass, substrates...].
        t_span (tuple, optional): Time interval for the simulation (start, end). Default is (0, 10).
        t_eval (array-like, optional): Time points at which the solution is evaluated. Default is np.linspace(0, 10, 100).
        params (tuple, optional): Kinetic parameters (v_max, k_s) for the Michaelis-Menten equation of the limiting substrate. Default is (1, 0.1).

    Returns:
        sol: Solution object from scipy.integrate.solve_ivp containing the time course of biomass and metabolite concentrations.
    """


    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = model_original.copy()
        v_max, k_s = params

        def system_dynamics(t, y):
            
            X = y[0]
            S = y[limited_substrate + 1]
            O = y[o2_index + 1]

            model.exchanges[limited_substrate].lower_bound = - v_max * S / (k_s + S)

            if O <= 0:
                model.exchanges[o2_index].lower_bound = 0.0
            
            
            sol = model.optimize()
            
            dydt = np.zeros_like(y)

            if sol.status != 'optimal':
            # Si el modelo es infactible, dydt se mantiene en ceros
                pass
            else:
                dydt[0]     = sol.objective_value * X  # Tasa de crecimiento
                dydt[1:]    = sol.fluxes[[rxn.id for rxn in model.exchanges]] * X # Flujos metabólicos

                # Fix for negative concentrations
                dydt[(y <= 0) & (np.concatenate(([sol.objective_value], sol.fluxes[[rxn.id for rxn in model.exchanges]])) <= 0)] = 0

            return dydt


        sol = solve_ivp(
            fun=system_dynamics,
            t_span=t_span,
            t_eval=t_eval,
            y0=init_concentrations,
            rtol=1e-8,
            atol=1e-8,
            method='RK45')

    return sol

