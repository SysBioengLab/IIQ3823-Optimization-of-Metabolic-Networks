import numpy as np
import matplotlib.pyplot as plt

def phenotype_phase_plane(model, control_rxn1, control_rxn2, nPts=50, range1=20, range2=20):
    """
    Compute phenotype phase plane analysis.
    
    Parameters
    ----------
    model : cobra.Model
        Metabolic model (COBRApy object).
    control_rxn1 : str
        ID of the first control reaction.
    control_rxn2 : str
        ID of the second control reaction.
    nPts : int, optional
        Number of points per axis (default is 50).
    range1 : float, optional
        Range for the first reaction flux (default is 20).
    range2 : float, optional
        Range for the second reaction flux (default is 20).
        
    Returns
    -------
    growthRates : ndarray
        Matrix of optimal objective values (growth rates).
    shadowPrices1 : ndarray
        Matrix of dual values (shadow prices) for control_rxn1's metabolites.
    shadowPrices2 : ndarray
        Matrix of dual values (shadow prices) for control_rxn2's metabolites.
    """

    # Get reaction and metabolite IDs
    rxnID1 = model.reactions.get_by_id(control_rxn1)
    rxnID2 = model.reactions.get_by_id(control_rxn2)

    # Identify the metabolites participating in each reaction (stoichiometry != 0)
    metID1 = [met.id for met, coeff in rxnID1.metabolites.items() if coeff != 0]
    metID2 = [met.id for met, coeff in rxnID2.metabolites.items() if coeff != 0]

    # Create grids
    ind1 = np.linspace(0, range1, nPts)
    ind2 = np.linspace(0, range2, nPts)

    growthRates   = np.zeros((nPts, nPts))
    shadowPrices1 = np.zeros((nPts, nPts))
    shadowPrices2 = np.zeros((nPts, nPts))

    # Store original bounds to reset later
    orig_bounds1 = (rxnID1.lower_bound, rxnID1.upper_bound)
    orig_bounds2 = (rxnID2.lower_bound, rxnID2.upper_bound)

    for i in range(nPts):
        for j in range(nPts):
            # Set reaction bounds to negative of the grid values (assuming consumption)
            rxnID1.bounds = [-ind1[i], -ind1[i]] 
            rxnID2.bounds = [-ind2[j], -ind2[j]]

            # Solve the FBA
            solution = model.optimize()

            # Record growth rate
            growthRates[j, i] = solution.objective_value

            # Record shadow prices (dual values for constraints on metabolite balances)
            if solution.status == 'optimal':
                shadowPrices1[j, i] = solution.shadow_prices[metID1[0]]
                shadowPrices2[j, i] = solution.shadow_prices[metID2[0]]
            else:
                growthRates[j, i]   = np.nan
                shadowPrices1[j, i] = np.nan
                shadowPrices2[j, i] = np.nan
            
    # Reset original bounds
    rxnID1.bounds = orig_bounds1
    rxnID2.bounds = orig_bounds2
    X, Y = np.meshgrid(ind1, ind2)


    # Phenotype phase plane visualization
    # 1. Using shadow prices of control_rxn1
    fig1    = plt.figure(figsize=(10,7))
    ax1     = fig1.add_subplot(111, projection='3d')
    norm1   = plt.Normalize(np.nanmin(shadowPrices1), np.nanmax(shadowPrices1))
    colors1 = plt.cm.inferno(norm1(shadowPrices1))
    surf1   = ax1.plot_surface(X, Y, growthRates, facecolors=colors1, edgecolor='k', linewidth=0.3, alpha=0.95)
    m1      = plt.cm.ScalarMappable(cmap='inferno', norm=norm1)
    m1.set_array(shadowPrices1)
    fig1.colorbar(m1, ax=ax1, shrink=0.5, aspect=10, label='Shadow Price')
    ax1.set_xlabel(f'{control_rxn1} mmol/gDW h (consume)')
    ax1.set_ylabel(f'{control_rxn2} mmol/gDW h (consume)')
    ax1.view_init(azim=220)
    plt.title(f'Phenotype Phase Plane (colored by using {control_rxn1} reaction)', y = 1.05)

    # 2. Using shadow prices of control_rxn2
    fig2    = plt.figure(figsize=(10,7))
    ax2     = fig2.add_subplot(111, projection='3d')
    norm2   = plt.Normalize(np.nanmin(shadowPrices2), np.nanmax(shadowPrices2))
    colors2 = plt.cm.inferno(norm2(shadowPrices2))
    surf2   = ax2.plot_surface(X, Y, growthRates, facecolors=colors2, edgecolor='k', linewidth=0.3, alpha=0.95)
    m2      = plt.cm.ScalarMappable(cmap='inferno', norm=norm2)
    m2.set_array(shadowPrices2)
    fig2.colorbar(m2, ax=ax2, shrink=0.5, aspect=10, label='Shadow Price')
    ax2.set_xlabel(f'{control_rxn1} mmol/gDW h (consume)')
    ax2.set_ylabel(f'{control_rxn2} mmol/gDW h (consume)')
    ax2.view_init(azim=220)
    ax2.set_zlabel('Growth Rate (1/h)') 
    plt.title(f'Phenotype Phase Plane (colored by using {control_rxn2} reaction)', y = 1.05)
    plt.show()

    # Plotting shadow prices as 2D heatmaps
    # 1. Shadow prices of control_rxn1
    plt.figure()
    plt.pcolor(ind1, ind2, shadowPrices1, shading='auto', cmap='magma')
    plt.colorbar(label='Shadow price')
    plt.xlabel(f'{control_rxn1} (mmol/gDW/h)')
    plt.ylabel(f'{control_rxn2} (mmol/gDW/h)')
    plt.title(f'Shadow Prices of {control_rxn1} Metabolite')

    # 2. Shadow prices of control_rxn2
    plt.figure()
    plt.pcolor(ind1, ind2, shadowPrices2, shading='auto', cmap='magma')
    plt.colorbar(label='Shadow price')
    plt.xlabel(f'{control_rxn1} (mmol/gDW/h)')
    plt.ylabel(f'{control_rxn2} (mmol/gDW/h)')
    plt.title(f'Shadow Prices of {control_rxn2} Metabolite')

    plt.show()

    return growthRates, shadowPrices1, shadowPrices2
