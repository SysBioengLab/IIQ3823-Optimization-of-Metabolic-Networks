from math import inf, isinf
from enum import IntEnum
from contextlib import contextmanager
from collections import OrderedDict, defaultdict

import cobra
from cobra import Model, Reaction, Metabolite
from gurobipy import Model as GRBModel, GRB, LinExpr

# =============================================================================
# Status/Solution utilities (compatibility)
# =============================================================================

class Status(IntEnum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    OTHER = 4

class Solution:
    def __init__(self, status, fobj=None, values=None):
        self.status = status
        self.fobj = fobj
        self.values = values or {}

class GurobiSteadyComSolver:
    def __init__(self, sense=GRB.MAXIMIZE, name="steadycom"):
        self.problem = GRBModel(name)
        self.problem.Params.OutputFlag = 0  # suppress solver output
        self._vars = {}                     # name -> Var
        self._constrs = {}                  # name -> Constr
        self._sense_default = sense         # Objective sense
        self._growth_rows = {}              # org_id -> {'constr', 'xvar', 'biomass'}

    def add_variable(self, name, lb=-inf, ub=inf):
        if isinf(lb) and isinf(ub):
            v = self.problem.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=name)
        elif isinf(lb):
            v = self.problem.addVar(lb=-GRB.INFINITY, ub=ub, name=name)
        elif isinf(ub):
            v = self.problem.addVar(lb=lb, ub=GRB.INFINITY, name=name)
        else:
            v = self.problem.addVar(lb=lb, ub=ub, name=name)
        self._vars[name] = v
        return v

    def add_constraint(self, name, coeffs, sense='=', rhs=0.0):
        expr = LinExpr()
        for var_name, coef in coeffs.items():
            expr.addTerms(float(coef), self._vars[var_name])

        if sense == '=':
            c = self.problem.addLConstr(expr == rhs, name=name)
        elif sense == '<':
            c = self.problem.addLConstr(expr <= rhs, name=name)
        elif sense == '>':
            c = self.problem.addLConstr(expr >= rhs, name=name)
        else:
            raise ValueError("Invalid sense")
        self._constrs[name] = c
        return c

    def update(self):
        self.problem.update()

    def _set_objective(self, objective, minimize=False):
        expr = LinExpr()
        for var_name, coef in objective.items():
            expr.addTerms(float(coef), self._vars[var_name])
        self.problem.setObjective(expr, GRB.MINIMIZE if minimize else GRB.MAXIMIZE)

    @contextmanager
    def _temporary_bounds(self, constraints):
        if not constraints:
            yield
            return
        old_bounds = {}
        for vid, (nlb, nub) in constraints.items():
            v = self._vars[vid]
            old_bounds[vid] = (v.LB, v.UB)
            v.LB = -GRB.INFINITY if (nlb is None or (isinf(nlb) and nlb < 0)) else nlb
            v.UB = GRB.INFINITY if (nub is None or (isinf(nub) and nub > 0)) else nub
        try:
            yield
        finally:
            for vid, (olb, oub) in old_bounds.items():
                v = self._vars[vid]
                v.LB = olb
                v.UB = oub

    def solve(self, objective, minimize=False, get_values=True, constraints=None):
        self._set_objective(objective, minimize=minimize)
        with self._temporary_bounds(constraints):
            self.problem.optimize()

        status = self.problem.Status
        if status == GRB.OPTIMAL:
            st = Status.OPTIMAL
        elif status == GRB.INFEASIBLE:
            st = Status.INFEASIBLE
        elif status == GRB.UNBOUNDED:
            st = Status.UNBOUNDED
        else:
            st = Status.OTHER

        if st == Status.OPTIMAL:
            fobj = self.problem.ObjVal
            vals = {name: var.X for name, var in self._vars.items()} if get_values else None
            return Solution(st, fobj=fobj, values=vals)
        else:
            return Solution(st, fobj=None, values=None)

    def register_growth_row(self, org_id, constr_name, xvar_name, biomass_var_name):
        self._growth_rows[org_id] = {
            'constr': self._constrs[constr_name],
            'xvar': self._vars[xvar_name],
            'biomass': self._vars[biomass_var_name]
        }

    def update_growth(self, value):
        for _org, row in self._growth_rows.items():
            self.problem.chgCoeff(row['constr'], row['xvar'], float(value))
        self.problem.update()

class OrganismWrapper:
    """
    Lightweight wrapper for storing the cobrapy model and the biomass reaction.
    """
    def __init__(self, model: cobra.Model, biomass_reaction_id: str):
        self.model = model
        self.biomass_reaction = biomass_reaction_id

    @property
    def reactions(self):
        return {rxn.id: rxn for rxn in self.model.reactions}

class Community:
    """
    SteadyCom community based on cobrapy.
    - organisms: OrderedDict {org_id: OrganismWrapper}
    - merged_model: merged cobra.Model
    - reaction_map: {(org_id, original_rxn_id): merged_rxn_id}
    - metabolite_map: {(org_id, original_met_id): merged_met_id}
    - biomass_reaction_id: community growth reaction id
    """
    def __init__(self, community_id: str, organism_models: dict, mu_init=1.0):
        self.id = community_id
        self.organisms = OrderedDict()
        self._merged_model = None
        self.reaction_map = {}
        self.metabolite_map = {}
        self.biomass_reaction_id = "community_growth"
        self.mu_init = mu_init

        for org_id, mdl in organism_models.items():
            biomass_rxn = detect_biomass_reaction(mdl)
            self.organisms[org_id] = OrganismWrapper(mdl, biomass_rxn)

    @property
    def merged_model(self) -> cobra.Model:
        if self._merged_model is None:
            self._merged_model = self._merge_models()
        return self._merged_model

    def _merge_models(self) -> cobra.Model:
        comm_model = Model(self.id)
        self.reaction_map   = {}
        self.metabolite_map = {}

        # Default IDs
        ext_comp_id = "ext"
        biomass_id  = "community_biomass"
        comm_growth = self.biomass_reaction_id

        # Shared external compartment
        comm_model.compartments = (comm_model.compartments or {})
        comm_model.compartments[ext_comp_id] = "extracellular environment"

        # Community biomass metabolite
        met_biomass = Metabolite(
            id=biomass_id,
            name="Total community biomass",
            compartment=ext_comp_id
        )
        comm_model.add_metabolites([met_biomass])

        # Community growth reaction (objective)
        r_growth = Reaction(comm_growth)
        r_growth.name = "Community growth rate"
        r_growth.lower_bound = 0.0
        r_growth.upper_bound = float('inf')  # no artificial upper bound
        r_growth.add_metabolites({met_biomass: -1.0})
        comm_model.add_reactions([r_growth])
        comm_model.objective = r_growth

        # External metabolites and inherited EX bounds
        ext_mets_ids = set()
        # Bounds for each external metabolite from EX reactions in the models
        # ex_bounds[met_id] = {'lb': [..], 'ub': [..]}
        ex_bounds = defaultdict(lambda: {'lb': [], 'ub': []})

        # External compartment heuristic
        def is_external_compartment(cid: str) -> bool:
            return cid in {"e", "extracellular", "ext", "exo"}

        # Step 1: collect EX bounds from each original model
        for org_id, org in self.organisms.items():
            for rxn in org.model.reactions:
                if not rxn.boundary:
                    continue
                # consider EX reactions with a single metabolite
                mets = list(rxn.metabolites.items())
                if len(mets) != 1:
                    continue
                (m, coeff) = mets[0]
                # must be an external metabolite
                if not is_external_compartment(m.compartment):
                    continue
                ext_mets_ids.add(m.id)
                ex_bounds[m.id]['lb'].append(float(rxn.lower_bound))
                ex_bounds[m.id]['ub'].append(float(rxn.upper_bound))

        # Step 2: incorporate organisms (compartments, metabolites, and internal reactions)
        for org_id, org in self.organisms.items():

            def rename(old_id: str) -> str:
                return f"{old_id}_{org_id}"

            # Renamed internal compartments
            internal_compartments = set(
                m.compartment for m in org.model.metabolites
                if not is_external_compartment(m.compartment)
            )
            for cid in internal_compartments:
                comm_model.compartments[rename(cid)] = org.model.compartments.get(cid, cid)

            # Metabolites
            for m in org.model.metabolites:
                if not is_external_compartment(m.compartment):
                    new_id = rename(m.id)
                    new_met = Metabolite(
                        id=new_id,
                        name=m.name,
                        compartment=rename(m.compartment)
                    )
                    new_met.formula = getattr(m, "formula", None)
                    new_met.charge  = getattr(m, "charge", None)
                    comm_model.add_metabolites([new_met])
                    self.metabolite_map[(org_id, m.id)] = new_id
                else:
                    if m.id not in comm_model.metabolites:
                        new_met = Metabolite(
                            id=m.id, name=m.name, compartment=ext_comp_id
                        )
                        new_met.formula = getattr(m, "formula", None)
                        new_met.charge = getattr(m, "charge", None)
                        comm_model.add_metabolites([new_met])
                        ext_mets_ids.add(new_met.id)

            # Internal reactions (non-boundary)
            for rxn in org.model.reactions:
                if rxn.boundary:
                    continue

                new_id = rename(rxn.id)
                new_rxn = Reaction(new_id)
                new_rxn.name = rxn.name
                new_rxn.lower_bound = rxn.lower_bound
                new_rxn.upper_bound = rxn.upper_bound

                new_stoich = {}
                for met_obj, coeff in rxn.metabolites.items():
                    if met_obj.id in ext_mets_ids:
                        met_new = comm_model.metabolites.get_by_id(met_obj.id)
                    else:
                        met_new = comm_model.metabolites.get_by_id(rename(met_obj.id))
                    new_stoich[met_new] = float(coeff)

                # If this is the organism's biomass reaction, contribute to community biomass
                if rxn.id == org.biomass_reaction:
                    new_stoich[comm_model.metabolites.get_by_id(biomass_id)] = 1.0

                new_rxn.add_metabolites(new_stoich)
                comm_model.add_reactions([new_rxn])
                self.reaction_map[(org_id, rxn.id)] = new_id

                try:
                    new_rxn.gene_reaction_rule = rxn.gene_reaction_rule
                except Exception:
                    pass

        # Step 3: community exchange reactions with inherited bounds
        for m_id in ext_mets_ids:
            rxn_id = f"EX_{m_id}" if not m_id.startswith("EX_") else m_id
            if rxn_id in comm_model.reactions:
                continue

            lb_list = ex_bounds[m_id]['lb']
            ub_list = ex_bounds[m_id]['ub']

            # If no model had an EX for this metabolite, close by default
            lb = min(lb_list) if lb_list else 0.0
            ub = max(ub_list) if ub_list else 0.0

            ex_rxn = Reaction(rxn_id)
            ex_rxn.name = f"Exchange {m_id}"
            ex_rxn.lower_bound = lb
            ex_rxn.upper_bound = ub
            met = comm_model.metabolites.get_by_id(m_id)
            ex_rxn.add_metabolites({met: -1.0})
            comm_model.add_reactions([ex_rxn])

        return comm_model

# =============================================================================
# Direct LP problem construction from cobrapy
# =============================================================================

def _build_metabolite_reaction_lookup(cobra_model: cobra.Model):
    table = {m.id: {} for m in cobra_model.metabolites}
    for rxn in cobra_model.reactions:
        for met, coeff in rxn.metabolites.items():
            table[met.id][rxn.id] = float(coeff)
    return table

def build_problem(community: Community, growth=1.0, bigM=1000.0):
    """
    Builds the SteadyCom LP for the given community.
    """
    model  = community.merged_model
    solver = GurobiSteadyComSolver(name=f"steadycom_{community.id}")

    # Abundance variables x_org in [0,1]
    for org_id in community.organisms:
        solver.add_variable(f"x_{org_id}", 0.0, 1.0)

    # Flux variables v_r for ALL merged reactions
    for rxn in model.reactions:
        if rxn.boundary:
            solver.add_variable(rxn.id, rxn.lower_bound, rxn.upper_bound)
        else:
            lb = -inf if rxn.lower_bound < 0 else 0.0
            ub =  inf if rxn.upper_bound > 0 else 0.0
            solver.add_variable(rxn.id, lb, ub)

    solver.update()

    # Sum x = 1
    solver.add_constraint(
        "abundance",
        {f"x_{org}": 1.0 for org in community.organisms},
        sense='=',
        rhs=1.0
    )

    # S·v = 0 (mass balance for each metabolite)
    table = _build_metabolite_reaction_lookup(model)
    for m_id in table:
        solver.add_constraint(m_id, table[m_id], sense='=', rhs=0.0)

    # Constraints per organism (bounds proportional to x_org)
    for org_id, org in community.organisms.items():
        for r_id, reaction in org.reactions.items():
            if (org_id, r_id) not in community.reaction_map:
                continue
            merged_rid = community.reaction_map[(org_id, r_id)]

            if r_id == org.biomass_reaction:
                cname = f"g_{org_id}"
                solver.add_constraint(
                    cname,
                    {f"x_{org_id}": growth, merged_rid: -1.0},
                    sense='=',
                    rhs=0.0
                )
                solver.register_growth_row(org_id, cname, f"x_{org_id}", merged_rid)
            else:
                lb = -bigM if isinf(reaction.lower_bound) else float(reaction.lower_bound)
                ub =  bigM if isinf(reaction.upper_bound) else float(reaction.upper_bound)

                if lb != 0.0:
                    # lb*x_org - v_r <= 0
                    solver.add_constraint(
                        f"lb_{merged_rid}",
                        {f"x_{org_id}": lb, merged_rid: -1.0},
                        sense='<',
                        rhs=0.0
                    )
                if ub != 0.0:
                    # ub*x_org - v_r >= 0
                    solver.add_constraint(
                        f"ub_{merged_rid}",
                        {f"x_{org_id}": ub, merged_rid: -1.0},
                        sense='>',
                        rhs=0.0
                    )

    solver.update()

    # Initialize μ
    solver.update_growth(community.mu_init if growth is None else growth)
    return solver

# =============================================================================
# Binary search + SteadyCom/SteadyComVA APIs
# =============================================================================

def binary_search(solver, objective, obj_frac=1.0, minimize=False,
                  max_iters=20, abs_tol=1e-3, constraints=None):
    previous_value = 0.0
    value          = 1.0
    fold           = 2.0
    feasible       = False
    last_feasible  = 0.0

    for i in range(max_iters):
        diff = value - previous_value
        if diff < abs_tol:
            break

        if feasible:
            last_feasible = value
            previous_value = value
            value = fold * diff + value
        else:
            if i > 0:
                fold = 0.5
            value = fold * diff + previous_value

        solver.update_growth(value)
        sol = solver.solve(objective, get_values=False, minimize=minimize, constraints=constraints)
        feasible = (sol.status == Status.OPTIMAL)

    if feasible:
        solver.update_growth(obj_frac * value)
    else:
        solver.update_growth(obj_frac * last_feasible)

    sol = solver.solve(objective, minimize=minimize, constraints=constraints)

    if i == max_iters - 1:
        print("Warning: maximum number of iterations reached in binary search.")

    return sol

def SteadyCom(community: Community, medium=None, solver=None):
    """
    Solves for the steady-state growth rate of a microbial community using a constraint-based optimization approach.

    Inputs:
        - community (Community): An object representing the microbial community, including its metabolic models and initial growth rate.
        - medium (optional): Additional constraints to apply to the optimization problem referring to the growth medium (default: None).
        - solver (optional): A pre-built optimization problem instance. If None, the problem is constructed using the community and its initial growth rate.

    Outputs:
        - sol: The solution object containing the optimal growth rate and flux distribution for the community.

    Functionality:
        - Builds or uses a provided optimization problem for the community.
        - Sets the objective to maximize the community biomass reaction.
        - Uses binary search to find the maximum feasible growth rate under the given constraints.
        - Returns the solution containing the optimal growth rate and associated fluxes.
    """
    if solver is None:
        solver = build_problem(community, growth=community.mu_init)
    objective = {community.biomass_reaction_id: 1.0}
    sol = binary_search(solver, objective, minimize=False, constraints=medium)
    return sol

def SteadyComVA(community: Community, obj_frac=1.0, medium=None, solver=None):
    """
    Performs variability analysis on organism abundances in a microbial community at steady-state growth using constraint-based optimization.
    
    Inputs:
        - community (Community): An object representing the microbial community, including organism IDs, biomass reaction, and initial growth rate.
        - obj_frac (float, optional): Fraction of the optimal community growth rate to use for variability analysis (default: 1.0).
        - medium (optional): Additional constraints to apply to the optimization problem referring to the growth medium (default: None).
        - solver (optional): A pre-built optimization problem instance. If None, the problem is constructed using the community and its initial growth rate.
    
    Outputs:
        - variability (dict): Dictionary mapping each organism ID to a list [min_abundance, max_abundance], representing the minimum and maximum feasible abundances at the specified growth rate.
    
    Functionality:
        - Builds or uses a provided optimization problem for the community.
        - Finds the optimal community growth rate using binary search.
        - Sets the community growth rate to a specified fraction of the optimum.
        - For each organism, minimizes and maximizes its abundance variable under the given constraints.
        - Returns the range of feasible abundances for each organism at the specified growth rate.
    """
    if solver is None:
        solver = build_problem(community, growth=community.mu_init)

    objective = {community.biomass_reaction_id: 1.0}
    sol = binary_search(solver, objective, constraints=medium)
    growth = obj_frac * sol.values[community.biomass_reaction_id]
    solver.update_growth(growth)

    variability = {org_id: [None, None] for org_id in community.organisms}

    # Minimize x_org
    for org_id in community.organisms:
        sol2 = solver.solve({f"x_{org_id}": 1.0}, minimize=True, get_values=False, constraints=medium)
        variability[org_id][0] = sol2.fobj

    # Maximize x_org
    for org_id in community.organisms:
        sol2 = solver.solve({f"x_{org_id}": 1.0}, minimize=False, get_values=False, constraints=medium)
        variability[org_id][1] = sol2.fobj

    return variability

def detect_biomass_reaction(model: cobra.Model) -> str:
    """
    Detects the biomass reaction in a cobrapy model.
    
    Strategy:
      1) If the model objective references a single reaction, use it.
      2) Otherwise, use the first reaction whose id starts with 'Biomass' (case-insensitive).
      3) If not found, raise an exception.
    """
    # 1) explicit objective
    try:
        expr = model.objective.expression
        terms = list(expr.as_coefficients_dict().items())
        if terms:
            # terms: [(Symbol(rxn_var), coef), ...]
            best_var, best_coef = max(terms, key=lambda t: float(t[1]))
            rxn_id = str(best_var).replace("R_", "")
            if rxn_id in {r.id for r in model.reactions}:
                return rxn_id
    except Exception:
        pass

    # 2) prefix heuristic (case-insensitive)
    for rxn in model.reactions:
        if rxn.id.lower().startswith("biomass"):
            return rxn.id

    # 3) failure
    raise ValueError(
        f"Could not detect biomass reaction in model '{model.id}'. "
        f"Set model.objective or rename the reaction (e.g. 'Biomass_Ecoli_core_w_GAM')."
    )

def build_steadycom_model(models, organism_ids, mu_init=1.0, community_id="community") -> Community:
    """
    Constructs a SteadyCom Community object and its merged metabolic model from a list of cobrapy models.

    Inputs:
        - models (list of cobra.Model): Individual organism metabolic models.
        - organism_ids (list of str): Unique identifiers for each organism, in the same order as models.
        - mu_init (float, optional): Initial community growth rate for solver initialization (default: 1.0).
        - community_id (str, optional): Identifier for the community (default: "community").

    Outputs:
        - community (Community): Community instance with merged model, ready for SteadyCom/SteadyComVA analysis.

    Functionality:
        - Validates input lengths and uniqueness of organism IDs.
        - Maps organism IDs to their respective models.
        - Creates a Community object and merges individual models into a single community model.
        - Returns the Community instance for further optimization and analysis.
    """
    if len(models) != len(organism_ids):
        raise ValueError("Length of 'models' and 'organism_ids' must match.")
    if len(set(organism_ids)) < len(organism_ids):
        raise ValueError("'organism_ids' must be unique.")

    org_map = OrderedDict((oid, mdl) for oid, mdl in zip(organism_ids, models))
    community = Community(community_id, org_map, mu_init=mu_init)

    # Force construction to validate
    _ = community.merged_model
    return community
