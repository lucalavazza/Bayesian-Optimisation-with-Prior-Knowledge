import pygraphviz
from networkx.drawing import nx_agraph
from src.experimental.experiments import optimal_sequence_of_interventions
from src.utils.dag_utils.graph_functions import make_graphical_model
from src.utils.sem_utils.toy_sems import (
    PISHCAT_SEM,
    LinearMultipleChildrenSEM,
    NonStationaryDependentSEM,
    StationaryDependentMultipleChildrenSEM,
    StationaryDependentSEM,
    StationaryIndependentSEM,
)
from src.utils.sem_utils.real_sems import (
    PredatorPreySEM as PPSEM,
    EconSEM,
)
from src.utils.sequential_intervention_functions import get_interventional_grids
from src.utils.utilities import powerset


def setup_PISHCAT(T: int = 3):
    # Setup used in `Developing Optimal Causal Cyber-Defence Agents via Cyber Security Simulation`.

    SEM = PISHCAT_SEM()
    init_sem = SEM.static(SEM)
    sem = SEM.dynamic(SEM)

    slice_node_set = ["P", "I", "S", "H", "C", "A", "T"]
    dag_view = make_graphical_model(0, T - 1, topology="dependent", nodes=slice_node_set, verbose=True)
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))

    # Modulate base structure to fit PISHCAT framework

    # Transitions
    for t in range(T - 1):
        # Add
        G.add_edge("S_{}".format(t), "H_{}".format(t + 1))
        # Remove
        G.remove_edge("P_{}".format(t), "P_{}".format(t + 1))
        G.remove_edge("I_{}".format(t), "I_{}".format(t + 1))
        G.remove_edge("H_{}".format(t), "H_{}".format(t + 1))
        G.remove_edge("C_{}".format(t), "C_{}".format(t + 1))
        G.remove_edge("A_{}".format(t), "A_{}".format(t + 1))
        G.remove_edge("T_{}".format(t), "T_{}".format(t + 1))

    # Emissions
    for t in range(T):
        # Remove
        G.remove_edge("P_{}".format(t), "I_{}".format(t))
        G.remove_edge("S_{}".format(t), "H_{}".format(t))
        G.remove_edge("C_{}".format(t), "A_{}".format(t))
        # Add
        G.add_edge("P_{}".format(t), "H_{}".format(t))
        G.add_edge("P_{}".format(t), "A_{}".format(t))
        G.add_edge("I_{}".format(t), "A_{}".format(t))
        G.add_edge("C_{}".format(t), "T_{}".format(t))

    dag_view

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["P", "I"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"P": [0, 1], "I": [0, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=slice_node_set,
        target_variable="T",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_linear_multiple_children_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = LinearMultipleChildrenSEM()
    init_sem = SEM.static(SEM)
    sem = SEM.dynamic(SEM)

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))
    G.add_edges_from([("X_0", "Y_0"), ("X_1", "Y_1"), ("X_2", "Y_2")])

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_stat_multiple_children_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryDependentMultipleChildrenSEM()
    init_sem = SEM.static(SEM)
    sem = SEM.dynamic(SEM)

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))
    G.add_edges_from([("X_0", "Y_0"), ("X_1", "Y_1"), ("X_2", "Y_2")])

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_stat_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryDependentSEM()
    init_sem = SEM.static(SEM)
    sem = SEM.dynamic(SEM)

    G_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = nx_agraph.from_agraph(pygraphviz.AGraph(G_view.source))

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        G_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_ind_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryIndependentSEM()
    init_sem = SEM.static(SEM)
    sem = SEM.dynamic(SEM)

    G_view = make_graphical_model(
        0, T - 1, topology="independent", nodes=["X", "Z", "Y"], target_node="Y", verbose=True
    )  # Base topology that we build on
    G = nx_agraph.from_agraph(pygraphviz.AGraph(G_view.source))

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, _, true_objective_values, _, _, _ = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return init_sem, sem, G_view, G, exploration_sets, intervention_domain, true_objective_values


def setup_nonstat_scm(T: int = 3):

    #  Load SEM from figure 3(c) in paper (upper left quadrant)
    SEM = NonStationaryDependentSEM(change_point=1)  #  Explicitly tell SEM to change at t=1
    init_sem = SEM.static(SEM)
    sem = SEM.dynamic()

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    dag = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, _, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=dag,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        dag,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_plankton_SEM(T):
    
    # p_dict = create_plankton_dataset(1, T)
    
    P_SEM = PPSEM()
    p_stat_sem = PPSEM.static(P_SEM)
    p_dyn_sem = PPSEM.dynamic(P_SEM)
    
    slice_node_set = ["M", "N", "P", "J", "A", "E", "D"]
    dag_view = make_graphical_model(0, T-1, topology="dependent", nodes=slice_node_set, verbose=True)
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))
    
    for t in range(T-1):
        G.add_edge("P_{}".format(t), "N_{}".format(t + 1))
        G.add_edge("A_{}".format(t), "J_{}".format(t + 1))
        G.remove_edge("M_{}".format(t), "M_{}".format(t+1))
        
    for t in range(T):
        G.remove_edge("J_{}".format(t), "A_{}".format(t))
        G.add_edge("P_{}".format(t), "A_{}".format(t))
        G.add_edge("P_{}".format(t), "E_{}".format(t))
        G.add_edge("J_{}".format(t), "D_{}".format(t))
        G.add_edge("A_{}".format(t), "D_{}".format(t))
        
    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["M", "J", "A"]))
    print
    # Specify the intervention domain for each variable
    intervention_domain = {"M": [40, 160], "J": [0, 20], "A":[0, 100]}
    
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)
    
    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=p_stat_sem,
        structural_equation_model=p_dyn_sem,
        G=G,
        T=T,
        model_variables=slice_node_set,
        target_variable="D",
    )
    
    return (
        p_stat_sem,
        p_dyn_sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_econ_SEM(T, functions_0, functions_t):
    E_SEM = EconSEM(functions_0, functions_t)
    init_sem = EconSEM.static(E_SEM)
    dyn_sem = EconSEM.dynamic(E_SEM)

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "F", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    dag = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))

    dag.add_edges_from([("X_0", "Y_0"), ("X_1", "Y_1"), ("X_2", "Y_2")])
    dag.add_edges_from([("X_0", "Z_0"), ("X_1", "Z_1"), ("X_2", "Z_2")])
    dag.remove_edges_from([("X_0", "F_0"), ("X_1", "F_1"), ("X_2", "F_2")])
    
    # dag.add_edge("X_{}".format(t), "Y_{}".format(t))
    # dag.add_edge("F_{}".format(t), "Z_{}".format(t))

    exploration_sets = list(powerset(['F', 'X']))
    intervention_domain = {'F':[-.3, 10], 'X':[-2., 6.]}
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=dyn_sem,
        G=dag,
        T=T,
        model_variables=["X", "F", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        dyn_sem,
        dag_view,
        dag,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )