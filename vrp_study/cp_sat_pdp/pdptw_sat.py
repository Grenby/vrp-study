from vrp_study.routing_manager import RoutingManager, PDRoutingManager
from ortools.sat.python import cp_model
from loguru import logger as log
from tqdm.auto import tqdm, trange

best = {'preferred_variable_order': 2,
        'clause_cleanup_protection': 1,
        'max_presolve_iterations': 5,
        'cp_model_probing_level': 1,
        'presolve_probing_deterministic_time_limit': 10.0,
        'search_branching': 2,
        'feasibility_jump_linearization_level': 0,
        'fp_rounding': 0,
        'polish_lp_solution': True,
        'linearization_level': 0,
        'cut_level': 2,
        'max_all_diff_cut_size': 128,
        'symmetry_level': 0,
        'num_workers': 8}


def get_solver():
    solver = cp_model.CpSolver()

    for k, v in best.items():
        if isinstance(v, list):
            for ss in v:
                solver.parameters.ignore_subsolvers.append(ss)
        else:
            if 'ignore_subsolvers' in k:
                if v:
                    solver.parameters.ignore_subsolvers.append(k.split(':')[1])
            else:
                exec(f'solver.parameters.{k} = {v}')

    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 60.0 * 30
    return solver


def find_optimal_paths(
        routing_manager: PDRoutingManager,
):
    log.info(f'problem size: {len(routing_manager.nodes())}')

    N = len(routing_manager.nodes())
    P = len(routing_manager.cars())
    M = 1_000_000

    X = {}
    T = {}
    Q = {}

    model = cp_model.CpModel()

    for p in trange(P, leave=False, desc='create_X_{p,i,j}'):
        for i in range(N):
            for j in range(N):
                if i == j:
                    X[p, i, j] = model.new_constant(0)
                else:
                    X[p, i, j] = model.new_bool_var(f'x_{p, i, j}')

    max_time = max(n.late_time for n in routing_manager.nodes())
    max_mass = max(car.capacity for car in routing_manager.cars())

    log.debug(f'max_time: {max_time}, max_mass: {max_mass}')
    for i in range(N):
        T[i] = model.new_int_var(lb=0, ub=max_time, name=f'time_{i}')
        Q[i] = model.new_int_var(lb=0, ub=max_mass, name=f'mass_{i}')

    for p, car in enumerate(routing_manager.cars()):
        model.add(sum(X[p, i, car.start_node.id] for i in range(N)) == 0)
        model.add(sum(X[p, car.end_node.id, i] for i in range(N)) == 0)

        model.add(sum(X[p, car.start_node.id, i] for i in range(N)) == 1)
        model.add(sum(X[p, i, car.end_node.id] for i in range(N)) == 1)
    # todo тут я осознал что это уже не имеет смысла, тк в задаче на сто грузов уже 1.3 млн переменных.
    for i, node in enumerate(routing_manager.nodes()):
        for p, car in enumerate(routing_manager.cars()):
            if car.start_node.id != node.id and car.end_node.id != node.id:
                model.add(
                    sum(X[p, i, j] for j in range(N)) == sum(X[p, j, i] for j in range(N))
                )

            for j in range(N):
                if i == j:
                    continue
                # log.debug(f'time_{i,j}:{routing_manager.get_time(i,j)} ')
                model.add(
                    T[i] + int(routing_manager.get_time(i, j) + routing_manager.nodes()[i].service_time) <=
                    T[j] + M * (1 - X[p, i, j]))

                model.add(
                    Q[i] + routing_manager.nodes()[i].demand <=
                    Q[j] + M * (1 - X[p, i, j]))

        if node.is_transit:
            model.add(
                sum(X[p, j, i] for p in range(P) for j in range(N)) == 1
            )

    for node_a, node_b in routing_manager.get_pick_up_and_delivery_nodes():
        a, b = node_a.id, node_b.id
        model.add(T[a] <= T[b])
        for p in range(P):
            model.add(sum(X[p, k, a] + X[p, k, b] for k in range(N)) == 2 * sum(X[p, k, a] for k in range(N)))

    for i, node in enumerate(routing_manager.nodes()):
        model.add(node.early_time <= T[i])
        model.add(T[i] <= node.late_time)

    obj = sum(
        X[p, i, j] * int(routing_manager.get_distance(i, j)) for p in range(P) for i in range(N) for j in range(N))
    model.minimize(obj)

    solver = get_solver()
    status = solver.solve(model)
    log.info(f'status: {status}')

    for p in range(P):
        for k, v in X.items():
            if k[0] == p and solver.value(v) == 1 and k != (p, 0, 1):
                print(k)
