import os
from multiprocessing import Pool
from typing import Optional

import networkx as nx
import numpy as np

import sys

sys.path.append('../')

from vrp_study.managers.pdptw_routing_manager_builder import PDRoutingManagerBuilder
from src.vrp_study.pdptw_model.routing_model import find_optimal_paths
from src.vrp_study.data_loader import parse_data
from src.vrp_study.configs import ModelConfig
from src.vrp_study.data_model import Tariff, Cargo, Node
from src.vrp_study.routing_manager import RoutingManager
from src.vrp_study.pdptw_model.solution_builder import SolutionBuilder
import pickle
from ortools.constraint_solver import routing_enums_pb2
from tqdm import tqdm
from loguru import logger as log


def func(du, dv):
    return np.sqrt((du[0] - dv[0]) ** 2 + (du[1] - dv[1]) ** 2)


def calc(data):
    u, du, p2coordinate = data
    return {(u, v): func(du, dv) for v, dv in p2coordinate.items()}


def build_routing_manager(
        depo: Node,
        cargos: list[Cargo],
        tariff: Tariff,
        p: Optional[Pool] = None
) -> RoutingManager:
    p2coordinates = {}

    p2coordinates.update({
        crg.nodes[i].id: crg.nodes[i].coordinates for crg in cargos for i in range(2)
    })
    p2coordinates[depo.id] = depo.coordinates

    if p is not None:
        res = list(p.imap_unordered(calc, [(u, du, p2coordinates) for u, du in p2coordinates.items()]))
    else:
        res = [calc((u, du, p2coordinates)) for u, du in p2coordinates.items()]

    distance_matrix = {}
    time_matrix = {}
    for r in res:
        for k, v in r.items():
            distance_matrix[k] = v
            time_matrix[k] = v

    routing_manager = PDRoutingManagerBuilder(
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        model_config=ModelConfig(max_execution_time_minutes=0.5)
    )

    routing_manager.add_cargos(cargos)
    routing_manager.add_tariff(tariff)

    routing_manager.add_depo(depo)

    routing_manager = routing_manager.build()
    return routing_manager


def graph_to_real_node(g: nx.DiGraph, rm: RoutingManager) -> nx.DiGraph:
    res = nx.DiGraph()
    for u, v, d in g.edges(data=True):
        a = rm.get_node(u).routing_node.id
        b = rm.get_node(v).routing_node.id
        res.add_edge(a, b, **d)
    return res


def get_file_name(path: str, name: str) -> str:
    i = 0
    while os.path.exists(f'{path}/{i}_{name}'):
        i += 1
    return f'{path}/{i}_{name}'


def rms_calc(data):
    routing_manager, benchmark_type, name = data
    routing_manager: RoutingManager = routing_manager

    sb = SolutionBuilder()

    p = f'../data/Li & Lim benchmark/graphs/{benchmark_type}/graph_{name}.pkl'

    if not os.path.exists(p):
        cg = generate_full_graph(routing_manager)
        with open(p, 'wb') as f:
            pickle.dump(graph_to_real_node(cg, routing_manager), f)

    sol = find_optimal_paths(routing_manager, sb)
    if sol is None:
        return
    sols = []

    for s in sol[0]:
        if len(s) > 0:
            sols.append([routing_manager.nodes()[i].routing_node.id for i in s[1:-1]])
    crg = routing_manager.get_model_config()
    with open(get_file_name(f'../data/Li & Lim benchmark/parsed_solutions/{benchmark_type}',
                            f'{crg.ls_type}_{crg.first_solution_type}_{name}'), 'wb') as f:
        pickle.dump((sols, routing_manager), f)
    del routing_manager, benchmark_type, name, sol, sols


def main():
    log.remove()
    NUM_WORKERS = 3
    MAX_SIZE = NUM_WORKERS  # создаем столько менеджеров. потом в парралель NUM_WORKERS  решают

    arr_ls = [
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    ]

    arr_first = [
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        # routing_enums_pb2.FirstSolutionStrategy.,
        # routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        # routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC

    ]
    pool = True
    x = 1
    if pool:
        with Pool(NUM_WORKERS) as p:
            for benchmark_type in ['pdp_100']:  # os.listdir('../data/Li & Lim benchmark'):
                rms = []
                for name in tqdm(os.listdir(f'../data/Li & Lim benchmark/benchmarks/{benchmark_type}')):
                    depo, cargos, tariff = parse_data(f'../data/Li & Lim benchmark/benchmarks/{benchmark_type}/{name}')
                    for ls in arr_ls:
                        for first in arr_first:
                            rm = build_routing_manager(depo, cargos, tariff, p)
                            rm.get_model_config().ls_type = ls
                            rm.get_model_config().first_solution_type = first
                            rms.append((rm, benchmark_type, name))
                            if len(rms) >= MAX_SIZE:
                                rr = list(p.imap_unordered(rms_calc, rms))
                                rms = []
                                x += 1
                                if x == 2:
                                    return
                break
    else:

        for benchmark_type in ['pdp_100']:  # os.listdir('../data/Li & Lim benchmark'):
            rms = []
            for name in tqdm(os.listdir(f'../data/Li & Lim benchmark/benchmarks/{benchmark_type}')):
                depo, cargos, tariff = parse_data(f'../data/Li & Lim benchmark/benchmarks/{benchmark_type}/{name}')
                for ls in arr_ls:
                    for first in arr_first:
                        rm = build_routing_manager(depo, cargos, tariff)
                        rm.get_model_config().ls_type = ls
                        rm.get_model_config().first_solution_type = first
                        rms.append((rm, benchmark_type, name))
                if len(rms) >= MAX_SIZE:
                    rr = [rms_calc(f) for f in rms]
                    # rr = list(p.imap_unordered(rms_calc, rms))
                    rms = []
            break
    # if len(rms) > 0:
    #     rr = list(p.imap_unordered(rms_calc, rms))
    #     rms = []


if __name__ == '__main__':
    main()
