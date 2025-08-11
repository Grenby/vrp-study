import dataclasses
import math
import pickle
from typing import List

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
from loguru import logger as log
from tqdm.auto import tqdm

from vrp_study.configs import ModelConfig
from vrp_study.initial_solution_builder import InitialSolutionBuilder
from vrp_study.pdptw_model.routing_model import find_optimal_paths
from vrp_study.routing_manager import RoutingManager, InnerNode


@dataclasses.dataclass
class SolutionBuilder(InitialSolutionBuilder):
    max_problem_size: int = 25
    inverse_weight: bool = False
    save_graph: bool = False
    name: str = ''

    def get_initial_solution(self, routing_manager: RoutingManager) -> List[List[InnerNode]]:
        cg = nx.DiGraph()
        start2end: dict[int, list[InnerNode]] = {}
        for pd in routing_manager.get_pick_up_and_delivery_nodes():
            a: InnerNode = routing_manager.nodes()[pd[0]]
            b: InnerNode = routing_manager.nodes()[pd[1]]
            cg.add_node(a.id)
            start2end[a.id] = [a, b]

        for pd1 in routing_manager.get_pick_up_and_delivery_nodes():
            pd1: list[InnerNode] = [routing_manager.nodes()[i] for i in pd1]
            for pd2 in routing_manager.get_pick_up_and_delivery_nodes():
                pd2: list[InnerNode] = [routing_manager.nodes()[i] for i in pd2]
                a, b = pd1[0], pd1[1]
                c, d = pd2[0], pd2[1]
                if a.id == c.id:
                    continue
                l0 = routing_manager.get_distance(a, b) + routing_manager.get_distance(c, d) + 0.01
                l1 = min(
                    get_len([a, b, c, d], routing_manager),
                    # get_len([c, d, a, b], routing_manager),

                    get_len([a, c, d, b], routing_manager),
                    get_len([a, c, b, d], routing_manager),

                    # get_len([c, a, b, d], routing_manager),
                    # get_len([c, a, d, b], routing_manager),
                )
                if l1 > 0 and math.isinf(l1):
                    continue
                cost = min((l1 - l0) / l0, 2)
                cost = np.exp(cost)
                # print(l)
                if cost < 1.6:
                    if (a.id, c.id) in cg.edges():
                        cg.edges()[a.id, c.id]['length'] = min(cost, cg.edges()[a.id, c.id]['length'])
                        cg.edges()[a.id, c.id]['l_ab'] = routing_manager.get_distance(a, b)
                        cg.edges()[a.id, c.id]['l_cd'] = routing_manager.get_distance(c, d)
                        cg.edges()[a.id, c.id]['l0'] = l0
                        cg.edges()[a.id, c.id]['l1'] = l1
                    else:
                        cg.add_edge(a.id, c.id,
                                    length=cost,
                                    l_ab=routing_manager.get_distance(a, b),
                                    l_cd=routing_manager.get_distance(c, d),
                                    l0=l0,
                                    l1=l1
                                    )

        if self.save_graph:
            with open(f'../data/graphs/{self.name}_cg.pkl', 'wb') as f:
                pickle.dump(cg, f)

        cg = cg.to_undirected()

        # log.info(f"{len(cg.nodes()), len(cg.edges), nx.is_connected(cg)}")

        if self.inverse_weight:
            for u, v, d in cg.edges(data=True):
                d['length'] = 1 / (0.0001 + d['length'])
        ucg = cg.to_undirected()
        if nx.is_connected(ucg):
            graphs = [cg]
        else:
            res = []
            for i, c in enumerate(nx.connected_components(cg)):
                res.append(cg.subgraph(c))
            graphs = res

        car2path = {}
        NUM_SOl = 0
        for cg in graphs:
            G: ig.Graph = ig.Graph.from_networkx(cg)
            l, r = 0.1, 128
            iterations = 0
            if len(cg.edges) == 0:
                cms = [{u} for u in cg.nodes()]
            else:
                cms = find_cms(G, resolution=(l + r) / 2)
                # cms = nx.community.louvain_communities(cg, weight='length', resolution=(l + r) / 2)
                max_len_cms = max(len(c) for c in cms)
                while max_len_cms > self.max_problem_size or max_len_cms < 20:
                    if max_len_cms > self.max_problem_size:
                        l = (r + l) / 2
                    else:
                        r = (l + r) / 2
                    # cms = nx.community.louvain_communities(cg, weight='length', resolution=(l + r) / 2)
                    cms = find_cms(G, resolution=(l + r) / 2)
                    max_len_cms = max(len(c) for c in cms)
                    log.info(f"{max_len_cms, (l + r) / 2}")
                    iterations += 1
                    if iterations == 10:
                        break

            for ii, c in enumerate(cms):
                nodes = [ccc for cc in c for ccc in start2end[cc]]
                cars = [car for car in routing_manager.cars() if car.id not in car2path]

                part = routing_manager.sub_problem(
                    nodes,
                    cars,
                    ModelConfig(max_solution_number=50, max_execution_time_minutes=0.5))

                solution = find_optimal_paths(part)[0]
                for i, s in enumerate(solution):
                    if len(s) > 0:
                        car2path[part.cars()[i].id] = [part.nodes()[point] for point in s if
                                                       part.nodes()[point].id not in {part.cars()[i].start_node.id,
                                                                                      part.cars()[i].end_node.id}]
                log.info(solution)
                # with open(f'./sols/sol_{NUM_SOl}', 'wb') as f:
                #     pickle.dump(car2path, f)
                #     NUM_SOl += 1
                # if ii > 0 and ii % 3 == 0:
                #     cars = [car for car in routing_manager.cars() if car.id in car2path]
                #     nodes = [p for path in car2path.values() for p in path]
                #
                #     part = routing_manager.sub_problem(
                #         nodes,
                #         cars,
                #         ModelConfig(max_execution_time_minutes=0.5)
                #     )
                #
                #     solution = []
                #     for i, car in enumerate(routing_manager.cars()):
                #         if car.id in car2path:
                #             solution.append(car2path[car.id])
                #         # else:
                #         #     solution.append([])
                #     solution = find_optimal_paths(part, init_solution=solution)[0]
                #     for i, s in enumerate(solution):
                #         if len(s) > 0:
                #             car2path[part.cars()[i].id] = [part.nodes()[point] for point in s if
                #                                            part.nodes()[point].id not in {part.cars()[i].start_node.id,
                #                                                                           part.cars()[i].end_node.id}]
                #         else:
                #             del car2path[part.cars()[i].id]

                # with open(f'./sols/sol_{NUM_SOl}', 'wb') as f:
                #     pickle.dump(car2path, f)
                #     NUM_SOl += 1

        solution = []
        for i, car in enumerate(routing_manager.cars()):
            if car.id in car2path:
                solution.append(car2path[car.id])
            else:
                solution.append([])
            # solution.append(car2path[car.id])
        return solution


def find_cms(g: ig.Graph, resolution):
    # Get clustering
    partition = la.find_partition(g,
                                  partition_type=la.CPMVertexPartition,
                                  weights=g.es['length'],  # Исправлено здесь
                                  resolution_parameter=resolution)
    communities = []
    for community in partition:
        node_set = set()
        for v in community:
            node_set.add(g.vs[v]['_nx_name'])
        communities.append(node_set)
    return communities


def get_len(nodes: list[InnerNode], routing_manager: RoutingManager) -> float:
    time = nodes[0].start_time + nodes[0].service_time
    total_length = 0
    prev = nodes[0]
    for node in nodes[1:]:
        time += routing_manager.get_time(prev, node)
        total_length += routing_manager.get_distance(prev, node)
        a, b = node.start_time, node.end_time
        if time > b:
            # log.info(f'time limit: {time}//{b}')
            return float('inf')
        time = max(time, a) + node.service_time
        prev = node
    return total_length
