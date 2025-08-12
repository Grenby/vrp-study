# def check_solution(List[Cargo], List[Route]):
#     ...
from vrp_study.routing_manager import RoutingManager


def check_solution(rm: RoutingManager, sols: list[list[int]]):
    for sol in sols:
        if len(sol) == 0:
            continue
        nodes = [rm.nodes()[i] for i in sol]
        time = int(nodes[0].service_time + nodes[0].start_time)
        prev = nodes[0]
        for node in nodes[1:]:
            time += int(rm.get_time(prev, node))
            a, b = node.start_time, node.end_time
            if time > b + 1:
                print(time, b)
                return False
            time = int(max(time, a) + node.service_time)
            prev = node
    return True
