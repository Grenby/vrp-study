import time
from threading import Lock

from loguru import logger as log
from ortools.constraint_solver import pywrapcp


class SolutionCallback:

    def __init__(self, routing: pywrapcp.RoutingModel | None = None):
        self._routing = routing
        self._best_objective = 1e10
        self.count = 0
        self.lock = Lock()
        self.start_time = time.time()

    def reset_time(self):
        self.start_time = time.time()

    @property
    def routing(self) -> pywrapcp.RoutingModel:
        return self._routing

    @routing.setter
    def routing(self, new_routing: pywrapcp.RoutingModel):
        self._routing = new_routing

    def __call__(self):
        with self.lock:
            self.count += 1
            count = self.count
            value = self.routing.CostVar().Max()
            self._best_objective = min(self._best_objective, value)
            best = self._best_objective
        delta = time.time() - self.start_time
        log.debug(f'time: {delta:.3f}; new solution ({count}): {value}; best solution: {best}')
