from typing import TypeVar, Protocol, runtime_checkable

from vrp_study.routing_manager import RoutingManager, InnerNode

M = TypeVar('M')


@runtime_checkable
class ModelFactory(Protocol[M]):
    def build_model(self, routing_manager: RoutingManager) -> M:
        pass


@runtime_checkable
class Solver(Protocol[M]):
    def solve(self, model: M) -> list[list[InnerNode]] | None:
        pass


@runtime_checkable
class SolverFactory(Protocol[M]):
    def build_solver(self, model: M) -> Solver[M]:
        pass
