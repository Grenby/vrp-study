
from tqdm import trange

import gc
import tracemalloc

import psutil
from tqdm import trange

import rm_generator
from vrp_study.pdptw_model import routing_model
from vrp_study.routing_manager import RoutingManager
from loguru import logger as log
process = psutil.Process()


# log.remove()
from loguru import logger
logger.remove()

def print_memory():
    print(f"RSS: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"VMS: {process.memory_info().vms / 1024 / 1024:.2f} MB")


def main():
    log.info("create rm")
    rm: RoutingManager = rm_generator.get_rm()
    log.info("start_solve")
    routing_model.find_optimal_paths(rm)
    del rm


if __name__ == "__main__":
    # print_memory()
    # tracemalloc.start()
    # snapshot1 = tracemalloc.take_snapshot()
    for _ in trange(1):
        main()
        # print_memory()

    gc.collect()

    # snapshot2 = tracemalloc.take_snapshot()
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    # for stat in top_stats:
    #     print(stat)
