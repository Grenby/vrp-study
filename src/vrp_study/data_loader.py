from .data_model import Tariff, TariffCost, Node, Cargo


def parse_data(path: str) -> tuple[Node, list[Cargo], Tariff]:
    id2info = {}
    cargos = []

    with open(path, 'r') as file:
        for i, line in enumerate(file):
            line = line.split('\t')
            if i == 0:
                tariff = Tariff(
                    id='car',
                    capacity=int(line[1]),
                    max_count=100,
                    cost_per_distance=[TariffCost(
                        min_dst_km=0,
                        max_dst_km=10000,
                        cost_per_km=1,
                        fixed_cost=0
                    )]
                )
            else:
                c_id = int(line[0])
                x = int(line[1])
                y = int(line[2])

                mass = int(line[3])

                et = int(line[4])
                lt = int(line[5])
                st = int(line[6])

                pick_up = int(line[7])
                delivery = int(line[8])
                if pick_up == delivery:
                    # print(12)
                    depo = Node(
                        id=0,
                        cargo_id=c_id,
                        capacity=0,
                        service_time=0,
                        start_time=0,
                        end_time=lt,
                        coordinates=(x, y)
                    )
                    continue
                if pick_up == 0:
                    if c_id not in id2info:
                        id2info[c_id] = {}
                    id2info[c_id][0] = (x, y, mass, et, lt, st, c_id, delivery)
                else:
                    delivery = c_id
                    c_id = pick_up
                    if c_id not in id2info:
                        id2info[c_id] = {}
                    id2info[c_id][1] = (x, y, mass, et, lt, st, pick_up, delivery)

    for k, v in id2info.items():
        cargos.append(
            Cargo(
                id=k,
                nodes=[
                    Node(
                        cargo_id=k,
                        id=v[i][6] if i == 0 else v[i][7],
                        capacity=v[i][2],
                        service_time=v[i][5],
                        start_time=v[i][3],
                        end_time=v[i][4],
                        coordinates=(v[i][0], v[i][1])
                    )
                    for i in range(2)
                ]
            )
        )

    return depo, cargos, tariff