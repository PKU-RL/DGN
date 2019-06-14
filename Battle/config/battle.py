""" battle of two armies """

import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 12, 'speed': 1,
         'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(1),
         'damage': 2, 'step_recover': 0.4,

         'step_reward': 0,  'kill_reward': 0, 'dead_penalty': -2, 'attack_penalty': -0.01,
         })

    big = cfg.register_agent_type(
        "big",
        {'width': 1, 'length': 1, 'hp': 8, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0,

         'step_reward': 0,  'kill_reward': 5, 'dead_penalty': 0, 'attack_penalty': -0.02,
         })

    g0 = cfg.add_group(small)
    g1 = cfg.add_group(big)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value= 5)
    #cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=a, value= -0.01)
    

    return cfg