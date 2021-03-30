import os
import simpy
import random

import tensorflow as tf
import numpy as np
import pandas as pd

from agent.ddqn import build_network
from environment.assembly import Assembly
from environment.panelblock import *
from environment.SimComponents import Source, Process, Sink, Monitor

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def simulation(panel_blocks, num_of_processes, len_of_queue, num_of_parts, model_path=None, event_path=None, method="SPT"):
    if method == "RL":
        state_size = num_of_processes + num_of_processes * len_of_queue
        action_size = len_of_queue

        model = build_network(state_size, action_size)
        ckpt = tf.train.Checkpoint(net=model)
        manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)

        assembly = Assembly(num_of_processes, len_of_queue, num_of_parts, event_path + '/log_test_ddqn.csv',
                            inbound_panel_blocks=panel_blocks)

        done = False
        step = 0
        episode_reward = 0
        avg_q_max = 0.0

        state = assembly.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            assembly.render()

            step += 1

            q_value = model(state)
            avg_q_max += np.amax(q_value[0])

            action = np.argmax(q_value[0, :len(assembly.queue)])
            next_state, reward, done = assembly.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            episode_reward += reward

            state = next_state

            if done:
                lead_time = assembly.model['Sink'].last_arrival
                print("RL - {0} | total lead time: {1}, avg_q_max: {2}, reward: {3}"
                      .format("DDQN", lead_time, avg_q_max / float(step), episode_reward))
        return assembly.model['Sink'].last_arrival
    else:
        if method == "SPT":
            panel_blocks = sorted(panel_blocks, key=lambda block: block.data.sum(level=1)["process_time"])
        elif method == "LPT":
            panel_blocks = sorted(panel_blocks, key=lambda block: block.data.sum(level=1)["process_time"])[::-1]
        elif method == "RANDOM":
            random.shuffle(panel_blocks)

        env = simpy.Environment()
        model = {}
        monitor = Monitor(event_path + '/log_test_{0}.csv'.format(method))
        source = Source(env, "Source", panel_blocks, model, monitor)
        for i in range(num_of_processes + 1):
            model['Process{0}'.format(i)] = Process(env, 'Process{0}'.format(i), 1, model, monitor, qlimit=1)
            if i == num_of_processes:
                model['Sink'] = Sink(env, "Sink", monitor)
        env.run()

        print("dispatching_rule - {0} | total lead time:{1}".format(method, model["Sink"].last_arrival))

        return model['Sink'].last_arrival


if __name__ == "__main__":
    test_iteration = 1
    results = {"RL": [], "SPT": [], "LPT": [], "RANDOM": []}

    num_of_processes = 7
    len_of_queue = 10
    num_of_parts = 600

    state_size = num_of_processes + num_of_processes * len_of_queue
    action_size = len_of_queue

    model_path = '../model/ddqn/queue-%d' % action_size
    event_path = '../simulation/ddqn/queue-%d' % action_size
    test_path = '../test/ddqn/queue-%d' % action_size

    if not os.path.exists(event_path):
        os.makedirs(event_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for i in range(test_iteration):
        print("iteration {0} ".format(i) + "=" * 100)
        for method in results.keys():
            panel_blocks = generate_block_schedule(num_of_parts)
            # panel_blocks = import_panel_block_schedule('../environment/data/PBS_assy_sequence_gen_000.csv')
            # num_of_parts = len(panel_blocks)
            lead_time = simulation(panel_blocks, num_of_processes, len_of_queue, num_of_parts, model_path, event_path, method=method)
            results[method].append(lead_time)

    df_results = pd.DataFrame(results)
    df_results.to_csv(test_path + "/results.csv")