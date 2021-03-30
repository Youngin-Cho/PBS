import sys
import random
import simpy
import numpy as np
import pandas as pd

from environment.SimComponents import Process, Sink, Monitor
from environment.PostProcessing import *
from environment.panelblock import *

from io import StringIO
from contextlib import closing


class Assembly(object):
    def __init__(self, num_of_processes, len_of_queue, num_of_parts, event_path, inbound_panel_blocks=None, random_block=False):
        self.num_of_processes = num_of_processes
        self.len_of_queue = len_of_queue
        self.num_of_parts = num_of_parts
        self.event_path = event_path
        if random_block:
            self.inbound_panel_blocks = generate_block_schedule(num_of_parts)
        else:
            self.inbound_panel_blocks = inbound_panel_blocks
            self.inbound_panel_blocks_clone = self.inbound_panel_blocks[:]
        self.random_block = random_block

        self.a_size = len_of_queue
        self.s_size = num_of_processes + num_of_processes * len_of_queue
        self.env, self.model, self.monitor = self._modeling(self.num_of_processes, self.event_path)
        self.queue = []
        self.block = None
        self.stage = 0
        self.time = 0.0
        self.lead_time = 0.0
        self.part_transfer = np.full(num_of_processes, 0.0)
        self.num_of_blocks_put = 0

    def step(self, action):
        self.stage += 1
        done = False
        self.block = self.queue.pop(action)
        block_working_time = np.array(self.block.data[:, 'process_time'])[:self.num_of_processes]
        self.monitor.record(self.env.now, "Source", None, part_id=self.block.id, event="part_created")
        self.model['Process0'].put(self.block)
        self.monitor.record(self.env.now, "Source", None, part_id=self.block.id, event="part_transferred")
        self.num_of_blocks_put += 1
        while True:
            self.env.step()
            if self.model['Process0'].parts_sent - self.num_of_blocks_put == 0:
                while self.env.peek() == self.env.now:
                    self.env.run(self.env.timeout(0))
                break

        part_transfer_update = self._predict_lead_time(block_working_time, self.part_transfer)
        self.part_transfer = part_transfer_update[:]

        if self.num_of_blocks_put == self.num_of_parts:
            done = True

        if len(self.inbound_panel_blocks) > 0 and len(self.queue) < self.len_of_queue:
            self.queue.append(self.inbound_panel_blocks.pop(0))

        reward = self._calculate_reward()
        next_state = self._get_state()

        self.lead_time = self.part_transfer[-1]
        self.time = self.env.now
        if done:
            self.env.run()

        return next_state, reward, done

    def reset(self):
        self.env, self.model, self.monitor = self._modeling(self.num_of_processes, self.event_path)
        if self.random_block:
            self.inbound_panel_blocks = generate_block_schedule(self.num_of_parts)
        else:
            self.inbound_panel_blocks = self.inbound_panel_blocks_clone[:]
            for panel_block in self.inbound_panel_blocks:
                panel_block.step = 0
        random.shuffle(self.inbound_panel_blocks)
        # self.inbound_panel_blocks = sorted(self.inbound_panel_blocks, key=lambda block: block.data.sum(level=1)["process_time"])
        for i in range(self.len_of_queue):
            self.queue.append(self.inbound_panel_blocks.pop(0))
        self.stage = 0
        self.time = 0.0
        self.lead_time = 0.0
        self.part_transfer = np.full(self.num_of_processes, 0.0)
        self.num_of_blocks_put = 0
        return self._get_state()

    def render(self):
        outfile = sys.stdout

        part_list = []
        for i in range(self.num_of_processes):
            process = self.model['Process{0}'.format(i)]
            for server in process.server:
                if server.part:
                    part_list.append(server.part.id)
                else:
                    part_list.append("      ")

        outfile.write("step {0}: [{1}] - [{2}] - [{3}] - [{4}] - [{5}] - [{6}] - [{7}]\n"
                  .format(self.stage, part_list[0], part_list[1], part_list[2], part_list[3], part_list[4], part_list[5], part_list[6]))

        # with closing(outfile):
        #     return outfile.getvalue()


    def _get_state(self):
        state = np.full(self.s_size, 0.0)

        server_feature = np.zeros(self.num_of_processes)
        for i in range(self.num_of_processes):
            process = self.model['Process{0}'.format(i)]
            for server in process.server:
                if server.part:
                    working_time_list = server.part.data.loc[slice(None), 'process_time']
                    working_time = working_time_list[server.part.step]
                    part_start_time = server.working_start
                    if working_time >= (self.env.now - part_start_time):
                        server_feature[i] = working_time - (self.env.now - part_start_time)
        state[:self.num_of_processes] = server_feature

        job_feature = np.zeros(self.num_of_processes * self.len_of_queue)
        for i in range(len(self.queue)):
            panel_block = self.queue[i]
            working_time = list(panel_block.data[:, 'process_time'])[:self.num_of_processes]
            job_feature[i * self.num_of_processes:i * self.num_of_processes + self.num_of_processes] = working_time

        state[self.num_of_processes:] = job_feature
        return state

    def _calculate_reward(self):
        increase = self.part_transfer[-1] - self.lead_time
        reward = 25 - increase #/ self.block.data.sum(level=1)["process_time"]
        return reward

    def _calculate_reward_rr(self):
        event_log = pd.read_csv(self.event_path)
        throughput = cal_throughput(event_log, "Process6", "Process", start_time=0.0, finish_time=self.env.now)
        reward = throughput
        return reward

    # def _calculate_reward_by_first_process_idle_time(self):
    #     event_log = pd.read_csv(self.event_path)
    #     utilization, idle_time, working_time \
    #         = cal_utilization(event_log, "Process0", "Process", start_time=self.time, finish_time=self.env.now)
    #     reward = utilization
    #     return reward
    #
    # def _calculate_reward_by_entire_idle_time(self):
    #     event_log = pd.read_csv(self.event_path)
    #     utilization_list = []
    #     for name in self.model:
    #         if name != "Sink":
    #             utilization, idle_time, working_time \
    #                 = cal_utilization(event_log, name, "Process", start_time=self.time, finish_time=self.env.now)
    #             utilization_list.append(utilization)
    #     reward = np.sum(utilization_list)
    #     return reward

    def _modeling(self, num_of_processes, event_path):
        env = simpy.Environment()
        model = {}
        monitor = Monitor(event_path)
        for i in range(num_of_processes + 1):
            model['Process{0}'.format(i)] = Process(env, 'Process{0}'.format(i), 1, model, monitor, qlimit=1)
            if i == num_of_processes:
                model['Sink'] = Sink(env, 'Sink', monitor)
        return env, model, monitor

    def _predict_lead_time(self, block_working_time, part_transfer):
        part_transfer_update = np.cumsum(block_working_time) + part_transfer[0]
        for i in range(self.num_of_processes - 1):
            if block_working_time[i] == 0.0:
                part_transfer_update[i + 1:] += (part_transfer[i + 1] - part_transfer_update[i - 1])
                part_transfer_update[i - 1] = part_transfer[i + 1]
                part_transfer_update[i] = part_transfer[i]
                continue
            delay = part_transfer[i + 1] - part_transfer_update[i]
            if delay > 0.0:
                part_transfer_update[i:] += delay
            if (i == self.num_of_processes - 2) and (block_working_time[i + 1] == 0.0):
                if delay > 0.0:
                    part_transfer_update[i:] -= delay
                part_transfer_update[-1] = part_transfer[-1]
        return part_transfer_update


if __name__ == '__main__':
    import os
    from environment.panelblock import *
    num_of_processes = 7
    len_of_queue = 10
    num_of_parts = 50

    event_path = './test_env'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    panel_blocks = import_panel_block_schedule('./data/PBS_assy_sequence_gen_000.csv')
    # panel_blocks = generate_block_schedule(num_of_parts)
    assembly = Assembly(num_of_processes, len_of_queue, num_of_parts, event_path + '/event_PBS.csv',
                        inbound_panel_blocks=panel_blocks)

    s = assembly.reset()
    r_cum = 0.0
    print("reset")
    print(s)
    for i in range(70):
        # print(assembly.queue[0].data.sum(level=1)["process_time"])
        s_next, r, d = assembly.step(0)
        r_cum += r
        print("step: {0} | parts_sent: {1} | parts_completed: {2} | reward: {3} | cumulative reward: {4}"
              .format(i, assembly.model['Process0'].parts_sent, assembly.model['Sink'].parts_rec, r, r_cum))
        s = s_next
        print(s)
        if d:
            break

    print(assembly.env.now)