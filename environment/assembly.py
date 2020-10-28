import random
import simpy
import numpy as np
import pandas as pd

from environment.SimComponents import Process, Sink, Monitor
from environment.PostProcessing import *


class Assembly(object):
    def __init__(self, num_of_processes, len_of_queue, event_path, inbound_panel_blocks=None):
        self.num_of_processes = num_of_processes
        self.len_of_queue = len_of_queue
        self.event_path = event_path
        self.inbound_panel_blocks = inbound_panel_blocks
        self.inbound_panel_blocks_clone = self.inbound_panel_blocks[:]
        self.num_of_parts = len(inbound_panel_blocks)

        self.a_size = len_of_queue
        self.s_size = num_of_processes + num_of_processes * len_of_queue
        self.env, self.model, self.monitor = self._modeling(self.num_of_processes, self.event_path)
        self.queue = []
        self.lead_time = 0.0
        self.part_transfer = np.full(num_of_processes, 0.0)
        self.work_finish = np.full(num_of_processes, 0.0)
        self.block_working_time_pre = np.full(num_of_processes, 0.0)
        self.block_working_time_cur = np.full(num_of_processes, 0.0)
        self.time = 0.0
        self.tau = 0.0
        self.num_of_blocks_put = 0

    def step(self, action):
        done = False
        block = self.queue.pop(action)
        self.block_working_time_pre = self.block_working_time_cur[:]
        self.block_working_time_cur = np.array(block.data[:, 'process_time'])[:self.num_of_processes]
        self.monitor.record(self.env.now, "Source", None, part_id=block.id, event="part_created")
        self.model['Process0'].put(block)
        self.monitor.record(self.env.now, "Source", None, part_id=block.id, event="part_transferred")
        self.num_of_blocks_put += 1
        while True:
            self.env.step()
            if self.model['Process0'].parts_sent - self.num_of_blocks_put == 0:
                while self.env.peek() == self.env.now:
                    self.env.run(self.env.timeout(0))
                break

        part_transfer_update, work_finish_update = self._predict_lead_time(self.block_working_time_cur, self.part_transfer, self.work_finish)
        self.part_transfer = part_transfer_update[:]
        self.work_finish = work_finish_update[:]

        if self.num_of_blocks_put == self.num_of_parts:
            done = True

        if len(self.inbound_panel_blocks) > 0 and len(self.queue) < self.len_of_queue:
            self.queue.append(self.inbound_panel_blocks.pop(0))

        reward = self._calculate_reward()
        next_state = self._get_state()

        self.lead_time = self.part_transfer[-1]
        self.tau = self.env.now - self.time
        self.time = self.env.now
        if done:
            self.env.run()

        return next_state, reward, done

    def reset(self):
        self.env, self.model, self.monitor = self._modeling(self.num_of_processes, self.event_path)
        self.inbound_panel_blocks = self.inbound_panel_blocks_clone[:]
        for panel_block in self.inbound_panel_blocks:
            panel_block.step = 0
        random.shuffle(self.inbound_panel_blocks)
        for i in range(self.len_of_queue):
            self.queue.append(self.inbound_panel_blocks.pop(0))
        self.lead_time = 0.0
        self.part_transfer = np.full(self.num_of_processes, 0.0)
        self.work_finish = np.full(self.num_of_processes, 0.0)
        self.block_working_time_pre = np.full(self.num_of_processes, 0.0)
        self.block_working_time_cur = np.full(self.num_of_processes, 0.0)
        self.time = 0.0
        self.tau = 0.0
        self.num_of_blocks_put = 0
        return self._get_state()

    def _get_state(self):
        state = np.full(self.s_size, 0.0)

        server_feature = np.zeros(self.num_of_processes)
        server_feature[:] = self.part_transfer - self.time
        state[:self.num_of_processes] = server_feature

        job_feature = np.zeros(self.num_of_processes * self.len_of_queue)
        for i in range(len(self.queue)):
            panel_block = self.queue[i]
            working_time = list(panel_block.data[:, 'process_time'])[:self.num_of_processes]
            job_feature[i * self.num_of_processes:i * self.num_of_processes + self.num_of_processes] = working_time

        # job_feature = np.zeros([1 + self.num_of_processes, self.len_of_queue])
        # total_working_time = np.zeros([1, len(self.queue)])
        # predicted_lead_time = np.zeros([self.num_of_processes, len(self.queue)])
        # for i in range(len(self.queue)):
        #     panel_block = self.queue[i]
        #     working_time = list(panel_block.data[:, 'process_time'])[:self.num_of_processes]
        #     part_transfer_predicted, _ = self._predict_lead_time(working_time)
        #     total_working_time[:, i] = np.sum(working_time)
        #     predicted_lead_time[:, i] = part_transfer_predicted
        #
        # temp = np.concatenate((total_working_time, predicted_lead_time))
        # rank = np.array([self._rankmin(i) for i in temp])
        # job_feature[np.where(rank == 0)] = 1

        state[self.num_of_processes:] = job_feature
        return state

    def _calculate_reward(self):
        reward = self.part_transfer[-1] - self.lead_time
        return reward

    def _modeling(self, num_of_processes, event_path):
        env = simpy.Environment()
        model = {}
        monitor = Monitor(event_path)
        for i in range(num_of_processes + 1):
            model['Process{0}'.format(i)] = Process(env, 'Process{0}'.format(i), 1, model, monitor, qlimit=1)
            if i == num_of_processes:
                model['Sink'] = Sink(env, 'Sink', monitor)
        return env, model, monitor

    def _predict_lead_time(self, block_working_time, part_transfer, work_finish):
        part_transfer_update = np.cumsum(block_working_time) + part_transfer[0]
        work_finish_update = np.cumsum(block_working_time) + part_transfer[0]
        for i in range(self.num_of_processes - 1):
            if block_working_time[i] == 0.0:
                part_transfer_update[i + 1:] += (part_transfer[i + 1] - part_transfer_update[i - 1])
                work_finish_update[i + 1] += (work_finish[i + 1] - work_finish_update[i - 1])
                part_transfer_update[i - 1] = part_transfer[i + 1]
                part_transfer_update[i] = part_transfer[i]
                work_finish_update[i] = work_finish[i]
                continue
            delay = part_transfer[i + 1] - part_transfer_update[i]
            if delay > 0.0:
                part_transfer_update[i:] += delay
                work_finish_update[i + 1:] += delay
            if (i == self.num_of_processes - 2) and (block_working_time[i + 1] == 0.0):
                if delay > 0.0:
                    part_transfer_update[i:] -= delay
                part_transfer_update[-1] = part_transfer[-1]
                work_finish_update[-1] = work_finish[-1]
        return part_transfer_update, work_finish_update

    def _rankmin(self, x):
        u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.zeros_like(counts)
        csum[1:] = counts[:-1].cumsum()
        return csum[inv]


if __name__ == '__main__':
    import os
    from environment.panelblock import *
    num_of_processes = 7
    len_of_queue = 10

    event_path = './test_env'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    panel_blocks = import_panel_block_schedule('./data/PBS_assy_sequence_gen_000.csv')
    assembly = Assembly(num_of_processes, len_of_queue, event_path + '/event_PBS.csv', inbound_panel_blocks=panel_blocks)

    s = assembly.reset()
    r_cum = 0.0
    print("reset")
    print(s)
    for i in range(70):
        s_next, r, d = assembly.step(0)
        r_cum += r
        print("step: {0} | parts_sent: {1} | parts_completed: {2} | reward: {3} | cumulative reward: {4}"
              .format(i, assembly.model['Process0'].parts_sent, assembly.model['Sink'].parts_rec, r, r_cum))
        s = s_next
        print(s)
        if d:
            break

    print(assembly.env.now)