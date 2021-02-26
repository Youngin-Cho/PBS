import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def generate_block_schedule(num_of_blocks=100):
    num_of_processes = 7
    block_id = ["block{0}".format(i) for i in range(num_of_blocks)]
    a, b = -1.5, 1.5
    loc = [2.26, 2.22, 0.5, 2.07, 1.88, 2.08, 2.78]
    scale = [0.794, 0.888, 0.0, 0.600, 1.12, 0.757, 1.65]
    process_time = np.zeros((num_of_blocks, num_of_processes))
    for i in range(num_of_processes):
        r = np.round(stats.truncnorm.rvs(a, b, loc[i], scale[i], size=num_of_blocks), 1)
        process_time[:, i] = r

    panel_blocks = []
    for i in range(num_of_blocks):
        panel_block = PanelBlock(block_id[i], pd.Series(process_time[i]))
        panel_blocks.append(panel_block)

    return panel_blocks


def import_panel_block_schedule(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    df_schedule = df_schedule.drop(columns=['unit_assy'])
    panel_blocks = []
    for i, block in df_schedule.iterrows():
        panel_block = PanelBlock(block['product'], block.drop(['product']))
        panel_blocks.append(panel_block)
    return panel_blocks


class PanelBlock(object):
    def __init__(self, block_id, block_working_time):
        # block_id
        self.id = block_id
        # 작업 시간 저장
        index = pd.MultiIndex.from_product([[i for i in range(len(block_working_time.index)+1)],
                                            ['start_time', 'process_time', 'process']])
        self.data = pd.Series(index=index, dtype=float)
        for i, process in enumerate(block_working_time.index):
            self.data[(i, 'process_time')] = block_working_time[process]
            self.data[(i, 'process')] = 'Process{0}'.format(i)
        self.data[len(block_working_time.index), 'process'] = 'Sink'
        # 지나간 공정 수
        self.step = 0


if __name__ == "__main__":
    blocks = generate_block_schedule()