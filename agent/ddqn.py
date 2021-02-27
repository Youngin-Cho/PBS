import os
import random
import numpy as np
import tensorflow as tf

from collections import deque
from environment.assembly import Assembly
from environment.panelblock import *


class Network(tf.keras.Model):
    def __init__(self, a_size):
        super().__init__(name='ddqn')
        self.hidden1 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")
        self.hidden2 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")
        self.hidden3 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")
        self.out = tf.keras.layers.Dense(a_size)

    def call(self, inputs):
        hidden1 = self.hidden1(inputs)
        hidden2 = self.hidden2(hidden1)
        hidden3 = self.hidden3(hidden2)
        q_values = self.out(hidden3)
        return q_values


class DDQN():
    def __init__(self, state_size, action_size, model_path=None, load_model=False):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.9
        self.learning_rate = 1e-5
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 100
        self.target_update_iter = 200

        self.memory = deque(maxlen=2000)

        if load_model:
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(model_path)
        else:
            self.model = Network(action_size)
            self.target_model = Network(action_size)
            self.update_target_model()
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

        self.avg_q_max, self.avg_loss = 0, 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, scope):
        if np.random.rand() <= self.epsilon:
            return random.randrange(scope)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0, :scope])

    def train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(states)
            best_actions = np.argmax(predicts, axis=-1)
            best_actions = tf.stop_gradient(best_actions)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            targets = rewards + (1 - dones) * self.discount_factor \
                      * np.array([target_predicts[i, best_actions[i]] for i in range(target_predicts.shape[0])])
            loss = tf.reduce_mean(tf.square(targets - predicts))

            self.avg_loss += loss.numpy()

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
    num_episode = 10001

    num_of_processes = 7
    len_of_queue = 20
    num_of_parts = 100

    state_size = num_of_processes + num_of_processes * len_of_queue
    action_size = len_of_queue

    load_model = False

    model_path = '../model/ddqn/queue-%d' % action_size
    summary_path = '../summary/ddqn/queue-%d' % action_size
    result_path = '../result/ddqn/queue-%d' % action_size
    event_path = '../simulation/ddqn/queue-%d' % action_size

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(event_path):
        os.makedirs(event_path)

    # if random_blocks:
    #     num_of_parts = 100
    #     assembly = Assembly(num_of_processes, len_of_queue, num_of_parts, event_path + '/log_train.csv')
    # else:
    #     panel_blocks = import_panel_block_schedule('../environment/data/PBS_assy_sequence_gen_000.csv')
    #     num_of_parts = len(panel_blocks)
    #     assembly = Assembly(num_of_processes, len_of_queue, num_of_parts, event_path + '/log_train.csv',
    #                         inbound_panel_blocks=panel_blocks)

    panel_blocks = generate_block_schedule(num_of_parts)
    assembly = Assembly(num_of_processes, len_of_queue, num_of_parts, event_path + '/log_train.csv',
                        inbound_panel_blocks=panel_blocks)

    agent = DDQN(state_size, action_size, load_model=load_model)
    writer = tf.summary.create_file_writer(summary_path)

    avg_max_q_list = []
    reward_list = []
    lead_time_list = []
    loss_list = []
    for e in range(1, num_episode):
        done = False
        step = 0
        episode_reward = 0
        state = assembly.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            step += 1

            action = agent.get_action(state, len(assembly.queue))
            next_state, reward, done = assembly.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.avg_q_max += np.amax(agent.model(state)[0])

            episode_reward += reward

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train()

            if e % agent.target_update_iter == 0:
                agent.update_target_model()

            state = next_state

            if done:
                lead_time = assembly.model['Sink'].last_arrival

                with writer.as_default():
                    tf.summary.scalar('Loss/Average Loss', agent.avg_loss / float(step), step=e)
                    tf.summary.scalar('Performance/Average Max Q', agent.avg_q_max / float(step), step=e)
                    tf.summary.scalar('Performance/Reward', episode_reward, step=e)
                    tf.summary.scalar('Performance/Lead time', lead_time, step=e)
                    avg_max_q_list.append(agent.avg_loss / float(step))
                    reward_list.append(episode_reward)
                    lead_time_list.append(lead_time)
                    loss_list.append(agent.avg_loss / float(step))

                if e % 250 == 0:
                    agent.model.save(model_path)
                    print("Saved Model at episode %d" % e)

                agent.avg_q_max, agent.avg_loss = 0, 0

    log_data = pd.DataFrame({"avg_max_q_": avg_max_q_list, "reward": reward_list, "lead_time": lead_time_list, "loss": loss_list})
    log_data.to_csv(summary_path + "/data.csv")
