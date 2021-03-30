import os
import random
import numpy as np
import tensorflow as tf

from collections import deque
from environment.assembly import Assembly
from environment.panelblock import *


def build_network(in_dim, out_dim):
    inputs = tf.keras.layers.Input(shape=(in_dim,))
    hidden1 = tf.keras.layers.Dense(512, activation="elu", kernel_initializer="he_normal")(inputs)
    hidden2 = tf.keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal")(hidden1)
    hidden3 = tf.keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal")(hidden2)
    outputs = tf.keras.layers.Dense(out_dim)(hidden3)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


class DDQN():
    def __init__(self, env, state_size, action_size, model_path=None, summary_path=None, event_path=None, load_model=False):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        self.summary_path = summary_path
        self.event_path = event_path

        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 100
        self.target_update_iter = 500

        self.memory = deque(maxlen=2000)

        self.model = build_network(self.state_size, self.action_size)
        self.target_model = build_network(self.state_size, self.action_size)

        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_path, max_to_keep=3)
        self.writer = tf.summary.create_file_writer(summary_path)

        if load_model:
            self.ckpt.restore(self.manager.latest_checkpoint)

        self.update_target_model()

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

    def run(self, num_episode):
        avg_max_q_list = []
        reward_list = []
        lead_time_list = []
        loss_list = []
        for e in range(num_episode):
            self.ckpt.step.assign_add(1)

            done = False
            step = 0
            episode_reward = 0
            state = self.env.reset()
            state = np.reshape(state, [1, state_size])

            while not done:
                step += 1

                action = self.get_action(state, len(assembly.queue))
                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                self.avg_q_max += np.amax(self.model(state)[0])

                episode_reward += reward

                self.append_sample(state, action, reward, next_state, done)

                if len(self.memory) >= self.train_start:
                    self.train()

                if e % self.target_update_iter == 0:
                    self.update_target_model()

                state = next_state

                if done:
                    lead_time = self.env.model['Sink'].last_arrival

                    with self.writer.as_default():
                        tf.summary.scalar('Loss/Average Loss', self.avg_loss / float(step), step=e)
                        tf.summary.scalar('Performance/Average Max Q', self.avg_q_max / float(step), step=e)
                        tf.summary.scalar('Performance/Reward', episode_reward, step=e)
                        tf.summary.scalar('Performance/Lead time', lead_time, step=e)
                        avg_max_q_list.append(self.avg_q_max / float(step))
                        reward_list.append(episode_reward)
                        lead_time_list.append(lead_time)
                        loss_list.append(self.avg_loss / float(step))

                    if e % 250 == 0:
                        self.manager.save(e)
                        print("Saved Model at episode %d" % e)

                    self.avg_q_max, self.avg_loss = 0, 0

        log_data = pd.DataFrame(
            {"avg_max_q_": avg_max_q_list, "reward": reward_list, "lead_time": lead_time_list, "loss": loss_list})
        log_data.to_csv(summary_path + "/data.csv")


if __name__ == "__main__":
    num_episode = 10001

    num_of_processes = 7
    len_of_queue = 10
    num_of_parts = 60

    state_size = num_of_processes + num_of_processes * len_of_queue
    action_size = len_of_queue

    load_model = False
    random_block = True

    model_path = '../model/ddqn/queue-%d' % action_size
    summary_path = '../summary/ddqn/queue-%d' % action_size
    event_path = '../simulation/ddqn/queue-%d' % action_size

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    if not os.path.exists(event_path):
        os.makedirs(event_path)

    panel_blocks = generate_block_schedule(num_of_parts)
    assembly = Assembly(num_of_processes, len_of_queue, num_of_parts, event_path + '/log_train.csv',
                        inbound_panel_blocks=panel_blocks, random_block=random_block)

    agent = DDQN(assembly,
                 state_size,
                 action_size,
                 model_path=model_path,
                 summary_path=summary_path,
                 event_path=event_path,
                 load_model=load_model)

    agent.run(num_episode)