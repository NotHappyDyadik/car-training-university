import gym
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

global team_name, folder, env_name
team_name = 'ml_team # 1' # TODO: change your team name
folder = 'tetris_race_qlearning'
env_name = 'TetrisRace-v0' # do not change this
Test_Episodes = 3000

class TetrisRaceQLearningAgent:
    def __init__(self,env,learning_rate = 0.5, discount_factor =0.5,
                 exploration_rate =0.5, exploration_decay_rate =0.5):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.actions = env.unwrapped.actions
        self._num_actions = len(self.actions)
        self.state = None
        self.action = None
        self.q_table = []

        # =============== TODO: Your code here ===============
        #  We'll use tabular Q-Learning in our agent, which means
        #  we need to allocate a Q - Table.Think about what exactly
        #  can be represented as states and how many states should
        #  be at all. Besides this remember about actions, which our
        #  agent will do. Q - table must contain notion about the state,
        #  represented as one single integer value (simplest option) and weights
        #  of each action, which agent can do in current env.

        self.wall_iterator = env.unwrapped.wall_iterator # passed walls counter

    def choose_action(self, observation):
        # =============== TODO: Your code here ===============
        #  Here agent must choose action on each step, solving exploration-exploitation
        #  trade-off. Remember that in general exploration rate is responsible for
        #  agent behavior in unknown world conditions and main motivation is explore world.
        #  Exploitation rate - choose already known actions and moving through known states.
        #  Think about right proportion that parameters for better solution
        self.check_state_exist(observation)
        action = np.random.choice(self.actions)

        return action

    def act(self, state, action, reward, state_):
        # =============== TODO: Your code here ===============
        #  Here agent takes action('moves' somewhere), knowing
        #  the value of Q - table, corresponds current state.
        #  Also in each step agent should note that current
        #  'Q-value' can become 'better' or 'worsen'. So,
        #   an agent can update knowledge about env, updating Q-table.
        #   Remember that agent should choose max of Q-value in  each step
        self.check_state_exist(state_)

    def get_info_about_state(self, state):
        for i in self.q_table:
            stateInfo= state[:2]
            q_table_state= np.asarray(i[2])
            compareFlag = True
            for j in range(0, len(stateInfo)):
                if not stateInfo[j] == q_table_state[j]:
                    compareFlag= False
            if compareFlag:
                return i

    def check_state_exist(self, state):
        # =============== TODO: Your code here ===============
        #  Here agent can write to Q-table new data vector if current
        #  state is unknown for the moment
        pass


class EpisodeHistory:
    def __init__(self,env,
                 learn_rate,
                 discount,
                 capacity,
                 plot_episode_count=200,
                 max_timesteps_per_episode=200,
                 goal_avg_episode_length=195,
                 goal_consecutive_episodes=100
                 ):
        self.lengths = np.zeros(capacity, dtype=int)
        self.plot_episode_count = plot_episode_count
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

        self.lr = learn_rate
        self.df = discount

        self.lvl_step = env.unwrapped.walls_per_level
        self.lvl_num = env.unwrapped.levels
        self.difficulty = env.unwrapped.level_difficulty
        self.point_plot = None
        self.mean_plot = None
        self.level_plots = []
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.lengths[episode_index]

    def __setitem__(self, episode_index, episode_length):
        self.lengths[episode_index] = episode_length

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title("Episode Length History. Team {}".format(team_name))

        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)
        self.ax.set_title("Episode Length History (lr {}, df {})".format(self.lr, self.df))
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")
        self.point_plot, = plt.plot([], [], linewidth=2.0, c="#1d619b")
        self.mean_plot, = plt.plot([], [], linewidth=3.0, c="#df3930")
        for i in range(0, self.lvl_num):
            self.level_plots.append(plt.plot([],[], linewidth =1.0, c="#207232",ls ='--'))

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        x = range(plot_left_edge, plot_right_edge)
        y = self.lengths[plot_left_edge:plot_right_edge]
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update levels plots
        for i in range(1, self.lvl_num+1):
            xl = range(plot_left_edge, plot_right_edge)
            yl = np.zeros(len(xl))
            yl[:] = i * self.lvl_step
            cur_lvl_curve = self.level_plots[i - 1][0]
            cur_lvl_curve.set_xdata(xl)
            cur_lvl_curve.set_ydata(yl)
            self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update rolling mean plot.
        #mean_kernel_size = 101
        #rolling_mean_data = np.concatenate((np.zeros(mean_kernel_size), self.lengths[plot_left_edge:episode_index]))
        #rolling_means = pd.rolling_mean(
        #    rolling_mean_data,
        #    window=mean_kernel_size,
        #    min_periods=0
        #)[mean_kernel_size:]
        #self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
        #self.mean_plot.set_ydata(rolling_means)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.0001)

    def is_goal_reached(self, episode_index):
        ''' DO NOT CHANGE THIS FUNCTION CODE.'''
        # From here agent will receive sygnal about end of learning
        arr = self.lengths[episode_index - self.goal_consecutive_episodes + 1:episode_index + 1]
        avg = np.average(arr)
        if self.difficulty == 'Easy':
            answer = avg >= self.goal_avg_episode_length + 0.5
        elif len(arr)>0:
            density = 2 * np.max(arr) * np.min(arr) / (np.max(arr) + np.min(arr))
            answer = avg >= self.goal_avg_episode_length + 0.5 and density >= avg

        return answer


class Controler:
    
    def __init__(self, parent_mode = True, episodes_num = Test_Episodes, global_env = []):
        self.team_name =  team_name
        self.exp_dir = folder + '/' + self.team_name
        random_state = 0
        self.agent_history = []
        self.history_f = True

        self.window = 50

        if parent_mode == False:
            # ====== TODO: your code here======
            # To run env with different parameters you can use another values of named variables, such as:
            #
            # walls_num = x --> number of obstacles (walls). The number must be  x > 6 and x % 3 == 0
            # walls_spread = x --> distance between walls. Too small value leads to no solution situation
            # episodes_to_run = x --> number of agent's tries to learn
            # world_type = 'Fat' or 'Thin' --> make objects more thicker or thinner
            # smooth_car_step = 5 -->  smoothness of car moves (value of step by x)
            # level_difficulty ='Easy' or 'Medium' --> change number of bricks in walls
            # car_spawn = 'Random' or 'Center' --> place, where car starts go
            #
            # EXAMPLE:
            # env.__init__(walls_num = 6, walls_spread = 3, episodes_to_run = episodes_num)
            #
            # Best choice will try any of this different options for better understanding and
            # optimizing the solution.
            env = gym.make(env_name)
            env.__init__(smooth_car_step=2, world_type="Fat", walls_num=30,
                         walls_spread=20, episodes_to_run=episodes_num, level_difficulty='Easy', car_spawn='Center')
            env.seed(random_state)
            np.random.seed(random_state)
            self.lr = 10
            self.df = 10
            self.exr = 10
            self.exrd = 10

            self.env = gym.wrappers.Monitor(env, self.exp_dir + '/video', force=True, resume=False,
                                            video_callable=self.video_callable)
            episode_history, end_index = self.run_agent(self, self.lr, self.df, self.exr, self.exrd, self.env, verbose=False)
        else:
            # Here all data about env will received from main script, so
            # each team will work with equal initial conditions
            # ====== TODO: your code here======
            env = global_env
            env.seed(random_state)
            np.random.seed(random_state)

            self.env = gym.wrappers.Monitor(env, self.exp_dir + '/video', force=True, resume=False,
                                            video_callable=self.video_callable)
            episode_history, end_index = self.run_agent(self, self.lr, self.df, self.exr, self.exrd, self.env, verbose=False)

    def run_agent(self, rate, factor, exploration, exp_decay, env, verbose=False):
        max_episodes_to_run = env.unwrapped.total_episodes
        max_timesteps_per_episode = env.unwrapped.walls_num

        goal_avg_episode_length = env.unwrapped.walls_num
        wall_coef = 6 / env.unwrapped.walls_num
        goal_consecutive_episodes = int(wall_coef * self.window)  # how many times agent can consecutive run succesful

        plot_episode_count = 200
        plot_redraw_frequency = 10

        # =============== TODO: Your code here ===============
        #   Create a Q-Learning agent with proper parameters.
        #   Think about what learning rate and discount factor
        #   would be reasonable in this environment.

        agent = TetrisRaceQLearningAgent(env,
                                         learning_rate=rate,
                                         discount_factor=factor,
                                         exploration_rate=exploration,
                                         exploration_decay_rate=exp_decay
                                         )
        # ====================================================
        episode_history = EpisodeHistory(env,
                                         learn_rate=rate,
                                         discount=factor,
                                         capacity=max_episodes_to_run,
                                         plot_episode_count=plot_episode_count,
                                         max_timesteps_per_episode=max_timesteps_per_episode,
                                         goal_avg_episode_length=goal_avg_episode_length,
                                         goal_consecutive_episodes=goal_consecutive_episodes)
        episode_history.create_plot()

        finish_freq = [0.5, True]  # desired percent finishes in window, flag to run subtask once
        for episode_index in range(0, max_episodes_to_run):
            timestep_index = 0
            observation = env.reset()

            while True:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)  # Perform the action and observe the new state.

                if verbose == True:
                    env.render()
                    # self.log_timestep(timestep_index, action, reward, observation)

                if done and timestep_index < max_timesteps_per_episode - 1:
                    reward = -max_episodes_to_run

                agent.act(observation, action, reward, observation_)
                observation = observation_

                if done:
                    self.done_manager(self, episode_index, [], [], 'D')
                    if self.done_manager(self, episode_index, [], finish_freq, 'S') and finish_freq[1]:
                        finish_freq[1] = False

                    episode_history[episode_index] = timestep_index + 1
                    if verbose or episode_index % plot_redraw_frequency == 0:
                        episode_history.update_plot(episode_index)

                    if episode_history.is_goal_reached(episode_index):
                        print("Goal reached after {} episodes!".format(episode_index + 1))
                        end_index = episode_index + 1
                        self.done_manager(self, [], plt, [], 'P')

                        return episode_history, end_index
                    break
                elif env.unwrapped.wall_iterator - timestep_index > 1:
                    timestep_index += 1
        print("Goal not reached after {} episodes.".format(max_episodes_to_run))
        end_index = max_episodes_to_run
        return episode_history, end_index

    def done_manager(self, episode_ind, plt, top, mode):
        # Call this function to handle episode end event and for storing some
        # result files, pictures etc

        if mode == 'D':  # work with history data
            refresh_each = 100
            self.agent_history.append(self.env.unwrapped.wall_iterator)
            if episode_ind % refresh_each == 0 and self.history_f:
                root = self.exp_dir.split('/')[0]
                base = '/_data'
                path = root + base
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(path + '/' + self.team_name + '.pickle', 'wb') as f:
                    pickle.dump(self.agent_history, f)
        if mode == 'P':  # work woth progress plot
            path = self.exp_dir + '/learn_curve'
            name = '/W ' + str(self.env.unwrapped.walls_num) + \
                   '_LR ' + str(self.lr) + '_DF ' + str(self.df)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name + '.png')
        if mode == 'S':  # call subtasks when condition
            if episode_ind > self.window:
                arr = self.agent_history[episode_ind - self.window: episode_ind]
                mx = np.max(arr)
                ind = np.where(arr == mx)[0]
                count = ind.shape[0]
                prc = count / self.window if mx > self.env.unwrapped.walls_per_level * 2  else 0
                x = self.agent_history
                total_finishes = sum(map(lambda x: x > self.env.unwrapped.walls_per_level * 2, x))

                return prc >= top[0] and total_finishes > 100

    def video_callable(episode_id):
        # call agent draw eact N episodes
        return episode_id % 300 == 0

    def log_timestep(self, index, action, reward, observation):
        # print parameters in console
        format_string = "   ".join(['Timestep:{}',
                                    'Action:{}',
                                    'Reward:{}',
                                    'Car pos:{}',
                                    'WallY pos:{}'])
        print('Timestep: format string ', format_string.format(index, action, reward, observation[0], observation[1]))

    def save_history(self, history, experiment_dir):
        # Save the episode lengths to CSV.
        filename = os.path.join(experiment_dir, "episode_history.csv")
        dataframe = pd.DataFrame(history.lengths, columns=["length"])
        dataframe.to_csv(filename, header=True, index_label="episode")


def main(env, parent_mode = True):
    obj = Controler
    obj.__init__(obj, parent_mode= parent_mode, global_env= env)

if __name__ == "__main__":
    if 'master.py' not in os.listdir('.'):
        main([],parent_mode=False)
    else:
        main(env,parent_mode)