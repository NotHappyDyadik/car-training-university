"""
Tetris-race system implemented by Ivan Gushchin
Environment, realizing the agent's behavior in the world, similar to the classic Tetris race
with the presence of a machine (agent) and walls (obstacles). The agent's goal is to reach the end,
avoiding collisions. Agent has two options to do - make left move of right move to avoid collision.
"""
# TODO: future release - provide more agent' options

import logging
import math
import gym
import random
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pickle
#import distutils

logger = logging.getLogger(__name__)

class TetrisRaceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, walls_num = 60, walls_spread = 5, episodes_to_run = 30,
                 world_type = 'Fat',smooth_car_step = 5, level_difficulty ='Easy', car_spawn = 'Random'):
        super(TetrisRaceEnv, self).__init__()
    # unmutable gui var
        self.screen_width = 400
        self.screen_height = 604
        self.episode_bar_height = 4 # horizontal on top or on bottom
        self.border_width = self.screen_width / 20 # use for cur episode progress bar
        self.road_width = self.screen_width - self.border_width * 2
        self.road_height = self.screen_height - self.episode_bar_height

    # unmutable model var
        self.levels = 3
        self.walls_num = walls_num
        self.pass_wall = False
        self.pass_count = 0  # walls steps down on car
        self.episode_count = 1
        self.total_episodes = episodes_to_run
        self.y_steps_counter = 0

    # mutable gui var
        self.car_width = self.road_width / 12 if world_type == 'Fat' else 10
        self.car_height = 2 * self.car_width if world_type == 'Fat' else 6 * self.car_width
        self.wall_width =  self.car_height
        self.wall_height =  self.car_height / 3 if world_type == 'Fat' else self.car_height / 6

        self.walls_per_level = self.walls_num / self.levels # noob -> exp -> pro
        assert self.walls_num % self.levels == 0 , 'Number of walls per level is not integer. Please change value of' \
                                                   ' "walls_num" parameter.'
        self.walls_spread = walls_spread
        self.walls_x_num = round(self.road_width / self.wall_width)

        blck =[]
        par = [0] * self.levels
        self.level_difficulty = level_difficulty
        for i in range(self.levels,0,-1):
            if level_difficulty == 'Easy':
                par[i-1] = i+1
            else:
                par[i-1] = i
        [blck.append([(self.walls_x_num - i) - 2, self.walls_x_num - par[i-1]]) for i in
             range(self.levels, 0, -1)]
        self.wall_blocks_per_level = blck # range of max available number of bricks in wall by levels

    # mutable moves var
        self.spawn = car_spawn
        self.car_step = self.car_width / smooth_car_step if world_type == 'Fat' else self.car_width

    # mutabale model var
        cx = []
        # car states
        self.car_states_num = round((self.road_width - self.car_width) / self.car_step) + 1
        [cx.append(self.border_width + i * self.car_step) for i in range(0,self.car_states_num)]
        self.car_states = cx  # car freedom to move by x

        self.wall_states_num = round(self.road_width / self.car_step)  # ++!
        wx =[] #++!
        [wx.append(self.border_width + i * self.car_step) for i in range(0,self.wall_states_num)] #++!
        self.wall_states = wx #++!

        all_states = self.walls_num * self.walls_spread + self.walls_spread
        self.wall_field = np.zeros([all_states, self.wall_states_num])
        self.points_num = int(self.wall_states_num / self.walls_x_num)

    #wall states
        wall_count = 0
        cur_level = 0
        for i in range(self.wall_field.shape[0]):
            if i % self.walls_spread == 0 and i != 0:
                if wall_count % (self.walls_per_level-1) == 0 and wall_count != 0 and cur_level < self.levels-1:
                    cur_level += 1
                w_pos = np.zeros(self.walls_x_num)

                for j in range(0, self.walls_x_num):
                    oc = len((np.where(w_pos == 1))[0]) # ones counter
                    rb = self.wall_blocks_per_level[cur_level][1] # range bound
                    if oc >= rb:
                        break
                    else:
                        w_pos[j] = random.getrandbits(1)

                min_val = self.wall_blocks_per_level[cur_level][0]
                accepted_z_n = self.walls_x_num-min_val
                if len(np.where(w_pos == 0)[0]) > accepted_z_n :  # least min val for cur level
                    ar = np.where(w_pos == 0)[0]
                    num_o = len(np.where(w_pos == 1)[0])
                    coord = np.random.choice(ar,min_val - num_o)
                    w_pos[coord[:]] = 1

                w_ind = np.where(w_pos == 1)[0]
                s =[]; e =[]
                [s.append(i * self.points_num) for i in range(self.walls_x_num)]
                [e.append(i * self.points_num) for i in range(1,self.walls_x_num+1)]
                for j in range(0,len(w_ind)):
                    self.wall_field[i][s[w_ind[j]]:e[w_ind[j]]] =1
                wall_count += 1

        self.action_space = spaces.Discrete(2)
        self.actions =np.array([0, 1])

        self._seed()
        self.reset()
        self.viewer = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _complexity(self):
        self.path_complexity = 0
        zero = self.car_states.index(self.state[0])
        # - count value
        for i in range(self.wall_field.shape[0]):
            if np.any(self.wall_field[i] == 1):
                left_ways = np.where(self.wall_field[i][:zero-1] ==0)
                right_ways = np.where(self.wall_field[i][zero+1:-1] == 0)

                left_lenght = left_ways[0].shape[0] - 1
                left_go = zero - left_ways[0].max() if left_ways[0].shape[0] != 0 else 100
                right_go = right_ways[0].min() if right_ways[0].shape[0] != 0 else 100

                self.path_complexity += np.minimum(left_go, right_go)

        # - normalize value
        hardest_option_x = [self.car_states_num * 0.3, self.car_states_num * 0.4, self.car_states_num * 0.5]
        range_options = [int(self.walls_per_level * hardest_option_x[0]),
                         int(self.walls_per_level * hardest_option_x[1]),
                         int(self.walls_per_level * hardest_option_x[2])]
        normalized_complexity = self.path_complexity / np.sum(range_options)

        self.path_complexity = [self.path_complexity, round(normalized_complexity,3)]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.cur_action = action
        car_x, wall_y, glob_st = self.state

# wall vs car
        car_top = int(self.car_height / self.wall_height)
        nearest_wall_ind =abs(wall_y) + car_top
        direct_flag =True
        side_flag = True

# ---- craash!?----------------
        CL = car_x
        CR = car_x + self.car_width
        condition = [False, False, False]
        delta = car_top + 1
        condition[2] = abs(self.state[1]) == self.wall_field.shape[0]- delta  # epic win

        if condition[2] == False:
            if direct_flag and np.any(self.wall_field[nearest_wall_ind + 1] ==1.): # direct crash
                found_wall_ind = np.where(self.wall_field[nearest_wall_ind + 1] == 1.)[0]
                for i in range(0, len(found_wall_ind), self.points_num):
                    tmp = self.car_states[found_wall_ind[i]:found_wall_ind[i]+self.points_num]
                    WL = tmp[0]; WR = tmp[-1]
                    if not (CR <= WL or CL >= WR):
                        direct_flag = False
                        break
                self.pass_wall = True

            if self.pass_wall and np.any(self.wall_field[nearest_wall_ind + 1] == 0.): # side crash
                block_ind = np.where(self.wall_field[nearest_wall_ind-self.pass_count] ==1.)
                block_ind = list(block_ind[0])
                block_xs = []; block_xe = []
                for i in range(0,len(block_ind),self.points_num):
                    tmp = self.car_states[block_ind[i]:block_ind[i] + self.points_num]
                    block_xs.append(tmp[0])
                    block_xe.append(tmp[1])

                if side_flag and CR in block_xs or CR in block_xe or CL in block_xs or CL in block_xe:
                    # TODO: rewrite side crash handler in future releases
                    side_flag = False
                    self.pass_wall = False
                    self.pass_count = 0

                self.pass_count += 1

            if self.pass_count == int(self.car_height / self.wall_height)+2:
                self.pass_wall = False
                self.pass_count =0
                self.wall_iterator += 1

# car states inc / dec
        if car_x > self.car_states[0] and car_x < self.car_states[-1]:
            car_x = car_x + self.car_step if action else car_x - self.car_step
            if action == 1:
                glob_st = glob_st + self.car_step + self.road_width
            else:
                glob_st = glob_st - self.car_step + self.road_width
        elif car_x == self.car_states[0]:
            car_x += self.car_step
            glob_st = glob_st + self.car_step + self.road_width
            self.cur_action = 1
        else:
            car_x -= self.car_step
            glob_st = glob_st - self.car_step + self.road_width
            self.cur_action = 0
# wall states dec
        wall_y -= 1

        self.state = car_x,wall_y, glob_st

        condition[0] = direct_flag == False
        condition[1] = side_flag == False

        done = bool(any(condition))

        if not done:
            reward = 0.
        else:
            self.episode_count += 1
            reward = -1.

        self.y_steps_counter += 1
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.wall_iterator = 1
        self.pass_wall = False
        self.pass_count = 0

        if self.spawn == 'Center':
            rnd = self.car_states[int(round(self.car_states_num / 2))]
        if self.spawn == 'Random':
            rnd = random.choice(self.car_states)
        self.state = (rnd, 0, rnd ) # cx, wy, global
        self._complexity()

        return np.array(self.state)

    def _render(self,mode = 'human', close = False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height, display=self.display)

            # finish line
            self.FGo = []
            for i in range(12):
                W = self.road_width / 12
                self.FGo.append(rendering.Transform())
                if i % 2 == 0:
                    l, r, t, b = self.border_width + W*i, self.border_width + W*i + W, \
                                 (self.wall_field.shape[0] - 1) * self.wall_height, \
                                 (self.wall_field.shape[0] - 1) * self.wall_height - self.wall_height
                    Fin = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    Fin.add_attr(self.FGo[i])
                    Fin.set_color(0.4, 0.5, 0.5)
                    self.viewer.add_geom(Fin)
                else:
                    l, r, t, b = W + self.border_width + W * i, self.border_width + W * i, \
                                 (self.wall_field.shape[0] - 2) * self.wall_height, \
                                 (self.wall_field.shape[0] - 2) * self.wall_height - self.wall_height
                    Fin = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    Fin.add_attr(self.FGo[i])
                    Fin.set_color(0.4, 0.5, 0.5)
                    self.viewer.add_geom(Fin)

            # episode progress bar and slider
            Episode = rendering.FilledPolygon([(self.border_width,self.road_height),
                                               (self.border_width,self.screen_height),
                                               (self.screen_width - self.border_width,self.screen_height),
                                               (self.screen_width - self.border_width, self.road_height)
                                              ])
            Episode.set_color(0, 0, 0)
            self.viewer.add_geom(Episode)

            EpisodeSlider = rendering.FilledPolygon([(self.border_width,self.road_height),
                                                     (self.border_width,self.screen_height),
                                                     (self.border_width + self.car_step, self.screen_height),
                                                     (self.border_width + self.car_step, self.road_height)
                                                    ])
            self.EpisodeGo = rendering.Transform()
            EpisodeSlider.add_attr(self.EpisodeGo)
            EpisodeSlider.set_color( 1, 0, 1)
            self.viewer.add_geom(EpisodeSlider)

            # borders
            LeftBorder = rendering.FilledPolygon([(0,0),(0,self.screen_height),
                                                  (self.border_width,self.screen_height),
                                                  (self.border_width,0)
                                                ])
            RightBorder = rendering.FilledPolygon([(self.screen_width - self.border_width, 0),
                                                   (self.screen_width - self.border_width, self.screen_height),
                                                   (self.screen_width,self.screen_height),
                                                   (self.screen_width,0)
                                                  ])
            LeftTrack = rendering.Line((self.border_width,0),(self.border_width,self.screen_height))
            RightTrack = rendering.Line((self.screen_width - self.border_width,0),
                                        (self.screen_width - self.border_width,self.screen_height))
            LeftBorder.set_color(.8, .6, .4)
            LeftTrack.set_color(0, 0, 0)
            RightBorder.set_color(.8, .6, .4)
            RightTrack.set_color(0, 0, 0)
            self.viewer.add_geom(LeftBorder)
            self.viewer.add_geom(LeftTrack)
            self.viewer.add_geom(RightBorder)
            self.viewer.add_geom(RightTrack)

            # progress bar scale and slider
            scale = self.screen_height / self.walls_num
            for k in range(0, self.walls_num):
                self.track_l = rendering.Line((0, k * scale), (self.border_width, k * scale))
                self.track_r = rendering.Line((self.screen_width - self.border_width, k * scale),
                                              (self.screen_width, k * scale))
                self.track_l.set_color(0, 0, 0)
                self.track_r.set_color(0, 0, 0)
                self.viewer.add_geom(self.track_l)
                self.viewer.add_geom(self.track_r)

            self.BarStep = (self.wall_iterator - 1) * (self.screen_height / self.walls_num)
            ProgressBar_L = rendering.FilledPolygon([(0, 0),
                                                     (0, self.BarStep + scale),
                                                     (self.border_width, self.BarStep + scale),
                                                     (self.border_width, 0)])
            ProgressBar_R = rendering.FilledPolygon([(self.screen_width - self.border_width, 0),
                                                     (self.screen_width - self.border_width, self.BarStep + scale),
                                                     (self.screen_width, self.BarStep + scale),
                                                     (self.screen_width, 0)])
            self.BarLGo = rendering.Transform()
            self.BarRGo = rendering.Transform()
            ProgressBar_L.add_attr(self.BarLGo)
            ProgressBar_R.add_attr(self.BarRGo)
            ProgressBar_L.set_color(.1, .5, .9)
            ProgressBar_R.set_color(.1, .5, .9)
            self.viewer.add_geom(ProgressBar_L)
            self.viewer.add_geom(ProgressBar_R)

            # --- walls ----
            self.WGo = []
            by_y = self.wall_field.shape[0]
            by_x = self.wall_field.shape[1]

            w_c = 0
            for i in range(by_y-1,-1,-1):  # each line
                self.WGo.append(rendering.Transform())
                if np.any(self.wall_field[i]) == 1.0:  # got wall
                    ind = np.where(self.wall_field[i] == 1.0)[0]
                    walls_amount = int(len(ind)/ self.points_num)
                    tmp = np.split(ind, walls_amount)

                    for j in range(0, walls_amount):
                        xs = tmp[j][0]
                        xe = tmp[j][-1]
                        yt = i

                        l,r,t,b = self.wall_states[xs],self.wall_states[xe] + self.car_step,\
                                  yt*self.wall_height,yt*self.wall_height - self.wall_height
                       # print(l,r)
                        WWW = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                        WWW.add_attr(self.WGo[w_c])
                        WWW.set_color(.9, .0, .0)
                        self.viewer.add_geom(WWW)
                    w_c += 1

            # --- car ---
            l, r, t, b = 0, self.car_width, self.car_height , -self.car_height
            Car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.CarGo = rendering.Transform()
            Car.add_attr(self.CarGo)
            self.viewer.add_geom(Car)

        x = self.state

        carx = x[0]
        self.CarGo.set_translation(carx, 0)

        # walls move
        for i in range(0, self.wall_field.shape[0]):
            self.WGo[i].set_translation(0, x[1] * self.wall_height)

        # progress bars move
        self.BarStep = (self.wall_iterator - 1) * (self.screen_height / self.walls_num)
        self.BarLGo.set_translation(0, self.BarStep)
        self.BarRGo.set_translation(0, self.BarStep)
        ep_step = self.road_width / self.total_episodes
        self.EpisodeGo.set_translation((self.episode_count-1) * ep_step,0)
        for i in range(12):
            self.FGo[i].set_translation(0, x[1] * self.wall_height)


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')






