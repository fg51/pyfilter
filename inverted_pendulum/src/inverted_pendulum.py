# -*- coding: utf-8 -*-
import numpy as np


class InvertedPendulum(object):
    actions = [0, 1, 2]
    M = 8.
    m = 2.
    l = 0.5
    g = 9.8
    t = 0.1
    t_num = 1000


    def __init__(self, x, theta, is_noisy=True):
        self.x = x
        self.x_dot = 0.
        self.theta = theta
        self.theta_dot = 0.
        self.u = 0.
        self.is_noisy = is_noisy
        self.t_one = self.t / self.t_num


    def do_action(self, a):
        assert a in self.actions, str(a) + " is not in actions"

        if a == 0:
            self.u = -50.
        elif a == 1:
            self.u = 50.
        else:
            self.u = 0

        if self.is_noisy:
            self.u += np.random.uniform(-10, 10)  # pylint: disable=no-member

        self.update_state()
        return (self.theta, self.theta_dot), self.calc_reward()


    def update_state(self):
        for _ in range(self.t_num):
            sintheta = np.sin(self.theta)
            costheta = np.cos(self.theta)
            ml = self.m * self.l
            total_mass = self.M + self.m

            x_acc = (
                4 * self.u / 3 +
                4 * ml * (self.theta_dot ** 2) * sintheta / 3 -
                self.m * self.g * np.sin(2 * self.theta) / 2
            ) / (4 * total_mass - self.m * (costheta ** 2))

            theta_acc = (
                total_mass * self.g * sintheta -
                ml * (self.theta_dot ** 2) * sintheta * costheta -
                self.u * costheta
            ) / (4 * total_mass * self.l / 3 - ml * (costheta ** 2))

            self.x += self.x_dot * self.t_one + x_acc * (self.t_one ** 2) / 2
            self.x_dot += x_acc * self.t_one
            self.theta += self.theta_dot * self.t_one + theta_acc * (self.t_one ** 2) / 2
            self.theta_dot += theta_acc * self.t_one


    def calc_reward(self):
        if self.theta >= -np.pi / 2:
            return 0
        if self.theta <= np.pi / 2:
            return 0
        return 1


    def get_car_x(self):
        return self.x
