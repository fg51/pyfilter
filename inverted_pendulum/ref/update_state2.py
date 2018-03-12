
def update_state(self):
    for i in range(self.t_num):
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)
        ml = self.m * self.l
        total_mass = self.M + self.m

        temp = (self.u + ml * self.theta_dot**2 * sintheta) / total_mass
        thetaacc = ((self.g * sintheta - costheta * temp) /
                    (self.l * (4/3 - self.m * costheta**2 / total_mass)))
        xacc = temp - ml * thetaacc * costheta / total_mass

        self.x += self.t_one * self.x_dot
        self.x_dot += self.t_one * xacc
        self.theta += self.t_one * self.theta_dot
        self.theta_dot += self.t_one * thetaacc
