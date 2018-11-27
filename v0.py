from gym.envs.box2d import car_racing
from pyglet.window import key
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt

a = np.array([0.0, 0.0, 0.0])

def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0
    if k == key.UP:    a[1] = 0
    if k == key.DOWN:  a[2] = 0

def flr(x):
    return int(x)

env = car_racing.CarRacing()
env.render()
record_video = False
if record_video:
    env.monitor.start('/tmp/video-test', force=True)
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        st, r, done, info = env.step(a)
        s = st[:80, 7:87, :]
        grayscale = st
        grayscale[:] = grayscale.mean(axis=-1, keepdims=1)
        low_res = np.vectorize(flr)(skimage.measure.block_reduce(s, (8, 8, 1), np.mean))
        left_grass = np.sum(low_res[6:, 2:4, 1]) - np.sum(low_res[6:, 5:7, 1])
        if left_grass > 60:
            a[0] = +0.7
            a[2] = +0.09
        elif left_grass < -60:
            a[0] = -0.7
            a[2] = +0.09
        else:
            a[0] = 0.0
            a[2] = 0.0
            if steps % 2 == 0:
                a[1] = +0.6
            else:
                a[1] = 0.0
        total_reward += r
        if steps % 200 == 0 or done:
            print('--------------------------')
            #print(s.shape)
            #print(low_res)
            print(total_reward)
            plt.imshow(grayscale)
            plt.show()
        steps += 1
        if steps > 2000:
            restart = True
        if not record_video:
            env.render()
        if done or restart: break
env.close()
