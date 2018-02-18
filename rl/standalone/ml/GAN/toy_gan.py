from rl.misc.utilies import fig_to_image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

sns.set(color_codes=True)
seed = 42
np.random.seed(seed=seed)
tf.set_random_seed(seed)

class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

def linear(input, output_dim, scope=None, stddev = 1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(name='w',
                            shape=[input.get_shape()[1], output_dim],
                            initializer=tf.random_normal_initializer(stddev))
        b = tf.get_variable(name='b',
                            shape=[output_dim],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, w) + b

def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, scope='g0'))
    h1 = linear(h0, 1, 'g1')
    return h1

def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))
    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3

def minibatch(input, num_kernels=5, kernel_dim=3):
    """"""
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1,2,0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                         global_step=step,
                                                         var_list=var_list)
    return opt

def log(x):
    return tf.log(tf.maximum(x, 1e-5))

batch_size = 8
hidden_size = 4
mini_batch = True
log_every = 10

class GAN(object):
    def __init__(self):

        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(batch_size, 1))
            self.G = generator(self.z, hidden_size)

        self.x = tf.placeholder(tf.float32, shape=(batch_size, 1))
        with tf.variable_scope('D'):
            self.D_x = discriminator(self.x, hidden_size, mini_batch)

        with tf.variable_scope('D', reuse=True):
            self.D_z = discriminator(self.G, hidden_size, mini_batch)

        # define the loss for discriminator and generator networks
        self.loss_d = tf.reduce_mean(-log(self.D_x) - log(1 - self.D_z))
        self.loss_g = tf.reduce_mean(-log(self.D_z))

        #
        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        #
        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)

def train(model, data, gen):
    frames = []
    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        #
        num_steps = 5000
        for step in range(num_steps + 1):
            # update discriminator
            x = data.sample(batch_size)
            z = gen.sample(batch_size)
            loss_d, _, = session.run([model.loss_d, model.opt_d],
                                     {model.x: np.reshape(x, (batch_size, 1)),
                                      model.z: np.reshape(z, (batch_size, 1))})

            # update generator
            z = gen.sample(batch_size)
            loss_g, _ = session.run([model.loss_g, model.opt_g],
                                    {model.z : np.reshape(z, (batch_size, 1))})

            if step % log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
                samps = samples(model, session, data, gen.range, batch_size)
                fig = plot_distributions(samps, gen.range)
                frame = fig_to_image(fig=fig)
                plt.close()
                frames.append(frame)
        imageio.mimsave("example.gif", frames, fps=10)
        #

def samples(model, session, data, sample_range, batch_size, num_points=10000, num_bins=100):
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)
    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size*i:batch_size*(i+1)] = session.run(model.D_x,
                                                        {model.x: np.reshape(xs[batch_size*i:batch_size*(i+1)],(batch_size, 1))})

    # data disctribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size*i: batch_size*(i+1)] = session.run(model.G,
                                                        {model.z: np.reshape(zs[batch_size*i:batch_size*(i+1)], (batch_size, 1))})
    pg, _ = np.histogram(g, bins=bins, density=True)
    return db, pd, pg

def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    return f
    # plt.show()

def main():
    model = GAN()
    train(model, DataDistribution(), GeneratorDistribution(range=8))


if __name__ == '__main__':
    main()