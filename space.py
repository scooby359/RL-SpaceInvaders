import tensorflow as tf  # Deep learning library
import numpy as np  # Handle matrices
import retro

from skimage import transform  # Image pre-processing
from skimage.color import rgb2gray  # Gray out images

import matplotlib.pyplot as plt  # Display graphs

from collections import deque  # Ordered collections
import random
import warnings  # Allows us to ignore warnings from skimage

warnings.filterwarnings('ignore')

# Create environment
env = retro.make(game='SpaceInvaders-Atari2600')

print("The size of our frame is: ", env.observation_space)
print("The action space size is: ", env.action_space.n)

# Crete an encoded list of actions
# e.g. [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0]..
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())


# Pre-process frames - take a frame, grayscale it, resize it, normalise it, return it
def preprocess_frame(frame):
    # Grayscale
    gray = rgb2gray(frame)

    # Crop the screen (remove bit below player)
    # [Up: down, Left:right]
    cropped_frame = gray[8:-12, 4:12]

    # Normalise pixel values
    normalized_frame = cropped_frame / 255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    # A 110 x 84 image with 1 layer (gray)
    return preprocessed_frame

stack_size = 4 # Need a stack of 4

# Initialise deque with 4 blank images (all zeros)
stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Pre-process frames
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear the stacked frames
        stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # As it's a new episode, copy the first frame four times
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append the frame to deque, auto removes the oldest
        stacked_frames.append(frame)

        # Build the stacked state
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

# Model hyperparameters
state_size = [110,84,4]         # 4 images of 110x84 dimension
action_size = env.action_space.n # 8 possible actions
learning_rate = 0.00025         # Alpha (learning rate)

# Training hyperparameters
total_episodes = 50             # Total episodes for training
max_steps = 50000               # Max steps in an episode
batch_size = 64                 # Batch size

# Exploration params for epsilon greedy strategy
explore_start = 1.0             # Exploration probability at start
explore_stop = 0.01             # Minimum exploration rate
decay_rate = 0.00001            # Exponential decay rate for exploration probability

# Q learning hyper parameters
gamma = 0.9                     # Discounting rate

# Memory hyper parameters
pretrain_length = batch_size    # Number of experiences stoed in the memory when initialised for first time
memory_size = 1000000           # Number of experiences the memory can keep

# Preprocessing hyper parameters
stack_size = 4                  # Number of frames stacked

# MODIFY TO FALSE IF JUST WANT TO SEE THE TRAINED AGENT
training = False

# TURN THIS ON IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # Create placeholders
            # state_size means tht we take each element of state_size in tuple, like [None, 84,84,4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remeber that target_Q = R(s,a) + ymax Qhat(s',a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First covnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(
                inputs = self.inputs_,
                filters = 32,
                kernel_size = [8,8],
                strides = [4,4],
                padding = "VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name = "conv1"
            )

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second covnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4,4],
                strides=[2,2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2"
            )

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third covnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=64,
                kernel_size=[3,3],
                strides=[2,2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3"
            )

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            # Flatten result
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(
                inputs = self.flatten,
                units = 512,
                activation = tf.nn.elu,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                name = "fc1"
            )

            self.output = tf.layers.dense(
                inputs = self.fc,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                units = self.action_size,
                activation = None
            )

            # Q is our predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # Loss is the difference between our predicted Q_Values and the Q_Target
            # Sum(QTarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# Setup experience replay
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arrange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]

# Instantiate memory and provide some initial random values
memory = Memory(max_size=memory_size)
for i in range(pretrain_length):
    # If it's first step..
    if i == 0:
        state = env.reset()

        state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Get the next state the rewards, by taking a random action
    choice = random.randint(1, len(possible_actions))-1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)

    # env.render()

    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

    # If the episode is finished (dead x 3)
    if done:
        # Finish the episode
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        state = env.reset()

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our state is now the next state
        state = next_state

# Setup tensor board writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

# Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

