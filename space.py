import tensorflow as tf  # Deep learning library - need v 1.10
import numpy as np  # Handle matrices
import retro
from skimage import transform  # Image pre-processing
from skimage.color import rgb2gray  # Gray out images
import matplotlib.pyplot as plt  # For drawing graphs
from collections import deque  # Ordered collections
import random
import warnings  # Allows us to ignore warnings from sk-image

# Ignore sci-kit warnings
warnings.filterwarnings('ignore')

# Create Gym Retro environment
env = retro.make(game='SpaceInvaders-Atari2600')

# For info,print size of frame and observation space
print("Size of frame: ", env.observation_space)
print("Size of action space: ", env.action_space.n)

# Get list of actions - creates a 2D array, the size of the action space squared, with 1s on the main diagonal
# I.e. a table for all inputs (n), with only one active in each row
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())

# Pre-process frames - take a frame, grayscale it, resize it, normalise it, return it
def preprocess_frame(frame):

    print("Pre-processing frame")

    # Grayscale
    gray = rgb2gray(frame)

    # Crop the screen (remove bit below player)
    # [Up: down, Left:right]
    cropped_frame = gray[8:-12, 4:-12]

    # Normalise pixel values
    normalized_frame = cropped_frame / 255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    # A 110 x 84 image with 1 layer (gray)
    return preprocessed_frame


stack_size = 4  # Need a stack of 4

# Initialise deque with 4 blank images (all zeros)
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frame, game_state, is_new_episode):

    print("Stacking frames")

    # Pre-process frames
    frame = preprocess_frame(game_state)

    if is_new_episode:
        # Clear the stacked frames
        stacked_frame = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # As it's a new episode, copy the first frame four times
        stacked_frame.append(frame)
        stacked_frame.append(frame)
        stacked_frame.append(frame)
        stacked_frame.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frame, axis=2)

    else:
        # Append the frame to deque, auto removes the oldest
        stacked_frame.append(frame)

        # Build the stacked state
        stacked_state = np.stack(stacked_frame, axis=2)

    return stacked_state, stacked_frame


# Model hyperparameters
state_size = [110, 84, 4]  # 4 images of 110x84 dimension
action_size = env.action_space.n  # 8 possible actions
learning_rate = 0.00025  # Alpha (learning rate)

# Training hyperparameters
total_episodes = 50  # Total episodes for training
max_steps = 50000  # Max steps in an episode
batch_size = 64  # Batch size

# Exploration params for epsilon greedy strategy
explore_start = 1.0  # Exploration probability at start
explore_stop = 0.01  # Minimum exploration rate
decay_rate = 0.00001  # Exponential decay rate for exploration probability

# Q learning hyper parameters
gamma = 0.9  # Discounting rate

# Memory hyper parameters
pretrain_length = batch_size  # Number of experiences stored in the memory when initialised for first time
memory_size = 1000000  # Number of experiences the memory can keep

# Preprocessing hyper parameters
stack_size = 4  # Number of frames stacked

# MODIFY TO FALSE IF JUST WANT TO SEE THE TRAINED AGENT
training = True
firstTrain = True

# TURN THIS ON IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True


class DQNetwork:
    def __init__(self, dq_state_size, dq_action_size, dq_learning_rate, name='DQNetwork'):
        self.state_size = dq_state_size
        self.action_size = dq_action_size
        self.learning_rate = dq_learning_rate

        with tf.variable_scope(name):
            # Create placeholders
            # state_size means tht we take each element of state_size in tuple, like [None, 84,84,4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *dq_state_size], name="inputs")
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
                inputs=self.inputs_,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1"
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
                kernel_size=[4, 4],
                strides=[2, 2],
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
                kernel_size=[3, 3],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3"
            )

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            # Flatten result
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1"
            )

            self.output = tf.layers.dense(
                inputs=self.fc,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=self.action_size,
                activation=None
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
# Removed for now to simplify process
"""
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]
"""

# Instantiate memory and provide some initial random values
# Removed for now
# memory = Memory(max_size=memory_size)

# Remove pre-training - meant to initialise training buffer with data
"""
for i in range(pretrain_length):
    # If it's first step..
    if i == 0:
        state = env.reset()

        state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Get the next state the rewards, by taking a random action
    choice = random.randint(1, len(possible_actions)) - 1
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
        # Removed for now
        # memory.add((state, action, reward, next_state, done))

        # Start a new episode
        state = env.reset()

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Add experience to memory
        # Removed for now
        # memory.add((state, action, reward, next_state, done))

        # Our state is now the next state
        state = next_state
"""

# Setup TensorBoard writer - output destination
writer = tf.summary.FileWriter("./tensorboard/rl-test/1")

# Configure TF to record losses for TensorBoard analysis
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

"""
Training Agent
"""


# Function to choose random or best known action
def predict_action(p_explore_start, p_explore_stop, decy_rate, p_decay_step, p_state, actions):

    print("Predicting action")

    # Pick a number for epsilon greedy
    exp_exp_tradeoff = np.random.rand()

    # Epsilon greedy strategy
    explore_probability = p_explore_stop + (p_explore_start - p_explore_stop) * np.exp(-decay_rate * p_decay_step)

    if explore_probability > exp_exp_tradeoff:
        # Make a random action (explore)
        print("Selecting random action")
        p_choice = random.randint(1, len(possible_actions)) - 1
        p_action = possible_actions[p_choice]

    else:
        print("Getting best action")
        # Get action from Q network (exploit)
        # Estimate Qs value state

        # sess.run runs a TF calculation. sess.run(fetches, feed_dict)
        # fetches = Tensor to calculate on
        # feed_dict = remaps values in the tensor, for each given key
        # So we're reusing output from last calc, and giving new inputs based on current state
        p_qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: p_state.reshape((1, *p_state.shape))})

        # Take the biggest Q Value
        # Returns list of score for each action - get index of highest scoring action
        p_choice = np.argmax(p_qs)

        # Select action, using available actions and best scoring index
        p_action = possible_actions[p_choice]

    return p_action, explore_probability


# Saver will help us to save the model
saver = tf.train.Saver()

if training:
    with tf.Session() as sess:

        # Initialise the variables
        if firstTrain:
            sess.run(tf.global_variables_initializer())
            print("First training run - creating new model")
        else:
            try:
                saver.restore(sess, "./models/model.ckpt")
                print("Loaded existing training model")
            except:
                print("Error loading existing training model - creating new")
                sess.run(tf.global_variables_initializer())

        # Initialise the decay rate to reduce epsilon
        decay_step = 0

        for episode in range(total_episodes):
            print("Training - episode: {}".format(episode))
            # set step to 0
            step = 0

            # Initialise rewards for te episode
            episode_rewards = []

            # make a new episode and observe first state
            state = env.reset()

            # Remember that stack frame function also calls preprocess function
            state, stacked_frames = stack_frames(stacked_frames, state, True)


            while step < max_steps:

                #(state, action, reward, next_state, done)
                step_state = (0, 0, 0, 0, 0)

                step += 1

                # if step % 50 == 0:
                print("Step {}".format(step))

                # Increase decay step
                decay_step += 1

                # Predict the action to take and take it
                action, explore_probablity = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                            possible_actions)

                # Perform the action and get nex state, reward and done info
                next_state, reward, done, _ = env.step(action)

                if episode_render:
                    env.render()

                # Add the reward to total reward
                episode_rewards.append(reward)

                # if the game is finished
                if done:
                    # Episode ends so no next state
                    next_state = np.zeros((110, 84), dtype=np.int)

                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    """
                    print("Episode: {}".format(episode), "Total reward: {}".format(total_reward),
                          "Explore P: {:.4f}".format(explore_probablity), "Training Loss: {:.4f}".format(loss))
                    """

                    # rewards_list.append((episode, total_reward))

                    # store transition <st,at,rt+1,st+1> in memory
                    # removed
                    # memory.add((state, action, reward, next_state, done))

                    # (state, action, reward, next_state, done) - store for training
                    step_state = (state, action, reward, next_state, done)

                else:
                    # Stack the fame of the next state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Add experience to memory
                    # removed
                    # memory.add((state, action, reward, next_state, done))

                    # (state, action, reward, next_state, done)
                    step_state = (state, action, reward, next_state, done)

                    # st+1 is now current state
                    state = next_state

                # Learning Part

                """
                # Get an experience tuple from memory - (state, action, reward, next_state, done)
                batch = memory.sample(batch_size) # batch_size == 64
                
                # states_mb is np.array, of each batches state, three dimensions (x,y of state, z for pile of states)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                                
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])
                """

                # Need to reshape into NP arrays for TF input
                l_state = step_state[0].reshape(1, *state_size)
                l_action = step_state[1].reshape(1, action_size)
                l_reward = step_state[2]
                l_next_state = step_state[3] # ndmin = minimum number of dimensions
                l_done = step_state[4]

                target_Qs_batch = 0

                # Get Q values for next state
                # Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: l_next_state})
                Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: l_state})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gammma * maxQ(s',a')


                terminal = l_done

                # if we are in a terminal state, only equals reward
                if terminal:
                    # target_Qs_batch.append(rewards_mb[i])
                    target_Qs_batch = l_reward
                else:
                    target = l_reward + gamma * np.max(Qs_next_state)
                    target_Qs_batch = target

                # Reshape to np array of 1
                targets_mb = np.array(target_Qs_batch).reshape(1)

                
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                   feed_dict={DQNetwork.inputs_: l_state,
                                              DQNetwork.target_Q: targets_mb,
                                              DQNetwork.actions_: l_action})

                # Write TF summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: l_state,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: l_action})
                
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                   feed_dict={DQNetwork.inputs_: l_state,
                                              DQNetwork.target_Q: targets_mb,
                                              DQNetwork.actions_: l_action})

                # Write TF summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: -l_state,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: l_action})

                writer.add_summary(summary, episode)
                writer.flush()

            # Save updated model
            save_path = saver.save(sess, "./models/model.ckpt")
            print("Model Saved")
            """

with tf.Session() as sess:
    total_test_rewards = []

    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(10):
        total_rewards = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))
            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action)
            env.render()

            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state

    env.close()
    """
    # force git
