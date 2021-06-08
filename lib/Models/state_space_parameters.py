"""
config file for defining search space rules and Q-learning hyperparameters
"""

########################
# search space rules
########################

# possible square conv kernel size
# kernel size from [3, 5] => stride = 1; conv size from [7, 9, 11] => stride = 2
conv_sizes = [1, 3, 5, 7, 9, 11]
conv_features = [16, 32, 64, 128, 256, 512, 1024]  # possible number of conv features
conv_strides = [1]

pool_sizes = [2, 3, 4]
pool_strides = [2, 3, 4]

max_num_fc_input = 20000  # number of units to use to check if conv to classifier transition is too large
max_image_size = 8  # constrain the maximum spatial dimension of the embedding, i.e. force architecture to downsample
# above option is reasonable when we consider that we want bottlenecks, in order to avoid overfitting identity mappings

fc_sizes = [32, 64, 128]  # possible FC layer sizes

################################
# q-learning hyperparameters
################################

# epsilon schedule for Q-learning
# Format : [[epsilon, # unique models]]
epsilon_schedule = [[1.0, 1500],
                    [0.9, 100],
                    [0.8, 100],
                    [0.7, 100],
                    [0.6, 100],
                    [0.5, 100],
                    [0.4, 100],
                    [0.3, 100],
                    [0.2, 100],
                    [0.1, 100],
                    [0.0, 3]]

replay_number = 50  # number of samples drawn from the replay buffer for Q-values update
