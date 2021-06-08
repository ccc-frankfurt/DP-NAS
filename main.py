########################
# importing libraries
########################
# system libraries
import os
import numpy as np
from time import gmtime, strftime
import torch

# custom libraries
from lib.cmdparser import parser
import lib.Datasets.datasets as datasets
from lib.MetaQNN.q_learner import QLearner as QLearner
import lib.Models.state_space_parameters as state_space_parameters
from lib.Models.initialization import WeightInit
from lib.Training.train_model import Trainer
from lib.Utility.visualize import visualize_rolling_reward
from lib.Utility.utils import get_best_architectures


def main():
    # set device for torch computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = './runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # parse command line arguments
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # create log file
    log_file = os.path.join(save_path, "stdout")

    # write parsed args to log file
    log = open(log_file, "a")
    for arg in vars(args):
        print(arg, getattr(args, arg))
        log.write(arg + ':' + str(getattr(args, arg)) + '\n')
    log.close()

    # instantiate the weight initializer
    print("Initializing network with: " + args.weight_init)
    weight_initializer = WeightInit(args.weight_init)

    # instantiate dataset object
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)

    # instantiate a tabular Q-learner
    q_learner = QLearner(args, dataset.num_classes, save_path)

    trainer = Trainer()

    # start new architecture search
    if int(args.task) == 1:
        if args.continue_search is True:
            # raise exceptions if requirements to start new search not met
            if args.continue_epsilon not in np.array(state_space_parameters.epsilon_schedule)[:, 0]:
                raise ValueError('continue-epsilon {} not in epsilon schedule!'.format(args.continue_epsilon))
            if (args.replay_buffer_csv_path is None) or (not os.path.exists(args.replay_buffer_csv_path)):
                raise ValueError('specify correct path to replay buffer to continue ')
            if (args.q_values_csv_path is None) or (not os.path.exists(args.q_values_csv_path)):
                raise ValueError('wrong path is specified for Q-values')

        # iterate as per the epsilon-greedy schedule
        counter = 0
        for episode in state_space_parameters.epsilon_schedule:
            epsilon = episode[0]
            m = episode[1]

            # raise exception if net number to continue from greater than number of nets for the continue_epsilon
            if epsilon == args.continue_epsilon and args.continue_ite > m:
                raise ValueError('continue-ite {} not within range of continue-epsilon {} in epsilon schedule!'
                                 .format(args.continue_ite, epsilon))

            # iterate through number of nets for an epsilon
            ite = 1
            while ite < m + 1:
                # check conditions to generate and train arc
                if (epsilon == args.continue_epsilon and ite >= args.continue_ite) or (epsilon < args.continue_epsilon):
                    print('ite:{}, epsilon:{}'.format(ite, epsilon))

                    # generate net states for search
                    q_learner.generate_search_net_states(epsilon)

                    # check if net already trained before
                    search_net_in_replay_dict = q_learner.check_search_net_in_replay_buffer()

                    reward = -1
                    # add to the end of the replay buffer if net already trained before
                    if search_net_in_replay_dict:
                        reward = q_learner.add_search_net_to_replay_buffer(search_net_in_replay_dict, verbose=True)
                    # train net if net not trained before
                    else:
                        # train/val search net
                        mem_fit, acc_best_val, acc_val_all_epochs, acc_train_all_epochs, acc_test, train_flag =\
                            trainer.train_val_net(q_learner.state_list, dataset, weight_initializer, device, args, save_path)

                        # check if net fits memory
                        while mem_fit is False:
                            print("net failed mem check even with batch splitting, sampling again!")

                            q_learner.generate_search_net_states(epsilon)
                            net_in_replay_dict = q_learner.check_search_net_in_replay_buffer()

                            if search_net_in_replay_dict:
                                q_learner.add_search_net_to_replay_buffer(net_in_replay_dict)
                                break
                            else:
                                mem_fit, acc_best_val, acc_val_all_epochs, acc_train_all_epochs, acc_test, train_flag =\
                                    trainer.train_val_net(q_learner.state_list, dataset, weight_initializer, device, args,
                                                  save_path)

                        # add new net and performance measures to replay buffer if it fits in memory after splitting
                        # batch
                        if mem_fit:
                            reward = q_learner.accuracies_to_reward(acc_val_all_epochs)
                            q_learner.add_search_net_to_replay_buffer(search_net_in_replay_dict,
                                                                      reward=reward, acc_best_val=acc_best_val,
                                                                      acc_val_all_epochs=acc_val_all_epochs,
                                                                      acc_train_all_epochs=acc_train_all_epochs,
                                                                      acc_test=acc_test,
                                                                      train_flag=train_flag, verbose=True)
                    # sample nets from replay buffer, update Q-values and save partially filled replay buffer and
                    # Q-values
                    q_learner.update_q_values_and_save_partial()

                    # visualize rolling rewards
                    counter += 1
                    visualize_rolling_reward(replay_dict=q_learner.replay_buffer, save_path=save_path,
                                             num_arch=q_learner.arc_count)

                    # update counter only if the sampled architecture is not too big
                    if reward > 0:
                        ite += 1
                    elif reward == -1:
                        raise Exception('Reward calculation went wrong, please check')
                else:
                    ite += 1

        # save fully filled replay buffer and final Q-values
        q_learner.save_final()

    # load architecture config from replay buffer and train till convergence (if full training is True)
    # or try it out with random weights (if full_training is False)
    elif int(args.task) == 2:
        # if an architecture string has been specified, always use it (independent of a csv file being specified)
        if len(args.net) > 0:
            num_arcs = 1
            state_string = args.net
        elif (args.replay_buffer_csv_path is not None) and (os.path.exists(args.replay_buffer_csv_path)):
            if int(args.fixed_net_index_no) < 0:
                arch_indices = get_best_architectures(args.replay_buffer_csv_path, save_path=save_path)
            else:  # else train only the chosen one
                arch_indices = [args.fixed_net_index_no]
            num_arcs = len(arch_indices)
        else:
            raise ValueError('wrong path specified for replay buffer, or no architecture given to train')

        for i in range(num_arcs):
            if len(args.net) > 0:
                # generate net according to string
                q_learner.generate_fixed_net_states_from_string(state_string)
            else:
                # generate states for a net id from a complete search
                q_learner.generate_fixed_net_states(arch_indices[i])

            # train/val fixed net exhaustively
            mem_fit, acc_best_val, acc_val_all_epochs, acc_train_all_epochs, acc_test, train_flag = \
                trainer.train_val_net(q_learner.state_list, dataset, weight_initializer, device, args, save_path)

            # add fixed net and performance measures to a data frame and save it
            q_learner.add_fixed_net_to_fixed_net_buffer(acc_best_val=acc_best_val,
                                                        acc_val_all_epochs=acc_val_all_epochs,
                                                        acc_train_all_epochs=acc_train_all_epochs,
                                                        acc_test=acc_test)

        # save fixed net buffer
        q_learner.save_final()

    # raise exception if no matching task
    else:
        raise NotImplementedError('Given task number not implemented.')


if __name__ == '__main__':
    main()
