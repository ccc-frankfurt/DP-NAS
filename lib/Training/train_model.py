########################
# importing libraries
########################
# system libraries
import os
import math
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from shutil import rmtree
import time

# custom libraries
from lib.Models.network import Net
from lib.Utility.utils import GPUMem
from lib.Utility.utils import save_checkpoint
from lib.Utility.metrics import AverageMeter
import lib.Models.state_space_parameters as state_space_parameters


class Trainer:
    def __init__(self):
        # outputs of a net through the random (and fixed) part of the network,
        # which is not being trained
        self.train_loader_fixed_layer_outputs = None
        self.val_loader_fixed_layer_outputs = None
        self.test_loader_fixed_layer_outputs = None

    def train_val_net(self, state_list, dataset, weight_initializer, device, args, save_path):
        """
        builds a net given a state list, and trains and validates it

        Parameters:
            state_list (list): list of states to build the net
            dataset (lib.Datasets.datasets.CODEBRIM): dataset to train and validate the net on
            weight_initializer (lib.Models.initialization.WeightInit): class for initializing the weights of
                                                                       the network
            device (torch.device): type of computational device available (cpu / gpu)
            args (argparse.ArgumentParser): parsed command line arguments
            save_path (string): path for saving results to

        Returns:
            memfit (bool): True if the network fits the memory after batch splitting, False otherwise
            val_acc_all_epochs (list): list of validation accuracies in all epochs
            train_flag (bool): False if net's been early-stopped, False otherwise
        """

        # check the if the network in the state_list is feasible (if conv -> fc transition is not too big)
        max_num_fc_input = state_space_parameters.max_num_fc_input
        max_image_size = state_space_parameters.max_image_size

        for si, state in enumerate(state_list):
            # if transition from conv -> fc or it is the last layer
            if state_list[si - 1].as_tuple()[0] in ['conv', 'wrn', 'max_pool'] and \
                    (state.layer_type == 'fc' or state.terminate == 1) and \
                    state.filter_depth * state.image_size * state.image_size > max_num_fc_input:
                # mimic output for unsuitable architecture: fit memory, acc, acc_list, train flag
                # such that reward is 0
                return True, 0, [0], [0], 0, False

            # or if image_size bigger than 8x8
            if state_list[si - 1].as_tuple()[0] in ['conv', 'wrn', 'max_pool'] and \
                    (state.layer_type == 'fc' or state.terminate == 1) and \
                    state.image_size > max_image_size:
                # mimic output for unsuitable architecture: fit memory, acc, acc_list, train flag
                # such that reward is 0
                return True, 0, [0], [0], 0, False

        # remove temporary dataset folders if they exist
        if os.path.isdir(os.path.join(save_path, 'tmp_train')):
            rmtree(os.path.join(save_path, 'tmp_train'))
        if os.path.isdir(os.path.join(save_path, 'tmp_val')):
            rmtree(os.path.join(save_path, 'tmp_val'))
        if os.path.isdir(os.path.join(save_path, 'tmp_test')):
            rmtree(os.path.join(save_path, 'tmp_test'))

        # reset the data loaders
        dataset.train_loader, dataset.val_loader, dataset.test_loader = \
            dataset.get_dataset_loader(args.batch_size, args.workers, torch.cuda.is_available())
        net_input, _ = next(iter(dataset.train_loader))

        num_classes = dataset.num_classes
        batch_size = net_input.size(0)

        # gets number of available gpus and total gpu memory
        num_gpu = float(torch.cuda.device_count())
        gpu_mem = GPUMem(torch.device('cuda') == device)

        # builds the net from the state list
        model = Net(state_list, num_classes, net_input, args.batch_norm, args.drop_out_drop)

        print(model)

        # sets cudnn benchmark flag
        cudnn.benchmark = True

        # initializes weights
        weight_initializer.init_model(model)

        # puts model on gpu/cpu
        model = model.to(device)

        # gets available gpu memory
        if torch.device('cuda') == device:
            gpu_avail = (gpu_mem.total_mem - gpu_mem.total_mem * gpu_mem.get_mem_util()) / 1024.
            print('gpu memory available:{gpu_avail:.4f}'.format(gpu_avail=gpu_avail))

        # prints estimated gpu requirement of model but actual memory requirement is higher than what's estimated (from
        # experiments)
        print("model's estimated gpu memory requirement: {gpu_mem_req:.4f} GB".format(gpu_mem_req=model.gpu_mem_req))

        # scaling factor and buffer for matching expected memory requirement
        # with empirically observed memory requirement
        scale_factor = 4.0
        scale_buffer = 1.0
        if torch.device('cuda') == device:
            scaled_gpu_mem_req = (scale_factor / num_gpu) * model.gpu_mem_req + scale_buffer
            print("model's empirically scaled gpu memory requirement:"
                  " {scaled_gpu_mem_req:.4f}".format(scaled_gpu_mem_req=scaled_gpu_mem_req))
        split_batch_size = batch_size
        # splits batch into smaller batches
        if (torch.device('cuda') == device) and gpu_avail < scaled_gpu_mem_req:
            # estimates split batch size as per available gpu mem. (may not be a factor of original batch size)
            approx_split_batch_size = int(((gpu_avail - scale_buffer) * num_gpu / scale_factor) //
                                          (model.gpu_mem_req / float(batch_size)))

            diff = float('inf')
            temp_split_batch_size = 1
            # sets split batch size such that it's close to the estimated split batch size, is also a factor of original
            # batch size & should give a terminal batch size of more than 1
            for j in range(2, approx_split_batch_size + 1):
                if batch_size % j == 0 and abs(j - approx_split_batch_size) < diff and (len(dataset.trainset) % j > 1):
                    diff = abs(j - approx_split_batch_size)
                    temp_split_batch_size = j
            split_batch_size = temp_split_batch_size

        print('split batch size:{}'.format(split_batch_size))
        print('*' * 80)

        # returns memfit = False if model doesn't fit in memory even after splitting the batch size to as small as 1
        if split_batch_size < 2:
            return False, None, None, None, None, None

        # set the data loaders using the split batch size
        dataset.train_loader, dataset.val_loader, dataset.test_loader = \
            dataset.get_dataset_loader(split_batch_size, args.workers, torch.cuda.is_available())
        # use data parallelism for multi-gpu machine
        if torch.device('cuda') == device:
            model = torch.nn.DataParallel(model)

        if torch.device('cuda') == device:
            classifier = model.module.classifier
        else:
            classifier = model.classifier

        # cross entropy loss criterion (LogSoftmax and NLLoss together)
        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

        # optimizer
        if args.full_training:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        train_flag = True
        epoch = 0
        loss_val_all_epochs = []
        acc_val_all_epochs = []
        acc_train_all_epochs = []

        while epoch < args.epochs:
            # train and validate the model
            acc_train = self.train_epoch(dataset.train_loader, model, criterion, epoch,
                                         optimizer, device, args, split_batch_size, save_path)
            loss_val, acc_val = self.val_epoch(dataset.val_loader, model, criterion, device,
                                               is_val=True, precompute=not(args.full_training), save_path=save_path,
                                               args=args)

            if int(args.task) == 2 or epoch == args.epochs-1:
                # test the performance of selected networks
                # it can be with or without precomputing
                acc_test = self.val_epoch(dataset.test_loader, model, criterion, device,
                                          is_val=False, precompute=not(args.full_training), save_path=save_path,
                                          args=args)

            loss_val_all_epochs.append(loss_val)
            acc_val_all_epochs.append(acc_val)
            acc_train_all_epochs.append(acc_train)

            if int(args.task) == 2:
                # saves model dict while training fixed net
                state = {'epoch': epoch,
                         'arch': 'Fixed net: replay buffer - {}, index no - {}'.format(args.replay_buffer_csv_path,
                                                                                       args.fixed_net_index_no),
                         'state_dict': model.state_dict(),
                         'acc_val': acc_val,
                         'acc_test': acc_test,
                         'optimizer': optimizer.state_dict()
                         }
                save_checkpoint(state, max(acc_val_all_epochs) == acc_val,
                                os.path.join(save_path, str(args.fixed_net_index_no)))

            # checks for early stopping; early-stops if the mean of the validation accuracy from the last 3 epochs
            # before the early stopping epoch isn't at least as high as the early stopping threshold
            if epoch == (args.early_stopping_epoch - 1) and float(np.mean(acc_val_all_epochs[-5:])) <\
                    (args.early_stopping_thresh * 100.):
                train_flag = False
                acc_test = 0
                break

            epoch += 1
        acc_best_val = max(acc_val_all_epochs)

        # free up memory by deleting objects
        del model, criterion, optimizer, \
            self.train_loader_fixed_layer_outputs, self.val_loader_fixed_layer_outputs, \
            self.test_loader_fixed_layer_outputs

        self.train_loader_fixed_layer_outputs = None
        self.val_loader_fixed_layer_outputs = None
        self.test_loader_fixed_layer_outputs = None

        if os.path.isdir(os.path.join(save_path, 'tmp_train')):
            rmtree(os.path.join(save_path, 'tmp_train'))
        if os.path.isdir(os.path.join(save_path, 'tmp_val')):
            rmtree(os.path.join(save_path, 'tmp_val'))
        if os.path.isdir(os.path.join(save_path, 'tmp_test')):
            rmtree(os.path.join(save_path, 'tmp_test'))

        return True, acc_best_val, acc_val_all_epochs, acc_train_all_epochs, acc_test, train_flag

    # dataset_type is "train", "val" or "test"
    def preprocess_input(self, dataloader, model, device, dataset_type=None, save_path='./datasets',
                         store_embedding=False, args=None):
        print('preprocessing input')
        # store data
        store_path = os.path.join(save_path, 'tmp_' + dataset_type)

        # check if to save embedding and create a permanent path
        # only save embedding if task 2, as well as when either the architecture string or index given
        if store_embedding:
            # check conditions for embedding
            if args.task == 2 and (len(args.net) > 0 or int(args.fixed_net_index_no) > -1):
                print('storing embedding')
                store_embedding_path = os.path.join(os.path.join(save_path,'stored_embedding'),
                                                   dataset_type)
            else:
                raise ValueError('Embedding can be stored only for task 2 with specified architecture index or string')

        with torch.no_grad():
            count = 1
            for i, (input_, target) in enumerate(dataloader):
                input_ = input_.to(device)
                fixed_output = model(input_).detach().cpu()

                # loop through the batch and save it
                # keeping the whole dataset in CPU memory might lead to memory issues
                for j in range(fixed_output.size(0)):
                    img_path = store_path + '/' + format(target[j].item(), '06d')

                    try:
                        os.makedirs(img_path)
                    except OSError:
                        pass

                    if store_embedding:
                        img_path_permanent_embedding = store_embedding_path + '/' + format(target[j].item(), '06d')
                        try:
                            os.makedirs(img_path_permanent_embedding)
                        except OSError:
                            pass

                    np.save(img_path + '/' + format(count, '06d'), fixed_output[j].numpy())
                    if store_embedding:
                        np.save(img_path_permanent_embedding + '/' + format(count, '06d'), fixed_output[j].numpy())
                    count += 1

                del input_, fixed_output

        def npy_loader(path):
            sample = torch.from_numpy(np.load(path))
            return sample

        # load the stored data
        dataset_fixed_layer_outputs = torchvision.datasets.DatasetFolder(
            store_path, loader=npy_loader, extensions='.npy')

        return torch.utils.data.DataLoader(
                dataset_fixed_layer_outputs,
                batch_size=dataloader.batch_size, shuffle=True,
                num_workers=dataloader.num_workers, pin_memory=torch.device('cuda') == device)

    def train_epoch(self, train_loader, model, criterion, epoch, optimizer, device,
                    args, split_batch_size, save_path='./datasets'):
        """
        trains the model of a net for one epoch on the train set

        Parameters:
            train_loader (torch.utils.data.DataLoader): data loader for the train set
            model (lib.Models.network.Net): model of the net to be trained
            criterion (torch.nn.BCELoss): loss criterion to be optimized
            epoch (int): continuous epoch counter
            optimizer (torch.optim.SGD): optimizer instance like SGD or Adam
            device (torch.device): computational device (cpu or gpu)
            args (argparse.ArgumentParser): parsed command line arguments
            split_batch_size (int):  smaller batch size after splitting the original batch size for fitting the device
                                     memory
            save_path (str): the path to save the preprocessed randomly projected dataset to
        """
        # performance and computational overhead metrics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        factor = args.batch_size // split_batch_size
        last_batch = int(math.ceil(len(train_loader.dataset) / float(split_batch_size)))

        # switch to train mode
        model.train()

        optimizer.zero_grad()

        print('training')

        # if first epoch - save outputs of the dataset through the not-to-be-trained (fixed)
        # part of the network
        if not args.full_training:  # if not train from scratch
            if epoch == 0:
                self.train_loader_fixed_layer_outputs = self.preprocess_input(train_loader, model, device,
                                                                              'train', save_path,
                                                                              args.store_embedding, args)
            loader = self.train_loader_fixed_layer_outputs
        else:
            loader = train_loader

        if torch.device('cuda') == device:
            classifier = model.module.classifier
        else:
            classifier = model.classifier

        # compute the rest of the model
        correct_sum = 0
        total = 0

        for i, (input_, target) in enumerate(loader):
            # hacky way to deal with terminal batch-size of 1
            if target.size(0) == 1:
                print('skip last training batch of size 1')
                continue

            input_, target = input_.to(device), target.to(device)

            data_time.update(time.time() - end)

            if args.full_training:  # train from scratch
                output = classifier(model(input_))
            else:
                output = classifier(input_)

            # scale the loss by the ratio of the split batch size and the original
            loss = criterion(output, target) * input_.size(0) / float(args.batch_size)

            # update the 'losses' meter with the actual measure of the loss
            losses.update(loss.item() * args.batch_size / float(input_.size(0)), input_.size(0))

            # compute performance measures
            correct = np.sum(output.data.cpu().numpy().argmax(axis=1) == target.data.cpu().numpy().astype(int))
            correct_sum += correct
            total += target.shape[0]
            acc = (correct_sum / float(total))*100.

            loss.backward()

            # update the weights after every 'factor' times 'batch count' (after every batch had the batch size not been
            # split)
            if (i + 1) % factor == 0 or i == (last_batch - 1):
                optimizer.step()
                optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print performance and computational overhead measures after every 'factor' times 'batch count'
            # (after every batch had the batch size not been split)
            if i % (args.print_freq * factor) == 0:
                print('epoch: [{0}][{1}/{2}]\t'
                      'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      'acc {acc:.3f} '.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, losses=losses, acc=acc))

            del output, input_, target

        print(' * train: loss {losses.avg:.3f} acc {train_acc:.3f}'
              .format(losses=losses, train_acc=acc))
        print('*' * 80)

        del loader

        return acc

    def val_epoch(self, val_loader, model, criterion, device, is_val=True, precompute=False, save_path='./datasets', args=None):
        """
        validates the model of a net for one epoch on the validation set

        Parameters:
            val_loader (torch.utils.data.DataLoader): data loader for the validation set
            model (lib.Models.network.Net): model of a net that has to be validated
            criterion (torch.nn.BCELoss): loss criterion
            device (torch.device): device name where data is transferred to
            is_val (bool): validation or testing mode
            precompute (bool): flag whether to ompute + store randomly projected embedding (needs to be done only once)
            save_path (str): the path to save the preprocessed randomly projected dataset to

        Returns:
            losses.avg (float): average of the validation losses over the batches
            acc (float): average of the validation accuracy over the batches
        """
        # performance metrics
        losses = AverageMeter()

        # switch to evaluate mode
        model.eval()

        if is_val:
            print('validating')
        else:
            print('testing')

        if precompute:
            # pre-compute outputs for fixed layers
            # and compute for every iteration only the changing outputs
            if is_val:
                if self.val_loader_fixed_layer_outputs is None:  # first epoch
                    self.val_loader_fixed_layer_outputs = self.preprocess_input(val_loader, model, device,
                                                                                'val', save_path,
                                                                                args.store_embedding, args)
                loader = self.val_loader_fixed_layer_outputs
            else:
                if self.test_loader_fixed_layer_outputs is None:  # first epoch
                    self.test_loader_fixed_layer_outputs = self.preprocess_input(val_loader, model, device,
                                                                                 'test', save_path,
                                                                                 args.store_embedding, args)
                loader = self.test_loader_fixed_layer_outputs
        else:
            loader = val_loader

        if torch.device('cuda') == device:
            classifier = model.module.classifier
        else:
            classifier = model.classifier

        correct_sum = 0
        total = 0

        #  to ensure no buffering for gradient updates
        with torch.no_grad():
            for i, (input_, target) in enumerate(loader):
                if input_.size(0) == 1:
                    # hacky way to deal with terminal batch-size of 1
                    print('skip last val/test batch of size 1')
                    continue
                input_, target = input_.to(device), target.to(device)

                if not precompute:  # usually task 2 -test
                    output = classifier(model(input_))
                else:
                    output = classifier(input_)

                loss = criterion(output, target)

                # update the 'losses' meter
                losses.update(loss.item(), input_.size(0))

                # compute performance measures
                correct = np.sum(output.data.cpu().numpy().argmax(axis=1) == target.data.cpu().numpy().astype(int))
                correct_sum += correct
                total += target.shape[0]
                acc = (correct_sum / float(total)) * 100.

                del output, input_, target
        del loader

        if is_val:
            print(' * val: loss {losses.avg:.3f}, acc {acc:.3f} \t'
                  .format(losses=losses, acc=acc))
            print('*' * 80)
        else:
            print(' * test: loss {losses.avg:.3f}, acc {acc:.3f} \t'
                  .format(losses=losses, acc=acc))
            print('*' * 80)

        if is_val:
            return losses.avg, acc
        else:
            return acc
