########################
# importing libraries
########################
# system libraries
import torch
import torch.nn as nn
import collections


class WRNBasicBlock(nn.Module):
    """
    Convolutional or transposed convolutional block consisting of multiple 3x3 convolutions with short-cuts,
    ReLU activation functions and batch normalization.
    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, in_planes, out_planes, stride, batchnorm=1e-5):
        super(WRNBasicBlock, self).__init__()

        self.batchnorm = batchnorm

        self.layer1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.useShortcut = ((in_planes == out_planes) and (stride == 1))
        if not self.useShortcut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        if not self.useShortcut:
            if self.batchnorm > 0.0:
                x = self.relu1(self.bn1(x))
            else:
                x = self.relu1(x)

        else:
            if self.batchnorm > 0.0:
                out = self.relu1(self.bn1(x))
            else:
                out = self.relu1(x)
        if self.batchnorm > 0.0:
            out = self.relu2(self.bn2(self.layer1(out if self.useShortcut else x)))
        else:
            out = self.relu2(self.layer1(out if self.useShortcut else x))
        out = self.conv2(out)
        return torch.add(x if self.useShortcut else self.shortcut(x), out)


class Net(nn.Module):
    """
    builds a network from a state_list

    Parameters:
        state_list (list): list of states in the state sequence
        num_classes (int): number of data classes
        net_input (batch of data): batch of data for setting values of batch size, number of colors and image size
        bn_val (float): epsilon value for batch normalization
        do_drop (float): weight drop probability for dropout

    Attributes:
        gpu_mem_req (float): estimated gpu memory requirement for the model by adding memory for storing model
            weights/biases and activations
        feature_extractor (torch.nn.Sequential): sequential container holding the layers in the feature extractor
                                                of a network
        classifier (torch.nn.Sequential): sequential container holding the layers in the classifier of a network
    """

    def __init__(self, state_list, num_classes, net_input, bn_val, do_drop):
        super(Net, self).__init__()
        batch_size = net_input.size(0)
        num_colors = net_input.size(1)
        image_size = net_input.size(2)

        # class attribute for storing total gpu memory requirement of the model
        # (4 bytes/ 32 bits per floating point no.)
        self.gpu_mem_req = 32 * batch_size * num_colors * image_size * image_size

        # lists for appending layer definitions
        feature_extractor_list = []
        classifier_list = []

        wrn_no = conv_no = fc_no = relu_no = bn_no = pool_no = 0
        out_channel = num_colors
        no_feature = num_colors * (image_size ** 2)

        # for pretty-printing
        print('*' * 80)

        for state_no, state in enumerate(state_list):
            # last layer is classifier (linear with sigmoid)
            if state_no == len(state_list)-1:
                break

            if state.layer_type == 'wrn':
                wrn_no += 1
                in_channel = out_channel
                out_channel = state.filter_depth
                no_feature = (state.image_size ** 2) * out_channel

                feature_extractor_list.append(('wrn_' + str(wrn_no), WRNBasicBlock(in_channel, out_channel,
                                                                                   stride=state.stride,
                                                                                   batchnorm=bn_val)))

                # gpu memory requirement for wrn block due to layer parameters (optional batchnorm parameters have been
                # ignored)
                self.gpu_mem_req += 32 * (3 * 3 * in_channel * out_channel + 3 * 3 * out_channel * out_channel +
                                          int(in_channel != out_channel) * in_channel * out_channel)

                # gpu memory requirement for wrn block due to layer feature output
                self.gpu_mem_req += 32 * batch_size * state.image_size * state.image_size * state.filter_depth\
                                    * (2 + int(in_channel != out_channel))

            elif state.layer_type == 'conv':
                conv_no += 1
                in_channel = out_channel
                out_channel = state.filter_depth
                no_feature = (state.image_size ** 2) * out_channel

                # conv filters without padding
                feature_extractor_list.append(('conv' + str(conv_no), nn.Conv2d(in_channel, out_channel,
                                                                                    state.filter_size,
                                                                                    stride=state.stride,
                                                                                    padding=state.padding,
                                                                                    bias=False)))

                if bn_val > 0.0:
                    bn_no += 1
                    feature_extractor_list.append(('batchnorm' + str(bn_no), nn.BatchNorm2d(num_features=out_channel,
                                                                                            eps=bn_val)))
                relu_no += 1
                feature_extractor_list.append(('relu' + str(relu_no), nn.ReLU(inplace=True)))

                # gpu memory requirement for conv layer due to layer parameters (batchnorm parameters have been
                # ignored)
                self.gpu_mem_req += 32 * in_channel * out_channel * state.filter_size * state.filter_size

                # gpu memory requirement for conv layer due to layer feature output
                self.gpu_mem_req += 32 * batch_size * state.image_size * state.image_size * state.filter_depth

            elif state.layer_type == 'max_pool':
                pool_no += 1
                in_channel = out_channel
                out_channel = state.filter_depth
                no_feature = (state.image_size ** 2) * out_channel

                # pool without padding
                feature_extractor_list.append(('max_pool' + str(pool_no), nn.MaxPool2d(state.filter_size,
                                                                                       stride=state.stride,
                                                                                       padding=0)))

                # gpu memory requirement for conv layer due to layer parameters (batchnorm parameters have been
                # ignored)
                self.gpu_mem_req += 32 * in_channel * out_channel * state.filter_size * state.filter_size

                # gpu memory requirement for conv layer due to layer feature output
                self.gpu_mem_req += 32 * batch_size * state.image_size * state.image_size * state.filter_depth

            elif state.layer_type == 'fc':
                # this code is not used by default, as the number of fully-connected layers is 0 per default!
                # this portion is not used for our DP-NAS experiments.
                # As mentioned in the README, we have left this code for the user to experiment with deeper classifiers
                # or to run a "conventional" fully-trained architecture search.

                fc_no += 1
                # the first time an fc layer appears after conv /wrn -> flatten input to get dimension
                # temp = torch.randn(batch_size, out_channel, last_image_size, last_image_size)
                in_feature = no_feature
                no_feature = state.fc_size

                classifier_list.append(('fc' + str(fc_no), nn.Linear(in_feature, no_feature, bias=False)))

                if bn_val > 0.0:
                    classifier_list.append(('batchnorm_fc' + str(fc_no), nn.BatchNorm1d(num_features=no_feature,
                                                                                    eps=bn_val)))
                classifier_list.append(('relu_fc' + str(fc_no), nn.ReLU(inplace=True)))

                # gpu memory requirement for FC layer due to layer parameters (batchnorm parameters have been ignored)
                self.gpu_mem_req += 32 * batch_size * no_feature

                # gpu memory requirement for FC layer due to layer feature output
                self.gpu_mem_req += 32 * in_feature * no_feature

        fc_no += 1

        if do_drop > 0.0:
            classifier_list.append(('dropout', nn.Dropout(do_drop, inplace=True)))

        # add linear classification layer to classifier
        classifier_list.append(('fc' + str(fc_no), nn.Linear(no_feature, num_classes, bias=False)))

        # gpu memory requirement for classifier layer due to layer parameters
        self.gpu_mem_req += 32 * no_feature * num_classes

        # gpu memory requirement for classifier layer due to layer output
        self.gpu_mem_req += 32 * batch_size * num_classes

        # converting bits to GB
        self.gpu_mem_req /= (8.*1024*1024*1024)

        self.feature_extractor = nn.Sequential(collections.OrderedDict(feature_extractor_list))
        self.classifier = nn.Sequential(collections.OrderedDict(classifier_list))

    def forward(self, x):
        """
        overloaded forward method to carry out the forward pass with the network
        
        Parameters:
            x (torch.Tensor): input to the network
        
        Returns:
            output (torch.Tensor) of the network
        """

        # reshape since the classifier will be linear and the feature_extractor is convolutional
        x = self.feature_extractor(x).view(x.size(0), -1)

        # DO NOT ADD THE CLASSIFIER HERE
        # the model encoder and classifier are treated separately in training, so we only use the encoder to
        # automatically forward here.
        return x
