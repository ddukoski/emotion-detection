import torch.nn as nn

class EmotionCNN(nn.Module):

    # chosen model architecture
    architecture = [32, 32, 'pool', 64, 64, 'pool', 128, 128, 'pool']

    # todo: if slow convergence/overfit try ELU activation.

    def __init__(self, num_of_channels=1, num_of_classes=7):
        super(EmotionCNN, self).__init__()
        self.features = self.create_layers(num_of_channels)
        self.classifier = nn.Sequential(
                                        nn.Linear(6 * 6 * 128, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(64, num_of_classes)
                                        )

    def forward(self, x):
        """
        The code below is executed for every layer in the network
        :param x: Input to the layer
        :return: Output scores from the layer
        """
        output = self.features(x) # pass image through network
        output = output.view(output.size(0), -1)  # flatten input
        output = self.classifier(output)  # classify

        return output

    def create_layers(self, in_channels):
        """
        :param in_channels: number of input channels (always 1 in our case, but we use this argument for modularity)
        :return: A Sequential container of layers (torch layer Modules)
        """

        layers = list()

        for x in self.architecture:
            if x == 'pool':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                           nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)
                          ]

                in_channels = x

        return nn.Sequential(*layers)
