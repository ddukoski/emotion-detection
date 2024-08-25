import torch.nn as nn
import torch.functional as f

class EmotionCNN(nn.Module):

    # chosen model architecture
    architecture = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

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
        :return: Output from the layer
        """
        output = self.features(x)
        output = output.view(output.size(0), -1)
        output = f.dropout(output, p=0.5, training=True)

        output = self.classifier(output)  # pass through each layer

        return output

    def create_layers(self, in_channels):
        """
        :param in_channels: number of input channels (always 1 in our case, but we use this argument for modularity)
        :return: A Sequential container of layers (torch.nn Modules)
        """

        layers = list()

        for x in self.architecture:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                           nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)
                          ]

                in_channels = x

        return nn.Sequential(*layers)
