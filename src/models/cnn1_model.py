from torch import nn, flatten, sigmoid

class CNN1(nn.Module):
    def __init__(
        self, 
        input_size, in_channels, out_channels1, out_channels2, 
        filter_size1, filter_size2, maxpool_kernel_size1, maxpool_kernel_size2,
        intermediate_layer_size):
        super().__init__()
        self.padding1 = input_size%filter_size1
        self.l1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels1, filter_size1, padding=self.padding1),
            nn.MaxPool1d(maxpool_kernel_size1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.output_width_layer1 = int((input_size + 2*self.padding1 - filter_size1 + 1)/maxpool_kernel_size1)
        self.padding2 = self.output_width_layer1%filter_size2
        self.l2 = nn.Sequential(
            nn.Conv1d(out_channels1, out_channels2, filter_size2, padding=self.padding2),
            nn.MaxPool1d(maxpool_kernel_size2),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.output_width_layer2 = int((self.output_width_layer1 + 2*self.padding2 - filter_size2 + 1)/maxpool_kernel_size2)

        self.linear_layer1 = nn.Linear(self.output_width_layer2*out_channels2,intermediate_layer_size,bias=False)
        
        self.linear_layer2 = nn.Linear(intermediate_layer_size,1,bias=False)
        self.l_out = nn.Sequential(
            self.linear_layer1,
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_layer_size),
            nn.Dropout(p=0.5),
            self.linear_layer2,
            nn.ReLU()
        )
          

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[3], x.shape[2])
        #print('Init:', x.shape)
        x = self.l1(x)
        #print('After l1:', x.shape)
        x = self.l2(x)
        #print('After l2:', x.shape)
        x = flatten(x, start_dim=1)
        #print('After flatten:', x.shape)
        x = self.l_out(x)
        return sigmoid(x)