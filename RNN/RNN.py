import os
import torch
import torch.nn as nn

class RNN_Cell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNN_Cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.U = nn.Linear(input_size, hidden_size, bias=True)  # input to hidden (I, H), bias:b1
        self.W = nn.Linear(hidden_size, hidden_size, bias=True) # hidden to hidden (H, H), bias:b2
    
    def forward(self, x_t, h_prev):
        # x_t (input of an RNN Cell)  : (Batch_size, Input_size)
        # h_t (hidden of an RNN Cell) : (Batch_size, Hidden_size)

        a_t = self.U(x_t) + self.W(h_prev)  # x_t * U + h_(t-1) * U + b : (B, H)
        h_t = torch.tanh(a_t)               # tanh{x_t * U + h_(t-1) * W + b} : (B, H)

        return h_t # hidden of an RNN Cell


class RNN_Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_first=False):
        assert batch_first==True, "batch_first 값이 True일 때만 동작"
        
        super(RNN_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = RNN_Cell(input_size, hidden_size)
        self.rnn_layers = nn.ModuleList([self.rnn] + [RNN_Cell(hidden_size, hidden_size) for _ in range(self.num_layers - 1)]).to(device)

        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)  # (H, H)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True) # (H, H)
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=True) # (H, O)

    def forward(self, x):
        # Assume that Shape of x is batch_first=True : (B, L, I)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Initialize hidden state of each layers
        hiddens = [torch.zeros(batch_size, self.hidden_size).to(device) for i in range(self.num_layers)]
              
        # Reshape Input Size to make batch_first=True to "batch_first=False"
        x = x.transpose(0, 1) # (B, L, I) -> (L, B, I)
    
        # Propagate input through RNN
        # Input at t   : (Batch_size, Input_size)
        # Input        : (Seq_length, Batch_size, Input_size)
        # Hiddens      : (Num_layers, Batch_size, Hidden_size)
        # Hidden at t  : (Batch_size, Hidden_size)
        # Final Hidden : (Seq_length, Batch_size, Hidden_size)
        # Output       : (Seq_length, Batch_size, Hidden_size)

        final_hidden_states = []
        for t in range(seq_len):
            input = x[t, :, :]  
            for layer_idx, rnn_layer in enumerate(self.rnn_layers):
                hiddens[layer_idx] = rnn_layer(input, hiddens[layer_idx])
                input = hiddens[layer_idx] # (B, H)
            final_hidden_states.append(hiddens[-1])
            
        # list to tensor
        output = torch.stack(final_hidden_states, dim=0) # (L, B, H)
        
        output = self.drop1(self.fc1(output))   # (L, B, H) -> (L, B, H)
        output = self.drop2(self.fc2(output))   # (L, B, H) -> (L, B, H)
        output = self.fc3(output)               # (L, B, H) -> (L, B, O)
        
        # Reshape Output Size for batch_first=True
        output_sequnce = output.transpose(0, 1) # (L, B, O) -> (B, L, O)
           
        return output_sequence # (B, L, O)
    
    
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= "5"  # Set the GPU 5 to use

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    epochs = 40
    learning_rate = 0.01 # or use 1e-3

    input_size = 1
    hidden_size = 128
    output_size = 1
    seq_len = 24
    num_layers = 4

    # Instantiate RNN model
    model = RNN_Model(input_size, hidden_size, output_size, num_layers, batch_first=True).to(device)
    print(model)
    
    """
    RNN_Model(
      (rnn): RNN_Cell(
        (U): Linear(in_features=1, out_features=128, bias=True)
        (W): Linear(in_features=128, out_features=128, bias=True)
      )
      (rnn_layers): ModuleList(
        (0): RNN_Cell(
          (U): Linear(in_features=1, out_features=128, bias=True)
          (W): Linear(in_features=128, out_features=128, bias=True)
        )
        (1): RNN_Cell(
          (U): Linear(in_features=128, out_features=128, bias=True)
          (W): Linear(in_features=128, out_features=128, bias=True)
        )
        (2): RNN_Cell(
          (U): Linear(in_features=128, out_features=128, bias=True)
          (W): Linear(in_features=128, out_features=128, bias=True)
        )
        (3): RNN_Cell(
          (U): Linear(in_features=128, out_features=128, bias=True)
          (W): Linear(in_features=128, out_features=128, bias=True)
        )
      )
      (fc1): Linear(in_features=128, out_features=128, bias=True)
      (drop1): Dropout(p=0.25, inplace=False)
      (fc2): Linear(in_features=128, out_features=128, bias=True)
      (drop2): Dropout(p=0.25, inplace=False)
      (fc3): Linear(in_features=128, out_features=1, bias=True)
    )
    """