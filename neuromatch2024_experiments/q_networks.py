import torch
import torch.nn as nn


class RNNQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNQNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # print(hidden)
        out, hidden = self.rnn(x, hidden)
        q_values = self.fc(out)
        return q_values, hidden

    def init_hidden(self):
        return torch.ones(1, 1, self.hidden_size)


class GRUQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUQNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        q_values = self.fc(out)
        return q_values, hidden

    def init_hidden(self):
        return torch.ones(1, 1, self.hidden_size)


class LSTMQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMQNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        q_values = self.fc(out)
        return q_values, hidden

    def init_hidden(self):
        # Initialize both the hidden state and the cell state
        return (torch.ones(1, 1, self.hidden_size),
                torch.ones(1, 1, self.hidden_size))
