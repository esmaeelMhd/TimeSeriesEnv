"""
LSTM and Encoder-Decoder LSTM Models for Sequence to Sequence Prediction

This script defines LSTM and Encoder-Decoder LSTM models for sequence to sequence modeling tasks using PyTorch.
It includes the LSTMModel, EncoderLSTM, DecoderLSTM, and Net_LSTM classes for constructing and training these models.
The models take an input sequence of shape [batch_size, sequence_length, n_features] 
and produce an output sequence of shape [batch_size, prediction_length, n_features].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === lstm ====================================================================
class LSTMModel(nn.Module):
    """
    LSTM model for sequence-to-sequence prediction tasks.

    This model is designed to predict a sequence of values based on input sequences. It utilizes LSTM layers 
    to capture temporal dependencies and a fully connected layer to produce the final output.
    """

    def __init__(self, seq_len=None, in_features=None, out_features=None, pred_len=None, hidden_dim=None, layer_dim=None, dropout=None):
        """
        Initializes the LSTMModel class.

        Args:
            seq_len (int): Length of the input sequences.
            in_features (int): Number of input features per time step.
            out_features (int): Number of output features per time step.
            pred_len (int): Length of the predicted sequences.
            hidden_dim (int): Number of features in the hidden state of the LSTM.
            layer_dim (int): Number of recurrent layers in the LSTM.
            dropout (float): Dropout probability for LSTM layers (only applied between LSTM layers).
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.out_features = out_features
        self.pred_len = pred_len

        # LSTM layer(s)
        self.lstm = nn.LSTM(
            input_size=in_features, 
            hidden_size=hidden_dim, 
            num_layers=layer_dim, 
            dropout=dropout if layer_dim > 1 else 0.0,  # Only apply dropout if there is more than one layer
            batch_first=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(
            in_features=seq_len * hidden_dim, 
            out_features=pred_len * out_features,  # Adjust the output shape for multiple predictions
            bias=True
        )
        
    def forward(self, x, h0=None, c0=None):
        """
        Performs a forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_features).
            h0 (torch.Tensor, optional): Initial hidden state of shape (layer_dim, batch_size, hidden_dim). Defaults to None.
            c0 (torch.Tensor, optional): Initial cell state of shape (layer_dim, batch_size, hidden_dim). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, pred_len, out_features).
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states if not provided, or if batch size changes
        if h0 is None or c0 is None or h0.size(1) != batch_size or c0.size(1) != batch_size:
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
        
        # Flatten parameters for data-parallel efficiency
        self.lstm.flatten_parameters()
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Reshape output for the fully connected layer
        out = out.reshape(batch_size, -1)
        
        # Fully connected layer
        out = self.fc(out)
        
        # Reshape to the desired output shape
        out = out.view(batch_size, self.pred_len, self.out_features)
                
        return out


# === encoder-decoder lstm ====================================================
class EncoderLSTM(nn.Module):
    """
    Encoder LSTM model for sequence to sequence prediction.
    """

    def __init__(self, args):
        """
        Initialize the EncoderLSTM class.

        Args:
            args: Arguments containing model configurations and settings.
        """
        super(EncoderLSTM, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(
            input_size=self.args.in_features, 
            hidden_size=self.args.hidden_dim, 
            num_layers=self.args.layer_dim, 
            dropout=self.args.dropout,
            batch_first=True,
        )

    def forward(self, input, hidden):
        """
        Forward pass through the Encoder LSTM.

        Args:
            input (torch.Tensor): Input tensor of the shape (batch_size, length T, dimensionality d)
            hidden (tuple): Tuple containing hidden and cell state.

        Returns:
            output (torch.Tensor): Output tensor.
            hidden (tuple): Updated hidden and cell state.
        """
        output, hidden = self.lstm(input, hidden)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden and cell states with zeros.

        Args:
            batch_size: Size of the batch.
            device: Device for tensor allocation.

        Returns:
            hidden (tuple): Initialized hidden and cell state.
        """
        return (torch.zeros(self.args.layer_dim, batch_size, self.args.hidden_dim, device=device),
                torch.zeros(self.args.layer_dim, batch_size, self.args.hidden_dim, device=device))

class DecoderLSTM(nn.Module):
    """
    Decoder LSTM model for sequence to sequence prediction.
    """

    def __init__(self, args):
        """
        Initialize the DecoderLSTM class.

        Args:
            args: Arguments containing model configurations and settings.
        """
        super(DecoderLSTM, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(
            input_size=self.args.in_features, 
            hidden_size=self.args.hidden_dim, 
            num_layers=self.args.layer_dim, 
            dropout=self.args.dropout,
            batch_first=True,
        )
        
        self.fc = nn.Linear(self.args.hidden_dim, self.args.d_ff)
        self.out = nn.Linear(self.args.d_ff, self.args.in_features)

    def forward(self, input, hidden):
        """
        Forward pass through the Decoder LSTM.

        Args:
            input (torch.Tensor): Input tensor.
            hidden (tuple): Tuple containing hidden and cell state.

        Returns:
            output (torch.Tensor): Output tensor.
            hidden (tuple): Updated hidden and cell state.
        """
        output, hidden = self.lstm(input, hidden)
        output = F.relu(self.fc(output))
        output = self.out(output)
        return output, hidden

class Net_LSTM(nn.Module):
    """
    Encoder-Decoder LSTM network for sequence to sequence prediction.
    """

    def __init__(self, encoder, decoder, args, device):
        """
        Initialize the Net_LSTM class.

        Args:
            encoder: Encoder LSTM model.
            decoder: Decoder LSTM model.
            args: Arguments containing model configurations and settings.
            device: Device for tensor allocation.
        """
        super(Net_LSTM, self).__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = self.args.pred_len
        self.device = device

    def forward(self, x):
        """
        Forward pass through the Encoder-Decoder LSTM network.

        Args:
            x (torch.Tensor): Input tensor of the shape (batch_size, sequence length, input_dim)

        Returns:
            torch.Tensor: Output tensor of the shape (batch_size, pred_len, output_dim)
        """
        batch_size = x.shape[0]
        input_length = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:, ei:ei+1, :], encoder_hidden)
            
        decoder_input = x[:, -1, :].unsqueeze(1)  # First decoder input is the last element of input sequence
        decoder_hidden = encoder_hidden
        
        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]).to(self.device)
        
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:, di:di+1, :] = decoder_output
            
        return outputs[:, :, :self.args.out_features]
