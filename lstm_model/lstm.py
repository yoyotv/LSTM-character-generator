import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from keras.utils import to_categorical


# build the model
class CharLSTM(nn.ModuleList):
    def __init__(self, sequence_len, vocab_size, hidden_dim, batch_size,int2char,char2int):
        super(CharLSTM, self).__init__()
        
        # init the meta parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.int2char = int2char
        self.char2int = char2int
        

        self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim).cuda()
        self.lstm_2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim).cuda()
        self.dropout = nn.Dropout(0.2).cuda()
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size).cuda()
        
    def forward(self, x, hc):
        
        self.train().cuda()
        
        # empty tensor for the output of the lstm
        output_seq = torch.empty((self.sequence_len, 
                                  self.batch_size, 
                                  self.vocab_size)).cuda()
        
        # pass the hidden and the cell state from one lstm cell to the next one
        # we also feed input at time step t to the cell
        # init the both layer cells with the zero hidden and zero cell states
        hc_1, hc_2 = hc, hc
        
        # for every step in the sequence
        for t in range(self.sequence_len):
            
            # get the hidden and cell states from the first layer cell
            hc_1 = self.lstm_1(x[t], hc_1)
            
            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1
        
            # pass the hidden state from the first layer to the cell in the second layer
            hc_2 = self.lstm_2(h_1, hc_2)
            
            # unpack the hidden and cell states from the second layer cell
            h_2, c_2 = hc_2
        
            # form the output of the fc
            output_seq[t] = self.fc(self.dropout(h_2)).cuda()
        
        # return the output sequence
        return output_seq.view((self.sequence_len * self.batch_size, -1))
          
    def init_hidden(self):
        
        return (torch.zeros(self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.batch_size, self.hidden_dim).cuda())
    
    def init_hidden_predict(self):

        return (torch.zeros(1, self.hidden_dim).cuda(),
                torch.zeros(1, self.hidden_dim).cuda())
    
    def predict(self, char, top_k=5, seq_len=1280):

        # set the evaluation mode
        self.eval()
        
        # placeholder for the generated text
        seq = np.empty(seq_len+1)
        seq[0] = self.char2int[char]
        
        # initialize the hidden and cell states
        hc = self.init_hidden_predict()
        
        # now we need to encode the character - (1, vocab_size)
        char = to_categorical(self.char2int[char], num_classes=self.vocab_size)
        
        # add the batch dimension
        char = torch.from_numpy(char).unsqueeze(0).cuda()
        
        # now we need to pass the character to the first LSTM cell to obtain 
        # the predictions on the second character
        hc_1, hc_2 = hc, hc
        
        # for the sequence length
        for t in range(seq_len):
            

            hc_1 = self.lstm_1(char, hc_1)
            h_1, _ = hc_1
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, _ = hc_2            
            h_2 = self.fc(h_2)
            h_2 = F.softmax(h_2, dim=1)
            
            # h_2 now holds the vector of predictions (1, vocab_size)
            # we want to sample 5 top characters
            p, top_char = h_2.cpu().topk(top_k)
            
            # get the top k characters by their probabilities
            top_char = top_char.squeeze().numpy()
            
            # sample a character using its probability
            p = p.detach().squeeze().numpy()
            char = np.random.choice(top_char, p = p/p.sum())
        
            # append the character to the output sequence
            seq[t+1] = char
            
            # prepare the character to be fed to the next LSTM cell
            char = to_categorical(char, num_classes=self.vocab_size)
            char = torch.from_numpy(char).unsqueeze(0).cuda()
            
        return seq

if __name__ == '__main__':
  main()
           
           



















