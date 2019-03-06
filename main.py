import sys
sys.path.append("function")
sys.path.append("lstm_model")



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import function
import lstm


from keras.utils import to_categorical


path1='data/bible.txt'


text=function.read_and_merge(path1)

#organize it as dict
characters = tuple(set(text))
int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}
encoded = np.array([char2int[char] for char in text])



net = lstm.CharLSTM(sequence_len=1280, vocab_size=len(char2int), hidden_dim=512, batch_size=128,int2char=int2char,char2int=char2int)
net.cuda()

# define the loss and the optimizer
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss().cuda()


# get the validation and the training data
val_index = int(len(encoded) * (0.9))                    #0.9 = 1-0.1
data, val_data = encoded[:val_index], encoded[val_index:]

# build lists for the validation losses
val_losses = list()
samples = list()

for epoch in range(10):

    hc = net.init_hidden()
    
    for i, (x, y) in enumerate(function.get_batches(data, 128, 1280)):

        x_train = torch.from_numpy(to_categorical(x, num_classes=net.vocab_size).transpose([1, 0, 2])).cuda()
        targets = torch.from_numpy(y.T).type(torch.LongTensor).cuda() # tensor of the target

        optimizer.zero_grad()

        # get the output sequence from the input and the initial hidden and cell states
        output = net(x_train, hc)
  
        # calculate the loss and we need to calculate the loss in all batches, so flat the targets tensor
        loss = criterion(output, targets.contiguous().view(1280*128))
        
        # calculate the gradients
        loss.backward()
        
        # update the parameters of the model
        optimizer.step()

        # feedback every 10 batches
        if i % 5 == 0:
          samples.append(''.join([int2char[int_] for int_ in net.predict("A", seq_len=1024)]))
                
        print("Epoch: {}, Batch: {}, Train Loss: {:.6f}".format(epoch, i, loss.item(), ))
                


print(samples[0])



