import numpy as np

def read_and_merge(path1):
    total = list()
    text1 = open(path1,'r').read()
    return text1

def get_batches(arr, n_seqs_in_a_batch, n_characters):

    batch_size = n_seqs_in_a_batch * n_characters
    n_batches = len(arr)//batch_size
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs_in_a_batch, -1))
    
    for n in range(0, arr.shape[1], n_characters):
        # The features
        x = arr[:, n:n+n_characters]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_characters]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

if __name__ == "__main__":
  main()
    









      
