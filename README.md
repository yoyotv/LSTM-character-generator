# LSTM-character-generator

Character to character generator

Case: one to many

## GET started

Success in :

pytorch : 0.4.1 , python 2.7

It runs under cuda, so if you want to run in cpu mode, just remove the "cuda( )" part in the code. 

### Try

`python main.py`
  
If every is fine, it should look like this

![result](https://raw.githubusercontent.com/yoyotv/LSTM-character-generator/master/figure/result.JPG)
### Scheme 

Use the concept of many to many.

Feed one character to LSTM in every timestep.
After doing it several times, calcalate the losses across all steps.

Train the net base on the total loss.

In the generate mode, giving it the first character.
It is able to generate the following character automatically.

### Acknowledgement

https://github.com/halahup
