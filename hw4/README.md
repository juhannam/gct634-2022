# Homework #4 (Optional) Music Transformer: Generating Music with Long-Term Structure

- 2019 ICLR, Cheng-Zhi Anna Huang, Google Brain
- Re-producer : Yang-Kichang
- [paper link](https://arxiv.org/abs/1809.04281) 
- [paper review](https://github.com/SSUHan/PaparReviews/issues/13)

This homework strongly refers to (https://github.com/jason9693/MusicTransformer-pytorch), with some minor changes.


# About the homework

The main goal of this homework is to understand basic symbolic music generation mechanism, by delving into the code structures of the pytorch implementation version of the paper 'Music Transformer: Generating Music with Long-Term Structure'. This paper utilizes the decoder part of the well-known transformer model, and generate midi events in an autoregressive manner. Transformer model is notorious on its heavy computational cost, which increases quadratically to the input length, $O(L^2)$. The model also utilizes relative positional self-attention (which is not the scope of our homework, but will be seen in the codes.), which will increase the total cost to $O(L^2D + L^2)$.  
The main contribution of this paper is introducing 'Relative Global Attention', which reduced the space complexity on obtaining relative embeddings. Therefore the total space complexity is reduced from $O(L^2D + L^2)$ to $O(LD + L^2)$. However, this main contribution part is out of our scope, we will target on understanding the overall code structure only. This includes **1) Preprocessing the midi into events. 2) How the transformer decoder model is built. 3) How the model is trained in the train phase, and how it generates events in the infer(generating) phase.**

Overall structure are as follows:

config 


# Simple Start ( Repository Setting )

After you've cloned this repository:

```bash

$ conda create -n 634hw4 python=3.7
$ conda activate 634hw4
$ sudo apt-get install gcc libgirepository1.0-dev libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 libapt-pkg-dev
$ cd gct634-2022-hw4
$ pip install -r requirements.txt # if there's warning, press 'w' and continue
$ conda config --set ssl_verify no
$ pip3 --default-timeout=500 install torch torchvision torchaudio
$ pip install tensorflow tqdm

```


# Preprocessing midi into events


* MIDI files of [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro) are used for training.
  Download the [maestro-v3.0.0-midi.zip](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip) (81MB uncompressed)

* Make a directory named 'dataset' and unzip the zip file inside the directory.

* We brought the preprocess implementation repository from [here](https://github.com/jason9693/midi-neural-processor).
  As we've learned from lectures, the MIDI file is represented as event numbers from 0 ~ 387.
  In the code, a last number 388 is added, referring to the padding token.
  
  ![](https://user-images.githubusercontent.com/11185336/51083282-cddfc300-175a-11e9-9341-4a9042b17c19.png)

After unzip, execute:
```bash
$ python preprocess.py ./dataset/maestro-v3.0.0 ./dataset/midi
```

# Train (optional)

Execute:
```bash
python3 train.py -m ./exp_config/train_results -c ./config/base.yml ./config/train.yml
```
We make a folder called ./exp_config/train_results, and save the checkpoint .pth files here.
./config/base.yml and ./config/train.yml are train configurations. Find out how each of the configures effect the model, and try to change them by yourself.
Note: The train will normally take a long time. Really long. Maybe a couple of days. Try to set the epoch large (maybe 10000) and try generating with intermediate .pth files. (ex) train-1098.pth, train-3000.pth and so on)
If you want to just generate results with trained model, I provided the pretrained .pth file, so you can use it.

##  Tensorboard for visualization!

Execute:
```bash
tensorboard --logdir logs/embedding256-layer6 --port 6006
```
Connect to https://localhost:6006 , and you should see the logged data!


## Hyper Parameter (change in ./config/base.yml)

* learning rate : 0.0001
* head size : 4
* number of layers : 6
* seqence length : 2048 (you can maybe decrease this value, if you want some speed..? 2048 event length corresponds to roughly 1 minute of midi.)
* embedding dim : 256 (dh = 256 / 4 = 64)
* batch size : 2 (depends on your computer's GPU power)


## Example Result (could be a bit worse than this result..)

-  Baseline Transformer ( Green, Gray Color ) vs Music Transformer ( Blue, Red Color )

* Loss

  ![loss](readme_src/loss.svg)

* Accuracy

  ![accuracy](readme_src/accuracy.svg)



# Generate Music

```bash
$ python3 generate.py -m ./exp_config/train_results -c ./config/base.yml ./config/generate.yml
```
Reads specific checkpoint .pth (should be specified inside generate.py) from ./exp_config/train_results and generate midi file inside a path specified in the generate.py script.

## Generated Sample Example (Youtube Link)
* click the image.

  [<img src="readme_src/sample_meta.jpeg" width="400"/>](https://www.youtube.com/watch?v=n6pi7QJ6nvk&list=PLVopZAnUrGWrbIkLGB3bz5nitWThIueS2)


# Some simple points to contemplate

1. Is this model deterministic or not? 

2. Why is the positional encoding based on sinusoidal form? At first thought, why not adding simple monotonic-increasing vector values for recognizing the absolute position of each input?

3. What is the role of 'pad token'? Why is it added, making the total event dimension 388+1=389? Is it necessary in this task? (How is the dataset obtained, and how does the code obtain (input, target) pair from the dataset, for each train step?)

4. This paper's tokenization method, also called as *MIDI-like*, is less frequently used than a more recent tokenization method called *REMI* (refer to [this paper](https://arxiv.org/abs/2002.00212)) Find out about the *REMI* tokenization method, and think about why this method is better than *MIDI-like*.

5. So what is 'relative global attention'? Why did the term 'relative' come about? (You can read the paper!)