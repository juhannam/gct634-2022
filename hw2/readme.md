# Homework #2: Music Auto Tagging (Multi-Label Classification)[Colab Notebook](https://colab.research.google.com/drive/15M1ecjYxEufk94g7AK60cFMKJM480M2E?usp=sharing)

Music auto-tagging is an important task that can be used in many musical applications such as music search or recommender systems. 

Your mission is to build your own Neural Network model to represent audio signal. Specifically, the goals of this homework are as follows:

* Experiencing the whole pipeline of deep learning based system: data preparation, feature extraction, model training and evaluation
* Getting familiar with the Neural Network architectures for music representation
* Using Pytorch in practice

## Preparing The Dataset
We use the [magnatagatune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) dataset which has been the most widely used in the music tagging task.  The MagnaTagATune dataset consists of 25k music
clips from 6,622 unique songs.

The dataset contains 30-second audio files including 189 different tags
For this homework, we are going to use a magnatagatune with 8-second audio and only 50 genres.

We use subset of magnatagatune dataset (9074 samples x 8 sec).
To make your life easier, place them in a directory as below:

```
%%capture
!wget http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv
!gdown --id 1f_kUF9nTLFI0pJaZxm6YNt4t_J6-Q-dg
!tar -xvf gct634.tar.gz
```

```
├── annotations_final.csv
├── waveform
  ├── 1
  ├── ...
  └── d
```

## Training CNNs from Scratch
The baseline code is provided so that you can easily start the homework and also compare with your own algorithm.
The baseline model extracts mel-spectrogram and has a simple set of CNN model that includes convolutional layer, batch normalization, maxpooling and fully-connected layer.

### Question 1: Implement a CNN based on a given model specification
An architecture of CNN will be provided. Implement a CNN following the architecture.

### Question 2: Improve the performenace. [Leader Board](https://docs.google.com/spreadsheets/d/1WTFrSzQKjRqPfktVtrGKnFDbam4pAnayMdr8kECRYzM/edit?usp=sharing)
Now it is your turn. You should improve the baseline code with your own algorithm. There are many ways to improve it. The followings are possible ideas: 

* You can try 1D CNN or 2D CNN models and choose different model parameters:
    * Filter size
    * Pooling size
    * Stride size 
    * Number of filters
    * Model depth
    * Regularization: L2/L1 and Dropout

* You should try different hyperparameters to train the model and optimizers:
    * Learning rate
    * Model depth
    * Optimizers: SGD (with Nesterov momentum), Adam, RMSProp, ...

* You can try training a model using both mel-spectrograms and features extracted using the pre-trained models. However, end-to-end training using additional external data is prohibited. (Performance doesn't have a huge impact on grading. don't waste time)
  * Pretrained Tagging Model: https://github.com/minzwon/sota-music-tagging-models

* You can try different parameters (e.g. hop and window size) to extract mel-spectrogram or different features as input to the network (e.g. MFCC, chroma features ...). 

* You can also use ResNet or other CNNs with skip connections. 

* Furthermore, you can augment data using digital audio effects.
  * Audio Augmentation: https://music-classification.github.io/tutorial/part3_supervised/data-augmentation.html


## Deliverables
You should submit your Python code (`.ipynb` or `.py` files) and homework report (.pdf file) to KLMS. Q1 does not need to be included in the report (scored by code). Please write a report with only Q2 (Improve Algorithm). The report should include:

* Algorithm Description
* Experiments and Results
* Discussion



## Evaluation Criteria

* Did the author clarify the problem definition and write the dataset analysis?
* Did the author analyze the dataset and labels and write a suitable hypothesis?
* Did the author construct a deep learning model and write down the difference from the baseline?
* Did the author write findings and discussions?
* Clarity of formatting, the English does not need to be flawless, the text should be understandable, the code should be re-implementable.

## Note
The code is written using PyTorch but you can use TensorFlow.

## Credit
Thie homework was implemented by Jongpil Lee, Soonbeom Choi, Taejun Kim and SeungHeon Doh in the KAIST Music and Audio Computing Lab.