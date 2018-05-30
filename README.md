Machine Translation using Encoder - Decoder Networks:

Author : Saiteja Sirikonda
Graduate from Arizona State University, class of 2018.

The main aim of the project was to understand how a simple Machine Translation machine works by understanding the intricacies of Encoder & Decoder networks and their implementation details.

For the sake of this project, I picked the German - English dataset as the size of the dataset is decent enough to train the Encoder - Decoder network.

Data Preprocessing:
As I explored the dataset, I notice that each line has both German and corresponding English sentence separated by a tab. It is ordered by sentence length, with shortest sentences on the top. While examining the data, I found that there are punctuations, contains a mix of Uppercase and Lowercase letters and has non-printable special characters in German that needed to handled in the data_preprocessing function.

Since, my PC cant handle training all the 100,000 training samples, I limit the training to 10,000 samples from the top.

Test & Train set:
I made a 90 - 10 split to the total data, with 90% used for training. The next thing is to enocode the

Data Encoding:
Before I got to the Model parameters, I needed to compute some metrics of the data. In our case, find the total size of the vocabulary for the entire data and that too for both the languages. Then, find the maximum sentence length in the dataset for both languages.
Since, this is a simple example and LSTMs take inputs of a fixed length, I first converted the input text to sequence and padded them.
To understand encode_input_sequences function, you can go through:
http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/

The Output sequence, has to be converted into a One-hot vector, I modelled my network as a Multiclass classification problem and Keras has a utility called to_categorical, which converts labelled data to onehot vectors, where in our case the number of classes is the size of the english vocabulary.
https://stackoverflow.com/questions/41494625/issues-using-keras-np-utils-to-categorical

Defining Model:
Now, that I have my trainX, trainY, textX and testY ready. I got started with defining the model.
 - Initialised a Sequential model.
 - An embedding layer: arguements - number of distinct words in training set, dimension of word embedding, input length of each sample.
 https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
 - Encoder LSTM network. dimension - 512 or 256. 512 gave me a better performance after experimentation.
 - RepeatVector - I realised that need the need for RepeatVector here, because the Encoder outputs a single vector, where as the decoder expects a sequential input. RepeatVector achieves that same vector is used for input for each timestep of Decoder.
 - Decoder LSTM
 - TimeDistributed - The use of this is explained best in the below links.
https://stats.stackexchange.com/questions/264546/difference-between-samples-time-steps-and-features-in-neural-network
https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/


Evaluating the Model:
I decided to evaluate the model using the Bleu score which can be imported from the nltk. The Basic theory behind it can be found here.
https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
https://stackoverflow.com/questions/40542523/nltk-corpus-level-bleu-vs-sentence-level-bleu-score

My model achieved a BLEU for weights (0.25,0.25,0.25,0.25) as 0.52

![Alt text](https://github.com/saiteja93/Machine-Translation-German---English-tentatively-/blob/master/img.PNG?raw=true "Screenshot of execution")








Links:
1. Goldberg, Yoav. "A primer on neural network models for natural language processing." Journal of Artificial Intelligence Research 57 (2016): 345-420.
2. Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems. 2014.
3. https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/
4. https://machinelearningmastery.com/configure-encoder-decoder-model-neural-machine-translation/
5. https://medium.com/syncedreview/english-japanese-neural-machine-translation-with-encoder-decoder-reconstructor-1a023eaab2a5
5. https://www.manythings.org/anki/
6. With Attention: https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
