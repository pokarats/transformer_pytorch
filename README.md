<!-- Build Your Own Transformer documentation master file, created by
sphinx-quickstart on Tue Jan  5 17:20:17 2021.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->
# Build Your Own Transformer’s documentation!

Implementation of the Transformer model. The submission also includes wrapper code for using the model for training and
testing in a translation.

For data, I used the provided dataset, specifically the `news-commentary-v8.de-en.de` and
`news-commentary-v8.de-en.en`.

For time efficiency, I have only trained the model on the first 1024 sentences partition of the dataset to make sure
that the model executes without errors. The results from simply running this small toy dataset does not seem to work as
it shows a BLEU score of 0.

I have not fully trained and evaluated the model on the entire dataset or on other datasets
to verify how well this works. However, from the `train_model.log` file, it does show that the training and validation
losses improve after a few epochs, which seems to be expected.

## Project File Structure

The project is organized inside the *transformer* directory as follows:

    
    * data: *training data files are inside this directory*


            * news-commentary-v8.de-en.de


            * news-commentary-v8.de-en.en


    * log: *logging files from running train)model.py and eval_model.py*


            * eval_model.log


            * train_model.log


    * pkl: *by default, script will save pickle files here; script will create this directory if it doesn’t exist*


    * saved_models: *by default, script will save model checkpoints here; script will create this directory if it doesn’t exist*


    * eval_model.py


    * preprocess.py


    * train_model.py


    * transformer_model.py


    * utils.py

## Required Libraries

In order to run the code, the following modules need to be installed first:

    
    * PyTorch


    * TorchText


    * spacy and ‘de’ and ‘en’ language models:


            * python -m spacy download en


            * python -m spacy download de


    * dill

Program is written and tested in Python 3.7.

## Run Instructions

The root level for this project is inside the *transformer* directory.

`python train_model.py` will train the model with default hyper-parameters and save the best model weights.

usage: train_model.py 

    [-h] [-data_path DATA_PATH] [-log_path LOG_PATH]
    [-pkl_path PKL_PATH] [-model_path MODEL_PATH]
    [-src_data SRC_DATA] [-trg_data TRG_DATA]
    [-src_lang SRC_LANG] [-trg_lang TRG_LANG]
    [-epochs EPOCHS] [-dmodel DMODEL] [-dff DFF]
    [-nlayers NLAYERS] [-heads HEADS] [-dropout DROPOUT]
    [-batch_size BATCH_SIZE] [-lr LR] [-max_len MAX_LEN]
    [-num_sents NUM_SENTS] [-toy_run TOY_RUN] [-debug DEBUG]
    [-save_model SAVE_MODEL] [-override OVERRIDE]
    [-seed SEED]

optional arguments:

  > -h               show this help message and exit  
  > -data_path       data directory; default='data'  
  > -log_path        log file directory; default='log'  
  > -pkl_path        pickle file directory; default='pkl'  
  > -model_path      saved models directory; default='saved_models'  
  > -src_data        src corpus filename; default='news-commentary-v8.de-en.de'  
  > -trg_data        trg corpus filename; default='news-commentary-v8.de-en.en'  
  > -src_lang        source language; default='de'  
  > -trg_lang        target language; default='en'  
  > -epochs          number of epochs to train for; default=25  
  > -dmodel          d_model or hidden size; default=512  
  > -dff             d_ff or hidden size of FFN sublayer; default=2048  
  > -nlayers         number of encoder/decoder layers; default=6  
  > -heads           number of attention heads; default=8  
  > -dropout         value for dropout p parameter; default=0.1  
  > -batch_size      number of samples per batch; default=48  
  > -lr              learning rate for gradient update; default=3e-4  
  > -max_len         maximum number of tokens in a sentence; default=150  
  > -num_sents       number of sentences to partition toy corpus; default=1024  
  > -toy_run         whether or not toy dataset; default=False  
  > -debug           turn logging to debug mode to display more info; default=False  
  > -save_model      True to save model checkpoint; default=True  
  > -override        override existing log file; default=False  
  > -seed            seed for the iterator random shuffling repeat; default=1234  

`train_model.py` has to be executed before eval_model.py.

`python eval_model.py` will evaluate the model on the test data and calculate BLEU score.

usage: eval_model.py 
    
    [-h] [-data_path DATA_PATH] [-log_path LOG_PATH]
    [-pkl_path PKL_PATH] [-model_path MODEL_PATH]
    [-src_data SRC_DATA] [-trg_data TRG_DATA]
    [-src_lang SRC_LANG] [-trg_lang TRG_LANG]
    [-dmodel DMODEL] [-dff DFF] [-nlayers NLAYERS]
    [-heads HEADS] [-dropout DROPOUT]
    [-batch_size BATCH_SIZE] [-lr LR] [-max_len MAX_LEN]
    [-num_sents NUM_SENTS] [-toy TOY] [-override OVERRIDE]
    [-seed SEED] [-sample_idx SAMPLE_IDX]

optional arguments:

  > -h                 
    >show this help message and exit  
  > -data_path         
    >data directory; default='data'  
  > -log_path        log file directory; default='log'  
  > -pkl_path        pickle file directory; default='pkl'  
  > -model_path      saved models directory; default='saved_models'  
  > -src_data        src corpus filename; default='news-commentary-v8.de-en.de'  
  > -trg_data        trg corpus filename; default='news-commentary-v8.de-en.en'  
  > -src_lang        source language; default='de'  
  > -trg_lang        target language; default='en'  
  > -dmodel          d_model or hidden size; default=512  
  > -dff             d_ff or hidden size of FFN sublayer; default=2048  
  > -nlayers         number of encoder/decoder layers; default=6  
  > -heads           number of attention heads; default=8  
  > -dropout         value for dropout p parameter; default=0.1  
  > -batch_size      number of samples per batch; default=48  
  > -lr              learning rate for gradient update; default=3e-4  
  > -max_len         maximum number of tokens in a sentence; default=150  
  > -num_sents       number of sentences to partition toy corpus; default=1024  
  > -toy_run         whether or not toy dataset; default=False  
  > -debug           turn logging to debug mode to display more info; default=False  
  > -save_model      True to save model checkpoint; default=True  
  > -override        override existing log file; default=False  
  > -seed            seed for the iterator random shuffling repeat; default=1234  
  > -sample_idx      index for a sample sentence pair example; default=8  

Results are recorded in the log files.


* Build Your Own Transformer’s documentation!


    * Project File Structure


    * Required Libraries


    * Run Instructions


* Modules


    * Preprocessing


    * Training


    * The Transformer Model


    * Testing


# Modules

## Preprocessing

**preprocess.py Module**

Script to pre-process data files into Transformer compatible batches and batch iterators


### class preprocess.Vocabulary(src_path, trg_path, tokenizer='spacy', src_lang='de', trg_lang='en')
Class to generate pre-processed dataset into batches and interators for training, validation, testing sets.
Module uses torchtext to generate dataset and iterators.
The default tokenizer is ‘spacy’


#### static make_batch_iterators(train_data, val_data, test_data, batch_size, device)
Create batch iterators for training, validation, and testing datasets using torchtext.data.BucketIterator


* **Parameters**

    
    * **train_data** – training set in TabularDataset object


    * **val_data** – validation set in TabularDataset object


    * **test_data** – test set in TabularDataset object


    * **batch_size** – number of sentences in a batch


    * **device** – cuda or cpu



* **Returns**

    


#### make_datasets(data_path, train_file, val_file, test_file, max_len, pkl_path, src_pkl_file='de_Field.pkl', trg_pkl_file='en_Field.pkl', sos='<sos>', eos='<eos>', saved_field=False)
Create training, validation, and test datasets from json files in TabularDataset format.


* **Parameters**

    
    * **data_path** – path to data directory


    * **train_file** – training json file (should be the same as the file saved in to_json function)


    * **val_file** – validation json file (should be the same as the file saved in to_json function)


    * **test_file** – test json file (should be the same as the file saved in to_json function)


    * **max_len** – maximum number of tokens in the sentence


    * **pkl_path** – path to pickle file directory


    * **src_pkl_file** – src pickle filename


    * **trg_pkl_file** – trg pickle filename


    * **sos** – start of sentence token


    * **eos** – end of sentence token


    * **saved_field** – whether or not there’s a saved torchtext.data.Field pickled file to be loaded



* **Returns**

    


#### make_train_val_test_splits(max_len, max_diff, test_split_size=0.2)
Create training, validation, and test splits


* **Parameters**

    
    * **max_len** – maximum number of tokens in the sentence


    * **max_diff** – maximum factor of difference in number of tokens between src and trg sentences


    * **test_split_size** – proportion to hold out for test set



* **Returns**

    


#### partition_raw_data(num_sents)
Partition dataset to specified number o sentences to create toy dataset


* **Parameters**

    **num_sents** – specify number of sentences to use for toy dataset



* **Returns**

    


#### static to_json(data_frame, path, filename)
Save data farm to json


* **Parameters**

    
    * **data_frame** – pandas dataframe


    * **path** – path to data directory


    * **filename** – filename for the .json file



* **Returns**

    

## Training

**train_model.py Module**

Wrapper module to train Transformer model and save model check points based on best validation loss.

This is the module to run to train the model. Model parameters can be configured/changed from default values
via command line options. Otherwise, default values are as per the “Attention is All You Need” paper:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
Attention Is All You Need. ArXiv:1706.03762 [Cs]. http://arxiv.org/abs/1706.03762


### train_model.epoch_time(start_time: float, end_time: float)
Function to calculate elapsed time in mins and secs


* **Parameters**

    
    * **start_time** – time function ended


    * **end_time** – time function finished



* **Returns**

    elapsed minutes, elapsed seconds



### train_model.evaluate(model, iterator, criterion, device)
run model in eval mode in batches


* **Parameters**

    
    * **model** – model to be used for evaluation, e.g. Transformer model


    * **iterator** – torchtext.data.BucketIterator object


    * **criterion** – loss function


    * **device** – cpu or cuda



* **Returns**

    epoch validation or test loss



### train_model.train(model, iterator, optimizer, criterion, clip, device)
Function to train in batches


* **Parameters**

    
    * **model** – model to be used for training, e.g. Transformer model


    * **iterator** – torchtext.data.BucketIterator object


    * **optimizer** – torch optimizer


    * **criterion** – loss function


    * **clip** – parameter for clip_grad_norm


    * **device** – cpu or cuda



* **Returns**

    epoch loss


## The Transformer Model

**transformer_model.py Module**

Implementation of the Transformer model as described in the ‘Attention is All You Need’ paper:
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
Attention Is All You Need. ArXiv:1706.03762 [Cs]. http://arxiv.org/abs/1706.03762

I followed the implementation examples in the following resources:

    
    * “The Annotated Transformer: [https://nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html)”


    * “Aladdin Perssons’ Transformer from Scratch YouTube video: [https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson](https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson)”


    * “Ben Trevett’s Language Translation with Transformer and Torchtext tutorial: [https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html?highlight=transformer](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html?highlight=transformer)”

I tried to balance between adhering to the variable names in the paper and using plain English for ease of comprehension


### class transformer_model.Decoder(trg_vocab_size, d_model, nx_layers, n_heads, d_ff, dropout_p, max_length, device)
Implement the Decoder block of nx DecodingLayers. From input token embeddings + position embeddings and
Nx DecodingLayers to generating the output that will be interpreted as the predicted translation


#### forward(trg: torch.Tensor, src_encoder_output: torch.Tensor, trg_mask: torch.Tensor, src_mask: torch.Tensor)

* **Parameters**

    
    * **trg** – src shape (N, trg_seq_len)


    * **src_encoder_output** – src_input_embeddings shape (N, src_seq_len, d_model)


    * **trg_mask** – shape (batch N, 1, trg_seq_len, trg_seq_len)


    * **src_mask** – shape (N, 1, 1, src_seq_len)



* **Returns**

    trg decoder ouput shape: (N, trg_seq_len, d_model)



### class transformer_model.DecoderLayer(d_model, n_heads, d_ff, dropout_p)
Implement the Decoder block with the 3 sublayers:

    
    1. decoder attention sublayer


    2. encoder attention sublayer


    3. FFN sublayer

This layer is stacked n_layers times in the Decoder part of the Transformer architecture


#### forward(trg, src_encoder_output, trg_mask, src_mask)

1. masked decoder attention sublayer

    1.a in the EncoderLayer V,K,Q are all from the same input trg


2. add and normalize attention sublayer output with residual input from before the decoder attention sublayer


3. apply dropout to the added and normed output of the attention sublayer


4. encoder attention sublayer

    4.a in the DecoderLayer Q is from output of previous decoder sublayer,
    V,K are from last encoder layer output


5. add and normalize encoder attention sublayer output with residual from the output of step 3.


6. apply dropout to the added and normed output of the encoder attention sublayer


7. output from encoder attention sublayer goes through FFN


8. add and normalize output_from_ffn with residual (output from decoder attention sublayer) from before the ffn layer


9. apply dropout to the added and normed output of the FFN sublayer


* **Parameters**

    
    * **trg** – shape (N, trg_seq_len, d_model)


    * **src_encoder_output** – shape (N, src_seq_len, d_model)


    * **trg_mask** – shape (batch N, 1, trg_seq_len, trg_seq_len)


    * **src_mask** – shape (N, 1, 1, src_seq_len)



* **Returns**

    


### class transformer_model.Encoder(src_vocab_size, d_model, nx_layers, n_heads, d_ff, dropout_p, max_length, device)
Implement the Encoder block of nx Encoder layers. From input token embeddings + position embeddings and
Nx EncodingLayers to generating the output that will be taken up by the Decoder block


#### forward(src: torch.Tensor, src_mask: torch.Tensor)

* **Parameters**

    
    * **src** – src shape (N, src_seq_len)


    * **src_mask** – src_mask shape (N, 1, 1, src_seq_len)



* **Returns**

    output from the last encoder layer in the stack



### class transformer_model.EncoderLayer(d_model, n_heads, d_ff, dropout_p)
Implement the Encoder block with the 2 sublayers: 1) attention sublayer 2) FFN sublayer.
This layer is stacked n_layers times in the Encoder part of the Transformer architecture


#### forward(src, src_mask)

1. attention sublayer

    1.a in the EncoderLayer V,K,Q are all from the same x input


2. add and normalize attention sublayer output with residual input from before the attention_sublayer


3. apply dropout to the added and normed output of the attention sublayer


4. output from attention sublayer goes through FFN


5. add and normalize output_from_ffn with residual (output from attention sublayer) from before the ffn layer


6. apply dropout to the added and normed output of the FFN sublayer


* **Parameters**

    
    * **src** – src shape (N, src_seq_len, d_model)


    * **src_mask** – src_mask shape (N, 1, 1, src_seq_len)



* **Returns**

    output Tensor that will become input Tensor in the next EncoderLayer



### class transformer_model.FeedForward(d_model, d_ff, dropout_p)
Implement the FeedForward sublayer in the TransformerBlock:
FFN = Relu(input_x \* W_1 + b_1) \* W_2 + b_2


#### forward(output_from_attention_sublayer: torch.Tensor)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.


### class transformer_model.Generator(d_model, trg_vocab_size)
Implement the last linear layer and the softmax of the Transformer architecture

This is not necessary if using nn.CrossEntropyLoss as softmax is already built-in


#### forward(output_from_decoder)

* **Parameters**

    **output_from_decoder** – shape (N, trg_seq_len, d_model)



* **Returns**

    output_from_decoder: shape (N, trg_seq_len, trg_vocab_size)???



### class transformer_model.MultiHeadedAttention(n_heads: int, d_model: int)
Implementation of the Multi-headed Attention sublayer in the Transformer block
This sublayer is the same architecture in both the Encoder and Decoder blocks
The multi-headded attention sublayer comprises the scaled-dot product attention mechanism


#### forward(value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor)

* **Parameters**

    
    * **value** – value Tensor shape (N, sequence length, d_model)


    * **key** – key Tensor shape (N, sequence length, d_model)


    * **query** – query Tensor shape (N, sequence length, d_model)


    * **mask** – src or trg masking Tensor shape src(N, 1, 1, src_seq_len) trg(N, 1, trg_seq_len, trg_seq_len)



* **Returns**

    


### class transformer_model.PositionEmbedding()
TODO implement PositionEmbedding per paper


### class transformer_model.Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, d_model, nx_layers, n_heads, d_ff, dropout_p, max_length, device)
Implement the Transformer architecture consisting of nx_layers of the encoder block and nx_layers of the decoder
block.


#### forward(src: torch.Tensor, trg: torch.Tensor)

* **Parameters**

    
    * **src** – src shape (N, src_seq_len)


    * **trg** – trg shape (N, trg_seq_len)



* **Returns**

    output shape (N, trg_seq_len, d_model)



#### make_src_mask(src: torch.Tensor)
Wherever src is not a pad idx, add dim of 1
:param src: src shape (N, src_seq_len)
:return: src_mask shape (N, 1, 1, src_seq_len)


#### make_trg_mask(trg: torch.Tensor)

* **Parameters**

    **trg** – trg shape (N, trg_seq_len)



* **Returns**

    trg_mask shape (N, 1, trg_seq_len, trg_seq_len)


## Testing

**eval_model.py Module**

Wrapper function to perform translation with saved model weights after training.
Evaluate model on the test dataset and calculate BLEU score from translation results.

<!-- Indices and tables -->
<!-- ================== -->
<!-- * :ref:`genindex` -->
<!-- * :ref:`modindex` -->
<!-- * :ref:`search` -->
