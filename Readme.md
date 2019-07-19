# Text Summarizer-NLP ( Project Code AC01 )
A seq-seq model with attention mechanism which produces short summaries for large texts. We have defined the training and the inference portion for the Encoder-Decoder based seq-seq model in our code. 

During pre-processing, we have converted all the text to lower case and removed all the punctuation marks, stop words and short words.

The cleaned text and summaries are added as dataframe columns. 

For the decoder section, we have added the start and end tokens to our cleaned summary files.

Tokenization is used to convert the sequences to words, calculate the overall vocabulary and use of rare words in the clean text and clean summary.
While converting the vocabularies into word embeddings we have used a 100 dimenional vector for each word in our model. 

Training has been done on 20,000 samples and can be modified by the user in the code. 

Towards the conclusion of inference phase, we convert the numerical vectors/embeddings back to our text sequences. These texts are the predicted summaries of our model. It keeps on appending the predicted words to our decoded sentence till we come across an end token. The words are predicted on the basis of the argmax of the vectors representing them. The word corresponding to the vector with maximum argument is chosen as our prediction.

The Copy of Untitled0.ipynb file consists of the predicted results at the bottom. It can be viewed directly over there.

LINK TO GOOGLE COLAB CODE: https://colab.research.google.com/drive/1M6nOjr8NeTnRWy5o9rAj0sbtqBVDhFr1
While accessing the link, please make sure to upload the dataset file and attention package on Colab.

## Team Members ( **Team ID** 1959 )
**Lakshay Virmani** [lakshayvirmani77@gmail.com](mailto:lakshayvirmani77@gmail.com)

**Utkarsh Kulshrestha** [utkarshkulshrestha0@gmail.com](mailto:utkarshkulshrestha0@gmail.com)

**Mohit Nagpal** [nagpalm7@gmail.com](mailto:nagpalm7@gmail.com)

## Dataset Used

Newsroom data set will be used for final submission but for simplicity we have used different dataset.
**Link :-** [Download Here](https://www.kaggle.com/snap/amazon-fine-food-reviews)


# Steps To Run The Code

 1. Clone the repository 
 2. Download the data set from above link and extract in in the same folder.
 3. Make sure all the dependencies ( Keras, sklearn, Tensorflow, numpy, pandas, matplotlib etc. ) are installed.
 4. Run the Text-Summarization.py using command
  `python3 Text-Summarization.py` .
  
  
