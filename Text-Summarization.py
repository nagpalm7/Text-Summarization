#############################################################
# Text Summarizer ( Project Code AC01 )                     #
# Team Members ( **Team ID** 1959 )                         #
#   * Lakshay Virmani [lakshayvirmani77@gmail.com]          #
#   * Utkarsh Kulshrestha [utkarshkulshrestha0@gmail.com]   #
#   * Mohit Nagpal [nagpalm7@gmail.com]                     #
#############################################################
# Import Attention layer from custom module
from Attention import AttentionLayer
# Import packages
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import nltk
from nltk import word_tokenize
nltk.download('all')
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")
###############################################
# Data Pre-processing
###############################################
# Helper function to clean data (remove punctuation marks 
# and converting to lower case)
def text_clean(text,n):
  processed_string=text.lower()
  processed_string=re.sub(r'[^\w\s]','',processed_string)
  
 #word_tokens=word_tokenize(processed_string)
  stop_words = set(stopwords.words('english'))
  new=processed_string.split()
  tokens=[]
  for w in new:
    if w not in stop_words:
      tokens.append(w)
  long_words=[]
  for i in tokens:
        if len(i)>1:                                                 #removing short word
            long_words.append(i)   
  return (" ".join(long_words)).strip()

# Import and clean data
data=pd.read_csv("Reviews.csv",nrows=20000)
cleaned_text=[]
cleaned_summary=[]
# for lines in data['Text']:
#   clean_text.append(text_clean(lines,0))  
for i in range(0,20000):
  cleaned_text.append(text_clean(data['Text'][i],0))
  cleaned_summary.append(text_clean(data['Summary'][i],0))

#Tokenization of sentences
##Sentences tokenized into words and stored in a list
#from nltk.tokenize import word_tokenize
#words=[]
#for i in range(0,5000):
 # words.append(word_tokenize(string[i]))
  
#print(len(words))
#print(len(words[0]))

print(cleaned_text[0])
print(data['Text'][0])

print(cleaned_summary[0])
print(data['Summary'][0])

data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary

max_length_text=30
max_summary_len=8

len(cleaned_text)
len(cleaned_summary)

short_text=[]
short_summary=[]
for i in range(0,len(cleaned_text)):
  if(len(cleaned_summary[i].split())<=max_length_summary and len(cleaned_text[i].split())<=max_length_text):
    short_text.append(cleaned_text[i])
    short_summary.append(cleaned_summary[i])
    
df=pd.DataFrame({"text":short_text,"summary":short_summary})

df['summary'] = df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(df['text']),np.array(df['summary']),test_size=0.1,random_state=0,shuffle=True)

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
x_tokenizer=Tokenizer()
x_tokenizer.fit_on_texts((X_train))

#Definings rare words
threshold=4
count=0
tot_count=0
frequency=0
tot_frequency=0
common_words=0
for key,value in x_tokenizer.word_counts.items():
  tot_count=tot_count+1
  tot_frequency=tot_frequency+value
  
  if(value<threshold):
    count=count+1
    frequency=frequency+value
    
common_words=tot_count-count
print(tot_frequency)

#Now we will prepare the tokenizer only for the topmost common words which are not rare

x_tokenizer=Tokenizer(common_words)

x_tokenizer.fit_on_texts((X_train))


x_tr_seq=x_tokenizer.texts_to_sequences(X_train)
x_val_seq=x_tokenizer.texts_to_sequences(X_test)

X_train  = pad_sequences(x_tr_seq,  maxlen=max_length_text, padding='post')
X_test   = pad_sequences(x_val_seq, maxlen=max_length_text, padding='post')


x_voc=len(x_tokenizer.word_index) + 1

y_tokenizer=Tokenizer()
y_tokenizer.fit_on_texts((y_train))
threshold=6
cnt=0
tot_cnt=0
freq=0
tot_freq=0
common_words=0
for key,value in y_tokenizer.word_counts.items():
  tot_cnt=tot_cnt+1
  tot_freq=tot_freq+value
  if(value<threshold):
    cnt=cnt+1
    freq=freq+value
    
common_words=tot_cnt-cnt
y_tokenizer=Tokenizer(common_words)
y_tokenizer.fit_on_texts((y_train))


y_tr_seq=x_tokenizer.texts_to_sequences(y_train)
y_val_seq=x_tokenizer.texts_to_sequences(y_test)

y_train  = pad_sequences(x_tr_seq,  maxlen=max_length_summary, padding='post')
y_test   = pad_sequences(x_val_seq, maxlen=max_length_summary, padding='post')


y_voc=len(y_tokenizer.word_index) + 1
y_voc

y_tokenizer.word_counts['sostok'],len(y_train)
ind=[]
for i in range(len(y_train)):
    cnt=0
    for j in y_train[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_train=np.delete(y_train,ind, axis=0)
X_train=np.delete(X_train,ind, axis=0)


ind=[]
for i in range(len(y_test)):
    cnt=0
    for j in y_test[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_train=np.delete(y_train,ind, axis=0)
X_test=np.delete(X_test,ind, axis=0)

from keras import backend as K
K.clear_session()
latent_dim=300
embedding_dim=100



#Encoder
encoder_inputs=Input(shape=(max_length_text,))

enc_emb=Embedding(x_voc,embedding_dim,trainable=True)(encoder_inputs)

encoder_lstm1=LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)


#encoder lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#dense layer
decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

print(np.shape(X_train))
print(np.shape(y_train))
print(X_train)

print("__________________________________________________________________")

print(y_train)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')





es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

model.fit([X_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=3,callbacks=[es],batch_size=64, validation_data=([X_test,y_test[:,:-1]], y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:]))

reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_length_text,latent_dim)) # change

dec_emb2= dec_emb_layer(decoder_inputs) 

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

decoder_outputs2 = decoder_dense(decoder_inf_concat) 

decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1,1))
    
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

for i in range(0,100):
    print("Review:",seq2text(X_train[i]))
    print("Original summary:",seq2summary(y_train[i]))
    print("Predicted summary:",decode_sequence(X_train[i].reshape(1,max_length_text)))
    print("\n")

