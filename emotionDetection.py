#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


# HELPER FUNCTIONS

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def read_csv(filename):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y


# In[3]:


X_train, Y_train = read_csv('emotionDetectionTraining.csv')
X_test, Y_test = read_csv('emotionDetectionTest.csv')


# In[4]:


Y_oh_train = convert_to_one_hot(Y_train, C = 10)
Y_oh_test = convert_to_one_hot(Y_test, C = 10)


# In[33]:


Y_oh_train[0]


# In[5]:


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt') #global vector of word


# In[6]:


def sentences_to_indices(X, word_to_index, max_len):

    m = X.shape[0]  # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):  # loop over training examples
        
        # Convert the ith sentence in lower case and split into a list of words
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1
    
    return X_indices


# In[7]:


X1 = np.array(["lol", "I am hungry", "this is very yummy", "i am sad"])
X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =", X1_indices)


# In[8]:


class NN(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, vocab_size, output_dim, batch_size):
        super(NN, self).__init__()

        self.batch_size = batch_size

        self.hidden_dim = hidden_dim

        self.word_embeddings = embedding

      # The LSTM takes word embeddings as inputs, and outputs hidden states
      # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, 
                          hidden_dim, 
                          num_layers=2,
                          dropout = 0.5,
                          batch_first = True)

      # The linear layer that maps from hidden state space to output space
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, sentence):
        sentence = sentence.to(device)

        embeds = self.word_embeddings(sentence)

        # initializing the hidden state to 0

        h0 = torch.zeros(2, sentence.size(0), hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(2, sentence.size(0), hidden_dim).requires_grad_().to(device)
      
        lstm_out, h = self.lstm(embeds, (h0, c0))
        # get info from last timestep only
        lstm_out = lstm_out[:, -1, :]

        # Dropout
        lstm_out = F.dropout(lstm_out, 0.5)

        fc_out = self.fc(lstm_out)
        out = fc_out
        out = F.softmax(out, dim=1)
        return out


# In[9]:


def pretrained_embedding_layer(word_to_vec_map, word_to_index, non_trainable=True):
    num_embeddings = len(word_to_index) + 1                   
    embedding_dim = word_to_vec_map["cucumber"].shape[0]  #  dimensionality of GloVe word vectors is 50

    # Initialize the embedding matrix as a numpy array of zeros of shape (num_embeddings, embedding_dim)
    weights_matrix = np.zeros((num_embeddings, embedding_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        weights_matrix[index, :] = word_to_vec_map[word]

    embed = nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix).type(torch.FloatTensor), freeze=non_trainable)

    return embed, num_embeddings, embedding_dim


# In[10]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, trainloader, criterion, optimizer, epochs=10):
    
    model.to(device)
    running_loss = 0
    
    train_losses, test_losses, accuracies = [], [], []
    for e in range(epochs):

        running_loss = 0
        
        model.train()
        
        for sentences, labels in trainloader:

            sentences, labels = sentences.to(device), labels.to(device)

            # 1) erase previous gradients (if they exist)
            optimizer.zero_grad()

            # 2) make a prediction
            pred = model.forward(sentences)

            # 3) calculate how much we missed
            loss = criterion(pred, labels)

            # 4) figure out which weights caused us to miss
            loss.backward()

            # 5) change those weights
            optimizer.step()

            # 6) log our progress
            running_loss += loss.item()
        
        
        else:
            model.eval()

            test_loss = 0
            accuracy = 0
          
          # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for sentences, labels in test_loader:
                    sentences, labels = sentences.to(device), labels.to(device)
                    log_ps = model(sentences)
                    test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(test_loader))
                accuracies.append(accuracy / len(test_loader) * 100)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
                   "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
        
    # Plot
    plt.figure(figsize=(20, 5))
    plt.plot(train_losses, c='b', label='Training loss')
    plt.plot(test_losses, c='r', label='Testing loss')
    plt.xticks(np.arange(0, epochs))
    plt.title('Losses')
    plt.legend(loc='upper right')
    plt.show()
    plt.figure(figsize=(20, 5))
    plt.plot(accuracies)
    plt.xticks(np.arange(0, epochs))
    plt.title('Accuracy')
    plt.show()


# In[11]:


import torch.utils.data

maxLen = len(max(X_train, key=len).split())
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)

X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)

embedding, vocab_size, embedding_dim = pretrained_embedding_layer(word_to_vec_map, word_to_index, non_trainable=True)

hidden_dim=128
output_size=5
batch_size = 32

model = NN(embedding, embedding_dim, hidden_dim, vocab_size, output_size, batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
epochs = 50
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_indices).type(torch.LongTensor), torch.tensor(Y_train).type(torch.LongTensor))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_indices).type(torch.LongTensor), torch.tensor(Y_test).type(torch.LongTensor))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

train(model, train_loader, criterion, optimizer, epochs)


# In[12]:


test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for sentences, labels in test_loader:
        sentences, labels = sentences.to(device), labels.to(device)
        ps = model(sentences)
        test_loss += criterion(ps, labels).item()

        # Accuracy
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
model.train()
print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
running_loss = 0


# In[13]:


# Python program to translate 
# speech to text and text to speech 
import speech_recognition as sr 
import pyttsx3 

# Initialize the recognizer 
r = sr.Recognizer() 

# Function to convert text to 
# speech 
def SpeakText(command): 
    # Initialize the engine 
    engine = pyttsx3.init() 
    engine.say(command) 
    engine.runAndWait() 


# In[14]:


def predict(input_text, print_sentence=True):
    labels_dict = {
        0 : "Life without love is like a tree without blossoms or fruit.",   #loving
        1 : "I can make arrangments for you to play.",   #activity
        2 : "Glad to see you Happy.",    #happy
        3 : "Oh No. It must be painful. Can I do something to make you feel better?", #sad/pain
        4 : "Yummy! Wanna eat something?",  #hungry
    }
  # Convert the input to the model
    x_test = np.array([input_text])
    X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
    sentences = torch.tensor(X_test_indices).type(torch.LongTensor)

    # Get the class label
    ps = model(sentences)
    top_p, top_class = ps.topk(1, dim=1)
    label = int(top_class[0][0])

    if print_sentence:
        SpeakText(labels_dict[label])
        print(labels_dict[label])

    return label


# In[16]:


# Loop infinitely for user to 
# speak 
i=1
while(i==1):

    # Exception handling to handle 
    # exceptions at the runtime 
    try: 

        # use the microphone as source for input. 
        with sr.Microphone() as source2: 
            # wait for a second to let the recognizer 
            # adjust the energy threshold based on 
            # the surrounding noise level 
            r.adjust_for_ambient_noise(source2, duration=0.2) 

            #listens for the user's input
            print("How Are you Feeling Today?")
            audio2 = r.listen(source2) 
            i=0
            # Using google to recognize audio 
            MyText = r.recognize_google(audio2) 
            print(MyText)
            MyText = MyText.lower() 
            predict(MyText)
            #SpeakText(MyText)

    except sr.RequestError as e: 
        print("Could not request results; {0}".format(e)) 

    except sr.UnknownValueError: 
        print("unknown error occured") 


# In[ ]:


#predict("awesome life")
#predict("I want a football")
#predict("I want noodles")
#predict("I have a stomach ache")
#predict("This is the worst day of my life")


# In[26]:


#using seaborn
import seaborn as sns
sns.countplot(Y_train, palette = "Set2")
plt.xlabel('Label')
plt.title('Emotions')


# In[27]:


sns.countplot(Y_test, palette = "Set2")
plt.xlabel('Label')
plt.title('Emotions')


# In[ ]:




