import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open('intents.json') as file:
	data = json.load(file)

try:
	with open('data.pickle','rb') as f: #rb stands for read bytes as we are going to save this as bytes my fellas!
		words, labels, training, output = pickle.load(f)


except:  #We use try except bcoz we won't clean the data if it's already cleaned out
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data['intents']:
		for pattern in intent['patterns']:
			#stemming means to get the root word of the sentence
			wrds = nltk.word_tokenize(pattern) #tokenize is breaking the dictionaries into words
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intents['tag'])

			if intent['tag'] not in labels:
				labels.append(intent['tag'])

	words = [stemmer.stem(w.lower()) for w in words if w != '?']
	words = sorted(list(set(words))) #set removes the duplicates

	labels = sorted(labels)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))] #output list will have the size of the length of the classes it has

	for x, doc in enumerate(docs_x):
		bag = []

		words = [stemmer.steam(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

	with open('data.pickle','wb') as f: #write all of the variables in pickle file so we can save it
		pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph() #Resetting all the default
net = tflearn.input_data(shape=[None, len(training[0])]) #expected shape of the input is the shape we have in our training data and all the training data have same pattern
net = tflearn.fully_connected(net,8) #8 neurons for the hidden layers
net = tflearn.fully_connected(net,8) #8 neurons for the hidden layers
net = tflearn.fully_connected(net,len(output[0]), activation = 'softmax') # to get probailities for each output. There are 6 outputs coz the no of lebels is 6
net = tflearn.regression(net)	

#Train the model 
model = tflearn.DNN(net)

try:
	model.load('model.tflearn')
except:
	model.fit(training, output, n_epoch = 10, batch_size=8, show_metric = True) #epoch is no of times we are going to show the same data to the model
	model.save('model.tflearn')


def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))] #blank bag of words

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se: # if current word we are looking at in the word list is equal to the word in our sentence
		 		bag[i].append(1)
	return numpy.array(bag)

def chat():
	print('Start talking with the bot! (Type quit to stop')
	while  True:
		inp = input('You:')
		if inp.lower() == 'quit':
			break

		results = model.predict([bag_of_words(inp, words)])[0]
		results_index = numpy.argmax(results) #argmax gives the index of the greatest value in the list
		tag = label[results_index]
		print(tag)
		if results[results_index] > 0.7:
			for tg in data[intents]:
				if tg['tag'] == tag:
					responses = tg['responses']

			print(random.choice(responses))

		else:
			print('I did not get that, try again.')


chat()













