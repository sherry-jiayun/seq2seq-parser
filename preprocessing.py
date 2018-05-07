import os
from pathlib import Path

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *

import numpy as np

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import random

def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	loc = ticker.MultipleLocator(base = 0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m,s)

def timeSince(since,percent):
	now = time.time()
	s = now -since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

INPUTSET = list()
OUTPUTSET = list()

MAX_LENGTH = 0

PAD_TOKEN = 0
START = 1
EOS = 2

USE_CUDA = torch.cuda.is_available()

# configure models
attn_model = 'dot'
hidden_dim = 256
n_layers = 3
dropout = 0.1
batch_size = 100
batch_size = 8

# configure training 
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 200000
epoch = 0
# 
# epoch = 3000
plot_every = 100
print_every = 500
evaluate_every = 500

print_loss_total = 0
plot_loss_total = 0

class Lang:
	def __init__(self,name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0:"PAD",1:"START",2:'EOS'}
		self.n_words = 3

	def index_words(self,sentence):
		for word in sentence.split(' '):
			self.index_word(word)

	def index_word(self,word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1
		return 

class GRUEncoder(nn.Module):
	def __init__(self,vocab_size, hidden_dim, embedding_dim, embedding):
		super(GRUEncoder, self).__init__()
		self.input_size = vocab_size
		self.hidden_dim = hidden_dim
		self.embedding = embedding
		self.dropout = 0.1
		self.gru = nn.GRU(embedding_dim,hidden_dim,3,dropout = self.dropout,bidirectional = True)
	
	def init_hidden(self):
		return Variable(torch.zeros(3,1,self.hidden_dim))
		#return (Variable(torch.zeros(1,1,self.hidden_dim)),
		#	Variable(torch.zeros(1,1,self.hidden_dim)))

	def forward(self,input,input_lengths,hidden = None):
		# seq_len = len(input)
		embedding = self.embedding(input)
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedding,input_lengths)
		outputs, hidden = self.gru(packed,hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
		outputs =outputs[:,:,:self.hidden_dim] + outputs[:,:,self.hidden_dim:]
		return outputs, hidden
class Attn(nn.Module):
	def __init__(self,method, hidden_dim, max_length = MAX_LENGTH):
		super(Attn,self).__init__()

		self.method = method
		self.hidden_dim = hidden_dim
		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_dim,self.hidden_dim)
		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
			self.other = nn.Parameter(torch.FloatTensor(1,self.hidden_dim))

	def forward(self,hidden, encoder_outputs):
		max_len = encoder_outputs.size(0)
		this_batch_size = encoder_outputs.size(1)
		# print(this_batch_size)
		# print (hidden,encoder_outputs)

		attn_energies = Variable(torch.zeros(this_batch_size,max_len))
		if USE_CUDA:
			attn_energies = attn_energies.cuda()
		# print(hidden.size())
		for b in range(this_batch_size):
			for i in range(max_len):
				# print(hidden[:,b].view(hidden[:,b].numel()).size())
				# print(encoder_outputs[i,b].size())
				# self.score(hidden[:,b],encoder_outputs[i,b].unsqueeze(0))
				attn_energies[b,i] = self.score(hidden[:,b].view(hidden[:,b].numel()),encoder_outputs[i,b])

		return F.softmax(attn_energies).unsqueeze(1)

	def score(self,hidden,encoder_output):

		if self.method == 'dot':
			energy = hidden.dot(encoder_output)
			#print(energy.shape)
			return energy
		elif self.method == 'general':
			energy = self.attn(encoder_output)
			energy = hidden.dot(energy)
			return energy
		elif self.method == 'concat':
			energy = self.attn(torch.cat((hidden,encoder_output),1))
			energy = hidden.dot(energy)
			return energy

class GRUADecoder(nn.Module):
	def __init__(self,attn_model, embedding_dim,hidden_dim,embedding,output_size,max_length = MAX_LENGTH): # how to decide output_size?
		super(GRUADecoder,self).__init__()

		self.attn_model = attn_model
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.output_size = output_size
		self.n_layers = 3
		self.dropout = 0.1

		self.embedding = nn.Embedding(self.output_size,self.hidden_dim)
		self.embedding_dropout = nn.Dropout(0.1)
		self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, 3, dropout = 0.1)
		self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
		self.out = nn.Linear(hidden_dim,output_size)

		if attn_model != 'none':
			self.attn = Attn(attn_model,hidden_dim)

	def forward(self,input_seq, last_hidden, encoder_outputs):

		batch_size = input_seq.size(0)
		embedding = self.embedding(input_seq)
		embedding = self.embedding_dropout(embedding)
		embedding = embedding.view(1, batch_size,self.hidden_dim)

		rnn_output, hidden = self.gru(embedding,last_hidden)

		attn_weights = self.attn(rnn_output,encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0,1))

		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output,context),1)
		concat_output = F.tanh(self.concat(concat_input))

		output = self.out(concat_output)
		return output, hidden,attn_weights

	def init_hidden(self):
		return Variable(torch.zeros(3,1,self.hidden_dim))

def save_checkpoint(state,filename='./checkpoint.pth.tar'):
	if os.path.isfile(filename):
		os.remove(filename)
	torch.save(state,filename)

def create_word_embeding(input_lang,path,embedding_dim):
	word2index = input_lang.word2index
	# print(word2index)
	embedding_file = open(path+'/glove.6B.'+str(embedding_dim)+'d.txt','r')
	embedding_lines = embedding_file.readlines()
	embedding_dict = dict()
	for line in embedding_lines:
		line = line.strip()
		lines = line.split(' ')
		if not len(lines) == (embedding_dim+1):
			pass
		else:
			embedding_dict[lines[0]] = []
			features_list = lines[1:]
			for f in features_list:
				embedding_dict[lines[0]].append(float(f))
	embedding_word2dict = {} 
	for key in word2index.keys():
		if key.lower() in embedding_dict.keys():
			embedding_word2dict[word2index[key]] = embedding_dict[key.lower()]
		#else:
		#	print(key)
	embedding_word2list = []
	index = len(word2index)
	for i in range(index+4): # the last index for <UKW>
		if i not in embedding_word2dict.keys():
			listtmp = []
			for j in range(embedding_dim):
				listtmp.append(0)
			embedding_word2list.append(listtmp)
		else:
			embedding_word2list.append(embedding_word2dict[i])
	numpy_word2list = np.asarray(embedding_word2list)

	# frozen weight!!
	embed = nn.Embedding(index+4, embedding_dim)
	embed.weight.data.copy_(torch.from_numpy(numpy_word2list))
	# print(numpy_word2list)
	# embed.weight.requires_grad = False
	return embed
	# print (numpy_word2list.shape)

def pad_seq(seq,max_length):
	seq += [PAD_TOKEN for i in range(max_length - len(seq))]
	return seq

'''def indexes_from_sentence(lang,sentence):
	return [lang.word2index[word] if word in lang.word2index.keys() else lang.n_words+1 for word in sentence.split(' ')]

def variable_from_sentence(lang,sentence):
	indexes = indexes_from_sentence(lang,sentence)
	indexes.append(EOS)
	# var = Variable(torch.LongTensor(indexes).view(-1,1,1))
	var = Variable(torch.LongTensor(indexes).view(-1,1))
	return var

def variables_from_pair(pair,input_lang,output_lang):
	input_variable = variable_from_sentence(input_lang,pair[0])
	# print(pair[1])
	target_variable = variable_from_sentence(output_lang,pair[1])
	return input_variable, target_variable'''
def indexes_from_sentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')] + [EOS]

def random_batch(batch_size):
	input_seqs = []
	target_seqs = []

	for i in range(batch_size):
		pair = random.choice(pairs)
		input_seqs.append(indexes_from_sentence(input_lang,pair[0]))
		target_seqs.append(indexes_from_sentence(output_lang,pair[1]))

	seq_pairs = sorted(zip(input_seqs,target_seqs),key = lambda p:len(p[0]),reverse=True)
	input_seqs, target_seqs = zip(*seq_pairs)

	input_lengths = [len(s) for s in input_seqs]
	input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
	target_lengths = [len(s) for s in target_seqs]
	target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

	input_var = Variable(torch.LongTensor(input_padded)).transpose(0,1)
	target_var = Variable(torch.LongTensor(target_padded)).transpose(0,1)

	if USE_CUDA:
		input_var = input_var.cuda()
		target_var = target_var.cuda()

	return input_var, input_lengths, target_var,target_lengths
def read_langs(path):
	path = "ontonotes.train.conll"
	train = open(path,'r')
	trainData = train.readlines()
	tag_linear = ''
	inputlist = list()
	outputlist = list()
	input_lines = list()
	output_lines = list()
	pairs = list()
	inputline = None
	outputline = None

	MAX_LENGTH = 0
	#for i in range(300):
	#	td = trainData[i]
	for td in trainData:
		tdstrip = ' '.join(td.split()) # clean
		tdlist = tdstrip.split(' ') # split
		if not len(tdlist) > 6:
			tags = []
			head = []
			tag_tmp = None
			start = False
			for c in tag_linear:
				if c == '(':
					if tag_tmp and not tag_tmp == '':
						tags.append(tag_tmp)
					if start:
						head.append(tag_tmp.replace('(',''))
					tag_tmp = c
					start = True
				elif c == ' ':
					if tag_tmp and not tag_tmp == '':
						tags.append(tag_tmp)
					if start:
						head.append(tag_tmp.replace('(',''))
					start = False
					tag_tmp = ''
				elif c == ')':
					tag_tmp = c + head[-1]
					tags.append(tag_tmp)
					head = head[:-1]
					tag_tmp = ''
				else:
					tag_tmp += c
			if inputline:
				inputline.reverse()
				inputline = ' '.join(inputline)
				outputline = ' '.join(tags)
				output_lines.append(outputline)
				input_lines.append(inputline)
				pairtmp = (inputline,outputline)
				if len(tags) > MAX_LENGTH:
					MAX_LENGTH = len(tags)
				pairs.append(pairtmp)
				inputline = None
				outputline = None
				
			tag_linear = ''
		else:
			token = tdlist[3]
			postag = tdlist[4]
			parsertag = tdlist[5]
			if postag.upper() == postag.lower():
				postagstr = ' '+postag + ' '
			else:
				postagstr = ' XX '
			parsertmp = parsertag.replace('*',postagstr)
			tag_linear += parsertmp
			if inputline:
				inputline.append(token)
				# inputline += token
			else:
				inputline = list()
				inputline.append(token)

	input_lang = Lang('sentence')
	output_lang = Lang('parsertag')

	for pair in pairs:
		# print(pair[0])
		input_lang.index_words(pair[0])
		output_lang.index_words(pair[1])

	return input_lang,output_lang,pairs, MAX_LENGTH

path = "ontonotes.train.conll"
input_lang, output_lang, pairs, MAX_LENGTH = read_langs(path)
'''print ("Input:\t\t",pairs[0][0])
print ("Parser tree:\t",pairs[0][1])
print()
print ("Input:\t\t",pairs[1][0])
print ("Parser tree:\t",pairs[1][1])
print()'''
print(len(pairs))
print(MAX_LENGTH)
path = "./glove.6B"
embedding = create_word_embeding(input_lang,path,50)

encoder_model_exist = Path('encoder_model.pt')
decoder_model_exist = Path('decoder_model.pt')
encoder_checkpoint = Path('encoder_checkpoint_1.pth.tar')
decoder_checkpoint = Path('decoder_checkpoint_1.pth.tar')
encoder = None
decoder = None
'''if encoder_model_exist.exists() and decoder_model_exist.exists():
	print('Load previous work')
	encoder = torch.load(encoder_model_exist)
	decoder = torch.load(decoder_model_exist)
	encoder_optimizer = optim.Adam(encoder.parameters(),lr = learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(),lr = learning_rate * decoder_learning_ratio )'''

if encoder_checkpoint.exists() and decoder_checkpoint.exists():
	checkpoint_en = torch.load(encoder_checkpoint)
	checkpoint_de = torch.load(decoder_checkpoint)
	epoch = checkpoint_en['epoch']
	encoder_optimizer = checkpoint_en['optimizer']
	decoder_optimizer = checkpoint_de['optimizer']
	encoder = model.load_state_dict(checkpoint_en['state_dict'])
	decoder = model.load_state_dict(checkpoint_de['state_dict'])

if not encoder or not decoder:
	print('fresh start')
	encoder = GRUEncoder(input_lang.n_words,hidden_dim,50,embedding)
	decoder = GRUADecoder(attn_model,50,hidden_dim,embedding,output_lang.n_words)
	encoder_optimizer = optim.Adam(encoder.parameters(),lr = learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(),lr = learning_rate * decoder_learning_ratio )

if USE_CUDA:
	encoder.cuda()
	decoder.cuda()
encoder_optimizer = optim.Adam(encoder.parameters(),lr = learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(),lr = learning_rate * decoder_learning_ratio )
criterion = nn.CrossEntropyLoss()

def evaluate(input_seq, max_length=MAX_LENGTH):
	input_seq = ''.join(input_seq)
	input_lengths = [len(input_seq.split(' '))]
	input_seqs = [indexes_from_sentence(input_lang,input_seq)]
	input_batches = Variable(torch.LongTensor(input_seqs),volatile = True).transpose(0,1)

	if USE_CUDA:
		input_batches = input_batches.cuda()

	encoder.train(False)
	decoder.train(False)
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
	decoder_input = Variable(torch.LongTensor([START]), volatile=True) # SOS
	decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
	if USE_CUDA:
		decoder_input = decoder_input.cuda()

	decoded_words = []
	decoder_attentions = torch.zeros(max_length + 1,max_length + 1)

	for di in range(max_length):
		decoder_output,decoder_hidden,decoder_attention = decoder(
			decoder_input,decoder_hidden,encoder_outputs)
		decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

		topv, topi = decoder_output.data.topk(1)
		ni = topi[0][0]
		if ni == EOS:
			decoded_words.append('EOS')
			break
		else:
			decoded_words.append(output_lang.index2word[ni])
		decoder_input = Variable(torch.LongTensor([[ni]]))
		if USE_CUDA: decoder_input = decoder_input.cuda()

	encoder.train(True)
	decoder.train(True)

	return decoded_words, decoder_attentions[:di+1,:len(encoder_outputs)]
def evaluate_and_show_attention(input_sentence,target_sentence = None):
	output_words,attentions = evaluate(input_sentence)
	output_sentence = ' '.join(output_words)
	print('>',input_sentence)
	if target_sentence is not None:
		print('=',target_sentence)
	print('<',output_sentence)

def evaluateRandomly(encoder,decoder, max_length = 0, n=5):
	[input_sentence, target_sentence] = random.choice(pairs)
	evaluate_and_show_attention(input_sentence,target_sentence)

def train(input_batches, input_lengths,target_batches,target_lengths,
	encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length=MAX_LENGTH):
	
	# encoder_hidden = encoder.init_hidden()
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0 

	encoder_outputs,encoder_hidden = encoder(input_batches,input_lengths,None)

	decoder_input = Variable(torch.LongTensor([START] * batch_size))
	decoder_hidden = encoder_hidden[:decoder.n_layers]

	max_target_length = max(target_lengths)
	all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

	if USE_CUDA:
		decoder_input = decoder_input.cuda()
		all_decoder_outputs = all_decoder_outputs.cuda()

	for t in range(max_target_length):
		decoder_output,decoder_hidden,decoder_attn = decoder(
			decoder_input,decoder_hidden,encoder_outputs)
		all_decoder_outputs[t] = decoder_output
		decoder_input = target_batches[t]

	loss = masked_cross_entropy(
		all_decoder_outputs.transpose(0,1).contiguous(),
		target_batches.transpose(0,1).contiguous(),
		target_lengths)
	loss.backward()

	ec = torch.nn.utils.clip_grad_norm(encoder.parameters(),clip)
	dc = torch.nn.utils.clip_grad_norm(decoder.parameters(),clip)

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.data[0],ec,dc
	# return 0, 0, 0

ecs = []
dcs = []
eca = 0
dca = 0
start = time.time()
while epoch < n_epochs:
	epoch += 1
	input_batches, input_lengths, target_batches,target_lengths = random_batch(batch_size)

	loss,ec,dc = train(
		input_batches,input_lengths,target_batches,target_lengths,
		encoder,decoder,
		encoder_optimizer,decoder_optimizer,criterion)
	print_loss_total += loss
	plot_loss_total += loss
	eca += ec
	dca += dc
	# print("HERE") 
	if epoch % print_every == 0:
		print_loss_avg = print_loss_total/print_every
		print_loss_total = 0
		print('%s (%d %d%%) %.4f' % (timeSince(start,epoch/n_epochs),epoch,epoch/n_epochs*100,print_loss_avg))
		if epoch % evaluate_every == 0:
			evaluateRandomly(encoder, decoder)
			save_checkpoint({'epoch':(epoch+1),
				'state_dict':encoder.state_dict(),
				'optimizer':encoder_optimizer},'encoder_checkpoint_1.pth.tar')
			save_checkpoint({'epoch':(epoch+1),
				'state_dict':decoder.state_dict(),
				'optimizer':decoder_optimizer},'decoder_checkpoint_1.pth.tar')
			# torch.save(encoder, 'encoder_model.pt')
			# torch.save(decoder, 'decoder_model.pt')

	if epoch % plot_every == 0:
		plot_loss_avg = plot_loss_total / plot_every
		# plot_losses.append(plot_loss_avg)
		plot_loss_total = 0

# showPlot(plot_losses)
# torch.save(encoder, 'encoder_model.pt')
# torch.save(decoder, 'decoder_model.pt')
