# coding=utf-8
import math
import random
import operator
import re

def rand_mat(x,y):
	"""
	Generate a x*y matrix where the sum of every row is 1
	"""
	mat=[]
	for i in xrange(x):
		rand=[]
		mat.append([])
		for k in xrange(y):
			rand.append(random.random())
		norm=sum(rand)
		for j in xrange(y):
			mat[i].append(rand[j]*1.0/norm)
	return mat


def load_stop_words(sw_file_path):
	"""
	Load StopWords list
	"""
	StopWords= set()
	sw_file = open(sw_file_path, "r")
	for word in sw_file:
		word = word.replace("\n", "")
		word = word.replace("\r\n", "")
		StopWords.add(word)
	sw_file.close()
	return StopWords


class document(object):
	"""
	Document class
	"""
	PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']
	CARRIAGE_RETURNS = ['\n', '\r\n']
	WORD_REGEX = "^[a-z']+$"
	def __init__(self, filepath):
		self.filepath=filepath
		self.file=open(self.filepath)
		self.line=[]
		self.words=[]

	def split(self, StorWords):
		"""
		split row texts
		"""
		self.lines=[l for l in self.file]
		for l in self.lines:
			words=l.split(' ')
			for w in words:
				clean_word=self._clean_word(w)
				if clean_word and (clean_word not in StorWords) and len(clean_word)>1:
					self.words.append(clean_word)
		self.file.close()

	def _clean_word(self, word):
		"""
		Convert words to lowercase
		Del the PUNCTUATION and CARRIAGE_RETURNS
		"""
		word=word.lower()	
		for punc in document.PUNCTUATION+document.CARRIAGE_RETURNS:
			word=word.replace(punc,'').strip("'")
		return word if re.match(document.WORD_REGEX,word) else None

class corpus(object):
	def __init__(self):
		self.documents=[]
		self.d_num=0
		self.vocabulary=[]
		self.v_num=0
		self.wordbag=[]

	def add_document(self, document):
		self.documents.append(document)
		self.d_num+=1

	def build_vocabulary(self):
		"""
		Build the vocabulary of corpus
		"""
		print "Building Vocabulary for %d document..."%self.d_num
		discrete_set=set()
		for document in self.documents:
			for word in document.words:
				discrete_set.add(word)
		self.vocabulary=list(discrete_set)
		self.v_num=len(self.vocabulary)
		print "Total %d words"%self.v_num

	def build_wordbag(self):
		"""
		Build the WordBag of corpus
		"""
		print "Building WordBag..."
		for i in xrange(self.d_num):
			self.wordbag.append({})
			for word in self.documents[i].words:
				if word not in self.wordbag:
					self.wordbag[i][self.vocabulary.index(word)]=1
				else:
					self.wordbag[i][self.vocabulary.index(word)]+=1

class pLSA(object):
	"""
	docs for pLSA model
	topics: the num of topic
	corpus: doc corpus, a list of docs{}, key: # of word, value: num of word
	docs: the num of document
	each: the sum of words of each doc
	words: the num of words of corpus
	likelihood: P(d,w)
	zw: P(w|z)
	dz: P(z|p)
	dw_z: P(z|d,w)
	_cal_p_dw(): Calculate P(d,w)
	topwords: output the # of high prob word in a topic
	"""
	def __init__(self, corpus, topics=5, output_path='./'):
		self.topics=topics
		self.corpus=corpus
		self.output_path=output_path
		self.docs=corpus.d_num
		self.each=map(sum,map(lambda x:x.values(),self.corpus.wordbag))
		self.words=max(map(max,map(lambda x:x.keys(), self.corpus.wordbag)))+1
		self.likelihood=0
		self.zw=rand_mat(self.topics,self.words)
		self.dz=rand_mat(self.docs,self.topics)
		self.dw_z=None
		self.topwords=10
#		self.beta=0.8
		self._cal_p_dw()

	def _cal_p_dw(self):
		self.p_dw=[]
		for d in xrange(self.docs):
			self.p_dw.append({})
			for w in self.corpus.wordbag[d]:
				tmp=0
				for z in xrange(self.topics):
					tmp+=self.zw[z][w]*self.dz[d][z]
				self.p_dw[d][w]=tmp

	def _e_step(self):
		self.dw_z=[]
		for d in xrange(self.docs):
			self.dw_z.append({})
			for w in self.corpus.wordbag[d]:
				self.dw_z[d][w]=[]
				for z in xrange(self.topics):
					self.dw_z[d][w].append((self.zw[z][w]*self.dz[d][z]*1.0)/self.p_dw[d][w])

	def _m_step(self):
		for z in xrange(self.topics):
			self.zw[z]=[0]*self.words
			for d in xrange(self.docs):
				for w in self.corpus.wordbag[d]:
					self.zw[z][w]+=self.corpus.wordbag[d][w]*self.dw_z[d][w][z]
			norm1=sum(self.zw[z])
			for w in xrange(self.words):
				self.zw[z][w]=self.zw[z][w]*1.0/norm1
		for d in xrange(self.docs):
			self.dz[d]=[0]*self.topics
			for z in xrange(self.topics):
				for w in self.corpus.wordbag[d]:
					self.dz[d][z]+=self.corpus.wordbag[d][w]*self.dw_z[d][w][z]
			for z in xrange(self.topics):
				self.dz[d][z]=self.dz[d][z]*1.0/self.each[d]
		self._cal_p_dw()

	def _cal_likelihood(self):
		self.likelihood=0
		for d in xrange(self.docs):
			for w in self.corpus.wordbag[d]:
				self.likelihood+=self.corpus.wordbag[d][w]*math.log(self.p_dw[d][w])

	def train(self, max_iter=100):
		cur=0
		print "Begin training..."
		for i in xrange(max_iter):
			print '%d iter'% i
			self._e_step()
			self._m_step()
			self._cal_likelihood()
			print 'likelihood: %f'%self.likelihood
			if cur!=0 and abs((self.likelihood-cur)*1.0/cur)<1e-8:
				break
			cur=self.likelihood
		print "Train is Completed!"
################# output #####################
	def p_dz_out(self):
		filename=self.output_path+'P-dz.txt'
		f=open(filename,'w')
		for di in xrange(self.docs):
			f.write("Doc #"+str(di)+":")
			for zi in xrange(self.topics):
				f.write(" "+str(self.dz[di][zi]))
			f.write("\n")
		f.close()

	def p_zw_out(self):
		filename=self.output_path+'P-zw.txt'
		f=open(filename,'w')
		for zi in xrange(self.topics):
			f.write("Topic #"+str(zi)+":\n")
			for wi in xrange(self.words):
				f.write(str(self.zw[zi][wi])+"\n")
		f.close

	def p_top_words(self):
		filename=self.output_path+'topwords.txt'
		f=open(filename,'w')
		cp=list(self.zw)
		for zi in xrange(self.topics):
			f.write('Topic #'+str(zi)+':\n')
			for i in xrange(self.topwords):
				maxprob=max(cp[zi])
				index=cp[zi].index(maxprob)
				f.write(str(self.corpus.vocabulary[index])+'\t')
				cp[zi].remove(maxprob)
			f.write('\n')
		f.close()



