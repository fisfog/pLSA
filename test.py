import pLSA
import os

def main():
	sw=pLSA.load_stop_words('stopwords.txt')
	homedir='./data/events_2010/'
	# print os.listdir(dir)
	documents=[]
	for dir in os.listdir(homedir):
		for file in os.listdir(homedir+dir):
			documents.append(pLSA.document(homedir+dir+'/'+file))
	corpus=pLSA.corpus()
	for d in documents:
		d.split(sw)
		corpus.add_document(d)	

	corpus.build_vocabulary()
	corpus.build_wordbag()
	p=pLSA.pLSA(corpus,5)
	p.train()
	p.p_dz_out()
	p.p_zw_out()
	p.p_top_words()

if __name__=="__main__":
	main()