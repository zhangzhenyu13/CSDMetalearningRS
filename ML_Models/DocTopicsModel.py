from gensim import corpora
import gensim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import copy,time
import pickle
from scipy import sparse

class LDAFlow:
    def __init__(self):
        self.n_features=1000

    def cleanDocs(self,docs_o):
        docs=copy.deepcopy(docs_o)
        stop = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        lemma = WordNetLemmatizer()
        for i in range(len(docs)):
            doc = docs[i]
            doc = " ".join([i for i in doc.lower().split() if i not in stop])
            doc = ''.join(ch for ch in doc if ch not in exclude)
            doc = " ".join(lemma.lemmatize(word) for word in doc.split())
            docs[i] = doc.split()
        return docs

    def transformVec(self,docs):
        #print(np.shape(docs),docs[0])
        docs=self.cleanDocs(docs)

        X = sparse.dok_matrix((len(docs), self.n_features))
        row = 0
        for doc in docs:
            doc_bow = self.dictionary.doc2bow(doc)
            lda_doc = self.lda[doc_bow]
            # print(type(lda_doc),lda_doc)
            for topic in lda_doc:
                X[row, topic[0]] = topic[1]
            row += 1
        return X.toarray()

    def train_doctopics(self,docs):
        #print(np.shape(docs),docs[0])
        t0 = time.time()

        docs = self.cleanDocs(docs)

        print("performing LDA ")
        self.dictionary = corpora.Dictionary(docs)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in docs]
        self.lda = gensim.models.LdaModel(doc_term_matrix, num_topics=self.n_features, id2word=self.dictionary)
        print("LDA built in %fs" % (time.time() - t0))
        with open("../data/saved_ML_models/ldamodel.pkl","wb") as f:
            model={}
            model["n_features"]=self.n_features
            model["dict"]=self.dictionary
            model["lda"]=self.lda
            pickle.dump(model,f)
    def loadModel(self):
        print("loading lda model")
        with open("../data/saved_ML_models/ldamodel.pkl","rb") as f:
            model=pickle.load(f)
            self.n_features=model["n_features"]
            self.dictionary=model["dict"]
            self.lda=model["lda"]
            print("loaded %d feature model"%self.n_features)
#######################################################################################################################
def hashingIDF(n_features):
    # Perform an IDF normalization on the output of HashingVectorizer
    hasher = HashingVectorizer(n_features=n_features,
                                stop_words='english', alternate_sign=False,
                                norm=None, binary=False)
    vectorizer = make_pipeline(hasher, TfidfTransformer())

    return vectorizer

def IDF(n_features):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    return vectorizer

class LSAFlow:
    def __init__(self):
        self.n_features=1000

    def transformVec(self,docs):
        X=IDF(self.n_features*10).fit_transform(docs)
        X = self.lsa.fit_transform(X)
        return X
    def train_doctopics(self,docs):
        t0 = time.time()

        X = IDF(self.n_features*10).fit_transform(docs)
        print("Performing  LSA")
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(self.n_features)
        normalizer = Normalizer(copy=False)
        self.lsa = make_pipeline(svd, normalizer)
        self.lsa.fit_transform(X)
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print("LSA built in %fs" % (time.time() - t0))
        with open("../data/saved_ML_models/lsamodel.pkl","wb") as f:
            model={}
            model["n_features"]=self.n_features
            model["lsa"]=self.lsa
            pickle.dump(model,f)

    def loadModel(self):
        print("loading lsa model")
        with open("../data/saved_ML_models/lsamodel.pkl","rb") as f:
            model=pickle.load(f)
            self.n_features=model["n_features"]
            self.lsa=model["lsa"]
            print("loaded %d feature model"%self.n_features)

#######################################################################################################################
