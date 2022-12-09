import lightgbm as lgb
import numpy as np
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
import random

# code based on: http://ristek.link/TutorialLETOR

class Letor :
  ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)
  NUM_LATENT_TOPICS = 200
  model = None
  dictionary = Dictionary()
  documents = {}
  queries = {}
  group_qid_count = []
  dataset = []

  def __init__(self, model="tfidf") :
    self.documents = {}
    with open("nfcorpus/train.docs") as file:
      for line in file:
        doc_id, content = line.split("\t")
        self.documents[doc_id] = content.split()

    self.queries = {}
    with open("nfcorpus/train.vid-desc.queries",encoding="utf-8") as file:
      for line in file:
        q_id, content = line.split("\t")
        self.queries[q_id] = content.split()

    NUM_NEGATIVES = 1

    q_docs_rel = {} # grouping by q_id terlebih dahulu
    with open("nfcorpus/train.3-2-1.qrel") as file:
      for line in file:
        q_id, _, doc_id, rel = line.split("\t")
        if (q_id in self.queries) and (doc_id in self.documents):
          if q_id not in q_docs_rel:
            q_docs_rel[q_id] = []
          q_docs_rel[q_id].append((doc_id, int(rel)))

    # group_qid_count & dataset
    group_qid_count = []
    dataset = []
    for q_id in q_docs_rel:
      docs_rels = q_docs_rel[q_id]
      group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
      for doc_id, rel in docs_rels:
        dataset.append((self.queries[q_id], self.documents[doc_id], rel))
      dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
    self.dataset = dataset
    self.group_qid_count = group_qid_count

    # models
    bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
    if model == "lsi" :
      self.model = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS) # 200 latent topics
    elif model == "tfidf" :
      self.model = TfidfModel(bow_corpus,smartirs='ntc')

    self.train()
    
  def vector_rep(self,text):
    rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
    return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

  def features(self, query, doc):
    v_q = self.vector_rep(query)
    v_d = self.vector_rep(doc)
    q = set(query)
    d = set(doc)
    cosine_dist = cosine(v_q, v_d)
    jaccard = len(q & d) / len(q | d)
    return v_q + v_d + [jaccard] + [cosine_dist]

  def train(self):
    X = []
    Y = []
    for (query, doc, rel) in self.dataset:
      X.append(self.features(query, doc))
      Y.append(rel)

    X = np.array(X)
    Y = np.array(Y)

    self.ranker.fit(X, Y,
              group = self.group_qid_count)

  def process_rank(self,query,docs,file):
    X_unseen = []
    for doc_id, doc in docs:
      X_unseen.append(self.features(query.split(), doc.split()))

    X_unseen = np.array(X_unseen)

    scores = self.ranker.predict(X_unseen)

    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)
    count = 0
    file.write("\nSERP/Ranking :")
    for (did, score) in sorted_did_scores:
      if count > 9:
        break
      file.write(f"\n{did:30} {score:>.3f}")
      count+=1

  def calc(self,query,docs):
    X_unseen = []
    for doc_id, doc in docs:
      X_unseen.append(self.features(query.split(), doc.split()))

    X_unseen = np.array(X_unseen)

    scores = self.ranker.predict(X_unseen)

    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)
    return [x[0] for x in sorted_did_scores]  
