import json
from DemoRe.demo_retriever import DemoRetriever
import numpy as np
import logging
# load demos
demos = json.load(open('../ALL_DATASETS_NEW/EE/ACE05/train.json','r'))
demo_embs = np.load(open('../ALL_DATASETS_NEW/EE/ACE05/train_question_embs.npy','rb'))

demo_retriever = DemoRetriever(demo_embs,demos,device='cuda:2')


query_id = np.random.randint(0,len(demos))
logging.info('Query: %s'%query_id)
query_emb = demo_embs[query_id]
query = demos[query_id]
logging.info('Query: %s'%query['sentence'])

retrieved_demos = demo_retriever.retrieve(query_emb,topk=5)
logging.info('Retrieved demos:')
for demo in retrieved_demos:
    logging.info(demo['sentence'])