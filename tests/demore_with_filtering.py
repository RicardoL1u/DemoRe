import json
from DemoRe.demo_retriever import DemoRetriever
import numpy as np
import logging
# load demos
demos = json.load(open('../ALL_DATASETS_NEW/NER/ACE 2005/train.json','r'))
demo_embs = np.load(open('../ALL_DATASETS_NEW/NER/ACE 2005/train_sentence_embs.npy','rb'))
demos.extend(json.load(open('../ALL_DATASETS_NEW/NER/ACE 2004/train.json','r')))
demo_embs = np.concatenate([demo_embs,np.load(open('../ALL_DATASETS_NEW/NER/ACE 2004/train_sentence_embs.npy','rb'))],axis=0)
demos.extend(json.load(open('../ALL_DATASETS_NEW/NER/CoNLL 2003/train.json','r')))
demo_embs = np.concatenate([demo_embs,np.load(open('../ALL_DATASETS_NEW/NER/CoNLL 2003/train_sentence_embs.npy','rb'))],axis=0)

print('The shape of demo_embs is: ',demo_embs.shape)
print('The length of demos is: ',len(demos))

demo_retriever = DemoRetriever(demo_embs,demos,device='cuda:2')


query_id = np.random.randint(0,len(demos))
logging.info('Query: %s'%query_id)
query_emb = demo_embs[query_id]
query = demos[query_id]
logging.info('Query: %s'%query['sentence'])

# NER class type full set
# this part need to construct manualy 
NER_FULL_SET = {'facility', 'geographical social political', 'location', 'organization', 'person', 'vehicle', 'weapon','else'}
# entity name in speicific dataset should be manualy align with FULL_SET
ACE05_ENTITIES = {'facility', 'geographical social political', 'location', 'organization', 'person', 'vehicle', 'weapon'}
ACE04_ENTITIES = {'facility', 'geographical social political', 'location', 'organization', 'person', 'vehicle', 'weapon'}
CoNLL03_ENTITIES = {'else', 'location', 'organization', 'person'}

# this map is used to map the entity type in each dataset
# to a universal type set, namely the NER_FULL_SET
UNIVERSAL_MAP = {
    'facility': 'facility',
    'geographical social political': 'geographical social political',
    'location': 'location',
    'organization': 'organization',
    'person': 'person',
    'vehicle': 'vehicle',
    'weapon': 'weapon',
    'else': 'else',
}

# retrieve with target dataset is ACE05
retrieved_demos = demo_retriever.retrieve_with_filtering(ACE05_ENTITIES,UNIVERSAL_MAP,query_emb,topk=5)
print('Retrieved demos:')
for demo in retrieved_demos:
    print(demo['sentence'])
    print('Entities: ',[entity['type'] for entity in demo['entities']])


