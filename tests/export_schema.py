import json
ace05 = json.load(open('../ALL_DATASETS_NEW/NER/ACE 2005/train.json','r'))
ace04 = json.load(open('../ALL_DATASETS_NEW/NER/ACE 2004/train.json','r'))
conll03 = json.load(open('../ALL_DATASETS_NEW/NER/CoNLL 2003/train.json','r'))

ace05_entities = set()
ace04_entities = set()
conll03_entities = set()

# split above loop into 3 loops
for demo in ace05:
    ace05_entities = ace05_entities.union(set([entity['type'] for entity in demo['entities']]))
for demo in ace04:
    ace04_entities = ace04_entities.union(set([entity['type'] for entity in demo['entities']]))
for demo in conll03:
    conll03_entities = conll03_entities.union(set([entity['type'] for entity in demo['entities']]))

# convert to list and sort
ace05_entities = list(ace05_entities)
ace04_entities = list(ace04_entities)
conll03_entities = list(conll03_entities)
ace05_entities.sort()
ace04_entities.sort()
conll03_entities.sort()
    
print(ace05_entities)
print(ace04_entities)
print(conll03_entities)