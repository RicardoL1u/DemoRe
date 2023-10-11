from DemoRe.emb_builder import EmbeddingBuilder
emb_builder = EmbeddingBuilder('sentence-transformers/all-mpnet-base-v2')
from DemoRe.text_modifier import EntityAnonymizer
# initialize the anonymizer with NER model
anonymizer = EntityAnonymizer('/data/lyt/models/flair/ner-english-ontonotes-large/pytorch_model.bin')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

import os
import json
import numpy as np
from tqdm import tqdm
# dataset list
dataset_name_list = ['ACE 2005','ACE 2004','CoNLL 2003']

datapath = 'ALL_DATASETS_NEW/NER'
# build embedding
for dataset_name in dataset_name_list:
    logging.info('Building embedding for dataset: {}'.format(dataset_name))
    dataset = json.load(open(os.path.join(datapath, dataset_name + '/train.json')))
    logging.info('Training dataset size: {}'.format(len(dataset)))
    
    sentence_embs = []
    anonymized_sentence_embs = []
    
    for unit in tqdm(dataset, desc='Building embedding for dataset: {}'.format(dataset_name)):
        sentence = unit['sentence']
        anonymized_sentence = anonymizer.anonymize(sentence)
        
        sentence_embs.append(emb_builder.get_mean_pooling_embedding(sentence))
        anonymized_sentence_embs.append(emb_builder.get_mean_pooling_embedding(anonymized_sentence))
    
    sentence_embs = np.concatenate(sentence_embs, axis=0)
    anonymized_sentence_embs = np.concatenate(anonymized_sentence_embs, axis=0)
    with open(os.path.join(datapath, dataset_name + '/train_sentence_embs.npy'),'wb') as f:
        np.save(f, sentence_embs)
    with open(os.path.join(datapath, dataset_name + '/train_anonymized_sentence_embs.npy'),'wb') as f:  
        np.save(f, anonymized_sentence_embs)