# DemoRe

This repository contains the code for the demo selection of In-Context Learning

## Installation
pip install git+https://github.com/RicardoL1u/DemoRe.git

## Demo Retriever
```python
from DemoRe.demo_retriever import DemoRetriever
# initialize the retriever
# with demo in list and the corresponding embeddings
retriever = DemoRetriever(demo_list, demo_embeddings)
```


## emb builder
we provide a simple emb builder for you to build the embeddings for your demos

```python
from DemoRe.emb_builder import EmbeddingBuilder
# provide the path or the name of the model
# used to build the embeddings
model_name_or_path = "bert-base-uncased"
builder = EmbeddingBuilder(model_name_or_path)
```
For different models, different sentences embedding methods are used. For example, for BERT, we use the embedding of the first token [CLS] as the sentence embedding. For MPNet, the mean of the embeddings of all tokens is used as the sentence embedding. For more details, please refer to the code.

## Text Modifier
we provide a simple text modifier for you to modify the text of your demos

For example, you can use EntityAnonymizer to anonymize the entities in your demos

```python
from DemoRe.text_modifier import EntityAnonymizer
# initialize the anonymizer with NER model
anonymizer = EntityAnonymizer('/data/lyt/models/flair/ner-english-ontonotes-large/pytorch_model.bin')
print(anonymizer.anonymize('George Washington went to Washington .'))
```
The ner model can be downloaded from [here](https://github.com/flairNLP/flair/#state-of-the-art-models)
