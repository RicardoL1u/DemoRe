import json
import faiss
import numpy as np
import logging
logging.basicConfig(
        format="[demo_retriever:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s"
)
logging.getLogger().setLevel(logging.INFO)
class DemoRetriever(object):
    def __init__(self,demo_embs:np.array,demos:list,device:str='cpu'):
        """
        demo_embs: (n_demo, dim)
        demos: list of demos
        """

        self.demo_embs = demo_embs
        self.demos = demos
        assert len(self.demo_embs) == len(self.demos), 'demo_embs and demos should have the same length'
        logging.info('Building index...')
        self.cpu_index_falt = faiss.IndexFlatL2(self.demo_embs.shape[1])
        if 'cuda' in device:
            logging.info('Using GPU...')
            if device == 'cuda':
                logging.info('Multiple GPUs detected, using cuda:0')
                device = 'cuda:0'
            device_id = int(device.split(':')[-1])
            self.gpu_index_falt = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), device_id, self.cpu_index_falt)
        self.index_falt = self.gpu_index_falt if 'cuda' in device else self.cpu_index_falt
        self.index_falt.add(self.demo_embs)
        logging.info('Done.')
        
    def retrieve(self,query_emb:np.array,topk:int=5)->list:
        """
        query_emb: (dim,)
        """
        D, I = self.index_falt.search(query_emb.reshape(1,-1), topk) 
        # retrieve demos
        retrieved_demos = [self.demos[i] for i in I[0]]
        
        return retrieved_demos