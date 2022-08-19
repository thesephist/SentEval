# To be placed within SentEval/examples
#
# Extra pip requirements: sentencepiece, transformers, torch

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import logging
import json
import numpy as np

from sentence_transformers import SentenceTransformer

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
MODEL_NAME = 'sentence-transformers/all-roberta-large-v1'
# MODEL_NAME = 'sentence-transformers/sentence-t5-base'
SAVE_PATH = 'roberta-large.json'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# SBERT prepare and batcher
def prepare(params, samples):
    params.model = SentenceTransformer(MODEL_NAME)
    params.batch_size = 16
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params.model.encode(batch,
            batch_size=16,
            show_progress_bar=False,
            device='cuda')
    return embeddings

# Set params for SBERT

# test config
test_params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
test_params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 2,
                                 'tenacity': 3, 'epoch_size': 2}
# eval config
eval_params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
eval_params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(eval_params, batcher, prepare)
    transfer_tasks = [
            # standard semantic textual similarity
            'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness',
            # sentiment analysis
            'SST2', 'SST5',
            # other NLI
            'TREC', 'SNLI', 'SICKEntailment', 'MRPC',
            # probing tasks
            'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
            'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    with open(SAVE_PATH, 'w+') as f:
        def serializeNdArray(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
        json.dump(results, f, default=serializeNdArray)

