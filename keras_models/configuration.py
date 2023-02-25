# variables NOT for optimizing
# FILENAME = 'final_annotation_cnn.csv'
FILENAME = './data/Essays/essays2007.csv'  # essays dataset
PATH_EMBEDDING = '../fast-text/corpus.friends+nyt+wiki+amazon.fasttext.skip.d100.bin'
dims = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']
column_to_read = ['text', 'single_text'][0]
validation_mode = [0, 1, 2][1]  # 0 for splitting; 1 for 10-fold CV; 2 for pure testing
multilabel = False  # False means multi class
''' 
models = MLP, textCNN, CNN, BLSTM, ABLSTM, ABCNN
'''
ModelName = 'BLSTM2'

MAX_NUM_WORDS = 10000  # truncated vocab_size
MAX_SEQ_LENGTH = 400
MAX_SENTS = 30
EMBEDDING_DIM = 100
ClassNum = 5 if multilabel else 2
kfolds = 2  # default 10


class Params:

    # fairly fixed variables
    # FILENAME = 'data.csv'
    # MAX_NUM_WORDS = 10000
    # MAX_SEQ_LENGTH = 500

    # variables for optimization
    n_epoch = 2  # default 30
    batch_size = 32
    dropout_rate = 0.5
    dense_layer_size = 256
    use_word_embedding = False
    train_embedding = True
    use_merge = True

    # TODO: BLSTM parameters
    merge_mode = ['mul', 'concat'][1]

    # TODO: Conv parameters
    filter_size = [3, 4, 5]
    num_filter = [100, 100, 100]
    pool_size = [3, 3, 3]

    # def setParams(self, dct):
    # 	if 'embeddings_dim' in dct:
    # 		self.embeddings_dim = dct['embeddings_dim']
    # 	if 'dense_layer_size' in dct:
    # 		self.dense_layer_size = dct['dense_layer_size']
