#coding=utf-8
import numpy as np
from sklearn.utils import shuffle

import configuration
############
# Read data
#############
BASE_DIR='./data/test/'
test_seq1 = np.load(BASE_DIR+'seq1.npy')
test_seq2 = np.load(BASE_DIR+'seq2.npy')
test_seq1_len = np.load(BASE_DIR+'seq1_len.npy')
test_seq2_len = np.load(BASE_DIR+'seq2_len.npy')
test_ids = np.load(BASE_DIR+'test_ids.npy')

print(2345796)
print("test seq1",len(test_seq1))
print("test seq2",len(test_seq2))
print("test seq1_len",len(test_seq1_len))
print("test seq2_len",len(test_seq1_len))
print("test ids",len(test_ids))

#############
#testing  (2345796 + 4 == 2345800)
#############
# 2345796 % (612)==0
#############
num_examples = len(test_seq1)
BATCH_SIZE = 612
def batch_inputs():
  for beg in xrange(0,num_examples,BATCH_SIZE):
    end=beg+BATCH_SIZE
    seqs = np.vstack([test_seq1[beg:end],test_seq2[beg:end]])
    # length to mask?
    mask_len = np.hstack([test_seq1_len[beg:end],test_seq2_len[beg:end]])
    masks = np.ones_like(seqs,dtype=np.int32)
    for i in xrange(len(mask_len)):
      masks[i][mask_len[i]:]=0
    lbs = test_ids[beg:end]
    
    yield (seqs,masks,lbs)
if __name__ == '__main__':
  print("hello")
