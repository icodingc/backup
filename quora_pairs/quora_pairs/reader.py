#coding=utf-8
import numpy as np
from sklearn.utils import shuffle

import configuration
############
# Read data
#############
BASE_DIR='./data/train/'
data_1 = np.load(BASE_DIR+'seq1.npy')
data_2 = np.load(BASE_DIR+'seq2.npy')
data_1_len = np.load(BASE_DIR+'seq1_len.npy')
data_2_len = np.load(BASE_DIR+'seq2_len.npy')
feature = np.load(BASE_DIR+'feature_sec.npy')   # other feature
labels = np.load(BASE_DIR+'labels.npy')


# Randomly shuffle data
shuffle_indices = np.random.permutation(np.arange(len(labels)))
data_1_shuffled = data_1[shuffle_indices]
data_2_shuffled = data_2[shuffle_indices]
data_1_len_shuffled = data_1_len[shuffle_indices]
data_2_len_shuffled = data_2_len[shuffle_indices]
feature_shuffled = feature[shuffle_indices]
labels_shuffled = labels[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(0.1 * float(len(labels_shuffled)))
train_seq1, dev_seq1 = data_1_shuffled[:dev_sample_index], data_1_shuffled[dev_sample_index:]
train_seq2, dev_seq2 = data_2_shuffled[:dev_sample_index], data_2_shuffled[dev_sample_index:]
train_seq1_len, dev_seq1_len = data_1_len_shuffled[:dev_sample_index], data_1_len_shuffled[dev_sample_index:]
train_seq2_len, dev_seq2_len = data_2_len_shuffled[:dev_sample_index], data_2_len_shuffled[dev_sample_index:]
train_labels, dev_labels = labels_shuffled[:dev_sample_index], labels_shuffled[dev_sample_index:]
train_feat,dev_feat = feature_shuffled[:dev_sample_index],feature_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(train_labels), len(dev_labels)))


modelconfig = configuration.ModelConfig()
#############
#Validation 
#############
class Test_batch(object):
  def __init__(self,batch_size = modelconfig.batch_size):
    self.num_examples = len(dev_seq1)
    self.batch_size = batch_size
    self.shuf()
  def shuf(self):
    self.cnt=0
    self.feats =shuffle(dev_seq1,dev_seq1_len,
                            dev_seq2,
                            dev_seq2_len,
                            dev_feat,
                            dev_labels)
    self.data1=self.feats[0]
    self.len1=self.feats[1]
    self.data2=self.feats[2]
    self.len2=self.feats[3]
    self.feat=self.feats[4]
    self.lab=self.feats[5]
  def next(self):
    if self.cnt + self.batch_size > self.num_examples:
      self.shuf()
    beg = self.cnt
    end = beg + self.batch_size
    seqs = np.vstack([self.data1[beg:end],self.data2[beg:end]])
    # length to mask?
    mask_len = np.hstack([self.len1[beg:end],self.len2[beg:end]])
    masks = np.ones_like(seqs,dtype=np.int32)
    for i in xrange(len(mask_len)):
      masks[i][mask_len[i]:]=0
    fts = self.feat[beg:end]
    lbs = self.lab[beg:end]
    self.cnt = end
    return (seqs,masks,lbs)
#############
#training
#############


def batch_inputs():
  """
  Output:
  seqs [batch*2,length]
  masks[batch*2,length]
  fts  [batch,28]
  lbs  [batch,]
  """
  num_examples = len(train_seq1)
  while True:
    data1,len1,data2,len2,feat,lab=shuffle(train_seq1,
                                            train_seq1_len,
                                            train_seq2,
                                            train_seq2_len,
                                            train_feat,
                                            train_labels)
    for beg in xrange(0,num_examples,modelconfig.batch_size):
      end=beg+modelconfig.batch_size
      if end>num_examples:break
      seqs = np.vstack([data1[beg:end],data2[beg:end]])
      # length to mask?
      mask_len = np.hstack([len1[beg:end],len2[beg:end]])
      masks = np.ones_like(seqs,dtype=np.int32)
      for i in xrange(len(mask_len)):
        masks[i][mask_len[i]:]=0

      fts = feat[beg:end]
      lbs = lab[beg:end]
      yield (seqs,masks,lbs)

########竟然有空串
def test(step=100):
  st=0
  for ft,ls,_lab in batch_inputs():
    print(ft.shape)
    print(ls.shape)
    print(lab.shape)
    st=st+1
    if st==step:
        print(ft[0])
        print(ls[0])
        print(lab[0])
        break

  print("OK")
if __name__ == '__main__':
  test()
