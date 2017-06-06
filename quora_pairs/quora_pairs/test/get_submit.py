#coding=utf-8
import numpy as np
import pandas as pd

test_ids = np.load('data/test_ids.npy')
print(test_ids.shape)

preds = np.load('data/preds.npy')
print(preds.shape)

assert len(test_ids)==len(preds)

submission = pd.DataFrame({'is_duplicate':preds.ravel(),'test_id':test_ids})
submission.to_csv('cnn_fc_second_2.csv', index=False)
