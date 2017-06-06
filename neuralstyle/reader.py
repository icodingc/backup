import numpy as np
import random
import os,sys
from PIL import Image

def sample_patch(x,y,image_size):
	image_x = Image.open(x).convert('RGB')
	image_y = Image.open(y).convert('RGB')
	start = random.choice(range(image_x.size[0]-image_size+1))
	image_x=image_x.crop((start,start,start+image_size,start+image_size))
	image_y=image_y.crop((start,start,start+image_size,start+image_size))
	return np.asarray(image_x).astype(np.float32),np.asarray(image_y).astype(np.float32)

def image(image_path,style_path,epoch=1,shuffle=True):
	filenames = []
	for f in os.listdir(image_path):
		filenames.append(f)
	if shuffle:
		names = []
		for i in xrange(epoch):names += np.random.permutation(filenames).tolist()
	else:names = filenames*epoch

	for i,pic in enumerate(names):
		sz = Image.open(os.path.join(image_path,pic)).size[0]
		if sz >=512:image_size = 512
		elif sz >=256:image_size = 256
		else:continue
		yield sample_patch(os.path.join(image_path,pic),
				os.path.join(style_path,pic),256)
if __name__ == "__main__":
	image_path = '/home/ldap/fengfangxiang/pub/prisma/Udnie/images_in'
	style_path = '/home/ldap/fengfangxiang/pub/prisma/Udnie/images_out'
	for i,(a,b) in enumerate(image(image_path,style_path)):
		print i,"-->",a.shape,b.shape
