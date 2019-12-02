from PIL import Image
import numpy as np

im = np.array(Image.open('/home/vinita/Documents/Neural_Networks/stargan_final/code/test1.jpg'))

print(im.dtype)
# uint8

print(im.ndim)
# 3

print(im.shape)

arr1 = im.reshape(-1, 128, 128, 3)

for n in np.arange(10):
	img_nm = 'img'+str(n)+'.jpg'
	img = arr1[n,:,:,:]
	img_pic = Image.fromarray(img)
	img_pic.save(img_nm)
