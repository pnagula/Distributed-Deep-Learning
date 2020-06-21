import sys
import os
#print(os.path.exists("/opt/intel/openvino/bin/setupvars.sh"))
import subprocess
exec(open("/opt/intel/openvino/bin/setupvars.sh").read())
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
from skimage.transform import resize
from skimage.io import imread,imsave
import sys

# load the intel optimized model for inference
model_xml = sys.argv[1]
model_bin = sys.argv[2]
plugin = IEPlugin("CPU", plugin_dirs=None)

# Build Inference Engine network using xml and bin files
net = IENetwork(model=model_xml, weights=model_bin)
# Establish input and output blobs (images)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = plugin.load(network=net)
del net

#preprocess image 
#    resize to 512,512 
#    normalize it to 0-1
#    transpose and reshape image channel format as openvino IE engine requires it as n,c,h,w
 
fileName = '/workspace/Usecases_Code/Image_Segmentation/training_512/0cdf5b5d0ce1_01.jpg'
img = imread(fileName)
imgresize = resize(img, (512,512), preserve_range=True)
imgresize=np.array(imgresize)
print(imgresize.shape)
imgresize=imgresize/255.
n, c, h, w = [1, 3, 512, 512]
imgresize = imgresize.transpose((2, 0, 1))
imgresize=  imgresize.reshape((n, c, h, w))

# run IE inferene 
res = exec_net.infer(inputs={input_blob: imgresize})
output_node_name = list(res.keys())[0]
print(res[output_node_name].shape)
fout=res[output_node_name].reshape((n,h,w,1))
print(fout.shape)

# set bitmask to 255 for the pixels that are predicted as 1 
image = (fout > 0.5).astype(np.uint8)
image1 = (fout * 255.).astype(np.uint8)
image1=image1.reshape((512,512))
print('writing image...')
# save the image
imsave('/workspace/image_512_pmask.jpg',image1)
