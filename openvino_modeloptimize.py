import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model,model_from_json
import os
#import keras.losses
from keras import backend as K

smooth=1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#keras.losses.custom_loss=dice_coef_loss

# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = '/Users/pnagula/Downloads/'
model_fname = '/Users/pnagula/Downloads/Usecases_Code/Image_Segmentation/nddcheckpoint-46.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

#model = load_model(model_fname)

# the below line is when we have complete model
model = load_model(model_fname,custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef':dice_coef})

session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

# Run intel openvino Model optimizer program ; inputs - frozen tensorflow graph  output - optimized model in xml and bin files

os.system("docker run -v /Users/pnagula/Downloads/:/workspace/ openvino_gpdb:latest /bin/bash -c 'source /opt/intel/openvino/bin/setupvars.sh; python3.7 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /workspace/frozen_model.pb --output_dir /workspace/ --input_shape [1,512,512,3] --data_type FP16'")
