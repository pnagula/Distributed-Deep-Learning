drop function if exists img_inference(text);
CREATE OR REPLACE FUNCTION img_inference(filename text) RETURNS integer AS $$
#container: openvino
    import sys
    import os
    from openvino.inference_engine import IENetwork, IEPlugin
    import numpy as np
    from skimage.transform import resize
    from skimage.io import imread,imsave

    # load the intel optimized model for inference
    model_xml = '/workspace/model/frozen_model.xml'
    model_bin = '/workspace/model/frozen_model.bin'
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

    #fileName = '/workspace/Usecases_Code/Image_Segmentation/training_512/0cdf5b5d0ce1_01.jpg'
    img = imread(filename)
    imgresize = resize(img, (512,512), preserve_range=True)
    imgresize=np.array(imgresize)
    #print(imgresize.shape)
    imgresize=imgresize/255.
    n, c, h, w = [1, 3, 512, 512]
    imgresize = imgresize.transpose((2, 0, 1))
    imgresize=  imgresize.reshape((n, c, h, w))

    # run IE inferene
    res = exec_net.infer(inputs={input_blob: imgresize})
    output_node_name = list(res.keys())[0]
    #print(res[output_node_name].shape)
    fout=res[output_node_name].reshape((n,h,w,1))
    #print(fout.shape)

    # set bitmask to 255 for the pixels that are predicted as 1
    image = (fout > 0.5).astype(np.uint8)
    image1 = (fout * 255.).astype(np.uint8)
    image1=image1.reshape((512,512))
    plpy.info('writing image...')
    # save the image
    strg=filename[16:filename.index('.')]
    plpy.info(strg)
    imsave('/workspace/output/'+strg+'_predict.jpg',image1)
    return 1
$$ language plcontainer ;

drop table if exists openvino_input;
create table openvino_input
(
filename text
);
copy openvino_input from '/home/gpadmin/openvino/inputfiles.txt' ;

select * from plcontainer_refresh_config
select img_inference(filename) from openvino_input;



drop function if exists img_keras_inference(text);
CREATE OR REPLACE FUNCTION img_keras_inference(filename text) RETURNS integer AS $$
#container: openvino
    from keras.models import Model
    from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation
    from keras.optimizers import Adam

    import sys
    import os
    import keras
    from keras.models import load_model
    import numpy as np
    from skimage.transform import resize
    from skimage.io import imread,imsave
    from keras import backend as K
    
    smooth = 1.



    def dice_coef(y_true, y_pred):
	    y_true_f = K.flatten(y_true)
	    y_pred_f = K.flatten(y_pred)
	    intersection = K.sum(y_true_f * y_pred_f)
	    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    def get_unet():
	    inputs = Input((512, 512, 3))
	    
	    convp0 = Conv2D(8, (3, 3),  padding='same')(inputs)
	    convp0 = BatchNormalization(momentum=0.99)(convp0)
	    convp0 = Activation('relu')(convp0)
	    convp0 = Conv2D(8, (3, 3),  padding='same')(convp0)
	    convp0 = BatchNormalization(momentum=0.99)(convp0)
	    convp0 = Activation('relu')(convp0)
	    poolp0 = MaxPooling2D(pool_size=(2, 2))(convp0)

	    conv0 = Conv2D(16, (3, 3),  padding='same')(poolp0)
	    conv0 = BatchNormalization(momentum=0.99)(conv0)
	    conv0 = Activation('relu')(conv0)
	    conv0 = Conv2D(16, (3, 3),  padding='same')(conv0)
	    conv0 = BatchNormalization(momentum=0.99)(conv0)
	    conv0 = Activation('relu')(conv0)
	    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

	    conv1 = Conv2D(32, (3, 3),  padding='same')(pool0)
	    conv1 = BatchNormalization(momentum=0.99)(conv1)
	    conv1 = Activation('relu')(conv1)
	    conv1 = Conv2D(32, (3, 3),  padding='same')(conv1)
	    conv1 = BatchNormalization(momentum=0.99)(conv1)
	    conv1 = Activation('relu')(conv1)
	    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	    conv2 = Conv2D(64, (3, 3),  padding='same')(pool1)
	    conv2 = BatchNormalization(momentum=0.99)(conv2)
	    conv2 = Activation('relu')(conv2)
	    conv2 = Conv2D(64, (3, 3),  padding='same')(conv2)
	    conv2 = BatchNormalization(momentum=0.99)(conv2)
	    conv2 = Activation('relu')(conv2)
	    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	    conv3 = Conv2D(128, (3, 3),  padding='same')(pool2)
	    conv3 = BatchNormalization(momentum=0.99)(conv3)
	    conv3 = Activation('relu')(conv3)
	    conv3 = Conv2D(128, (3, 3),  padding='same')(conv3)
	    conv3 = BatchNormalization(momentum=0.99)(conv3)
	    conv3 = Activation('relu')(conv3)
	    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	    conv4 = Conv2D(256, (3, 3),  padding='same')(pool3)
	    conv4 = BatchNormalization(momentum=0.99)(conv4)
	    conv4 = Activation('relu')(conv4)
	    conv4 = Conv2D(256, (3, 3),  padding='same')(conv4)
	    conv4 = BatchNormalization(momentum=0.99)(conv4)
	    conv4 = Activation('relu')(conv4)
	    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	    conv5 = Conv2D(512, (3, 3),  padding='same')(pool4)
	    conv5 = BatchNormalization(momentum=0.99)(conv5)
	    conv5 = Activation('relu')(conv5)
	    conv5 = Conv2D(512, (3, 3),  padding='same')(conv5)
	    conv5 = BatchNormalization(momentum=0.99)(conv5)
	    conv5 = Activation('relu')(conv5)
	    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
	   
	    convc = Conv2D(1024, (3, 3),  padding='same')(pool5)
	    convc = BatchNormalization(momentum=0.99)(convc)
	    convc = Activation('relu')(convc)
	    convc = Conv2D(1024, (3, 3),  padding='same')(convc)
	    convc = BatchNormalization(momentum=0.99)(convc)
	    convc = Activation('relu')(convc)
	    
	    convct = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(convc) 
	    convct = BatchNormalization(momentum=0.99)(convct)
	    convct = Activation('relu')(convct)
	    convct = concatenate([convct,conv5],axis=3)
	    convct = Conv2D(512, (3, 3),  padding='same')(convct)
	    convct = BatchNormalization(momentum=0.99)(convct)
	    convct = Activation('relu')(convct)
	    convct = Conv2D(512, (3, 3),  padding='same')(convct)
	    convct = BatchNormalization(momentum=0.99)(convct)
	    convct = Activation('relu')(convct)
	    
	    conv6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(convct) 
	    conv6 = BatchNormalization(momentum=0.99)(conv6)
	    conv6 = Activation('relu')(conv6)
	    conv6 = concatenate([conv6,conv4],axis=3)
	    conv6 = Conv2D(256, (3, 3),  padding='same')(conv6)
	    conv6 = BatchNormalization(momentum=0.99)(conv6)
	    conv6 = Activation('relu')(conv6)
	    conv6 = Conv2D(256, (3, 3),  padding='same')(conv6)
	    conv6 = BatchNormalization(momentum=0.99)(conv6)
	    conv6 = Activation('relu')(conv6)
	    
	    conv7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
	    conv7 = BatchNormalization(momentum=0.99)(conv7)
	    conv7 = Activation('relu')(conv7)
	    conv7 = concatenate([conv7, conv3], axis=3)
	    conv7 = Conv2D(128, (3, 3),  padding='same')(conv7)
	    conv7 = BatchNormalization(momentum=0.99)(conv7)
	    conv7 = Activation('relu')(conv7)
	    conv7 = Conv2D(128, (3, 3),  padding='same')(conv7)
	    conv7 = BatchNormalization(momentum=0.99)(conv7)
	    conv7 = Activation('relu')(conv7)

	    conv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
	    conv8 = BatchNormalization(momentum=0.99)(conv8)
	    conv8 = Activation('relu')(conv8)
	    conv8 = concatenate([conv8, conv2], axis=3)
	    conv8 = Conv2D(64, (3, 3),  padding='same')(conv8)
	    conv8 = BatchNormalization(momentum=0.99)(conv8)
	    conv8 = Activation('relu')(conv8)
	    conv8 = Conv2D(64, (3, 3),  padding='same')(conv8)
	    conv8 = BatchNormalization(momentum=0.99)(conv8)
	    conv8 = Activation('relu')(conv8)

	    conv9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
	    conv9 = BatchNormalization(momentum=0.99)(conv9)
	    conv9 = Activation('relu')(conv9)
	    conv9 = concatenate([conv9, conv1], axis=3)
	    conv9 = Conv2D(32, (3, 3),  padding='same')(conv9)
	    conv9 = BatchNormalization(momentum=0.99)(conv9)
	    conv9 = Activation('relu')(conv9)
	    conv9 = Conv2D(32, (3, 3),  padding='same')(conv9)
	    conv9 = BatchNormalization(momentum=0.99)(conv9)
	    conv9 = Activation('relu')(conv9)

	    conv10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv9)
	    conv10 = BatchNormalization(momentum=0.99)(conv10)
	    conv10 = Activation('relu')(conv10)
	    conv10 = concatenate([conv10, conv0], axis=3)
	    conv10 = Conv2D(16, (3, 3),  padding='same')(conv10)
	    conv10 = BatchNormalization(momentum=0.99)(conv10)
	    conv10 = Activation('relu')(conv10)
	    conv10 = Conv2D(16, (3, 3),  padding='same')(conv10)
	    conv10 = BatchNormalization(momentum=0.99)(conv10)
	    conv10 = Activation('relu')(conv10)

	    conv11 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv10)
	    conv11 = BatchNormalization(momentum=0.99)(conv11)
	    conv11 = Activation('relu')(conv11)
	    conv11 = concatenate([conv11, convp0], axis=3)
	    conv11 = Conv2D(8, (3, 3),  padding='same')(conv11)
	    conv11 = BatchNormalization(momentum=0.99)(conv11)
	    conv11 = Activation('relu')(conv11)
	    conv11 = Conv2D(8, (3, 3),  padding='same')(conv11)
	    conv11 = BatchNormalization(momentum=0.99)(conv11)
	    conv11 = Activation('relu')(conv11)

	    conv12 = Conv2D(1, 1, 1, activation='sigmoid')(conv11)

	    model = Model(inputs=[inputs], outputs=[conv12])

	    opt = keras.optimizers.Adam(.00013 * 8)

	    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])

	    return model

    model = get_unet()
    
    # load keras model
    model.load_weights('/workspace/model/nddcheckpoint-46.h5')
    #,custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss})

    #preprocess image
    #    resize to 512,512
    #    normalize it to 0-1
    #    transpose and reshape image channel format as openvino IE engine requires it as n,c,h,w

    img = imread(filename)
    imgresize = resize(img, (512,512), preserve_range=True)
    imgresize=np.array(imgresize)
    imginput=np.ndarray((1,512,512,3))
    imginput[0]=imgresize
    imginput=imginput/255.
    imgs_mask_test = model.predict(imginput, verbose=1) 
    
    image = (imgs_mask_test > 0.5).astype(np.uint8)
    image1 = (imgs_mask_test * 255.).astype(np.uint8)
    image1=image1.reshape((512,512))
    plpy.info('writing image...')
    # save the image
    strg=filename[16:filename.index('.')]
    plpy.info(strg)
    imsave('/workspace/output/'+strg+'_keras_predict.jpg',image1)
    return 1
$$ language plcontainer ;


select img_keras_inference(filename) from openvino_input;