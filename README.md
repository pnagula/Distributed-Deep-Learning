   # Distributed-Deep-Learning
 
   ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/DDL.jpg)
   
   ### Conceptually, the data-parallel distributed training paradigm is straightforward:

   1. Run multiple copies of the training script and each copy:
      1. Reads a chunk of the data
      1. Runs it through the model
      1. Computes model updates (gradients)

   1. Average gradients among those multiple copies

   1. Update the model

   1. Repeat (from Step 1.i)
   
   
   ### Horovod, MPI, Keras/Tensorflow Distributed Architeture 
   ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/MPI_Horovod1.jpeg)
   
   ## Code changes to support Horovod distributed training
   
   ### Optimizer changes
   ``` python
    opt = keras.optimizers.Adam(.00013 * hvd.size())

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)
    
    model.compile(optimizer=opt)
  ```  
   ### Initialisation
  ``` python
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 10
    config.inter_op_parallelism_threads =  1
    K.set_session(tf.Session(config=config))
  ```  
   ### Resuming a distributed training
   ``` python
    resume_from_epoch=int(sys.argv[1])
    print('resume_from_epoch:',resume_from_epoch)
    # resume from latest checkpoint file
    resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')
    
    verbose = 1 if hvd.rank() == 0 else 0
    
    if resume_from_epoch > 0 and hvd.rank() == 0:
       model = hvd.load_model('/workspace/nddcheckpoint-{epoch}.h5'.format(epoch=resume_from_epoch),custom_objects={'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss}) 
    else:
       model = get_unet()
   ```     
   ### Data Distribution to each worker node

    if hvd.size() > 1:
       number_of_examples_per_rank=imgs_train.shape[0]//hvd.size()
       remainder=imgs_train.shape[0]%hvd.size()
    if hvd.rank() < remainder:
       start_index= hvd.rank() * (number_of_examples_per_rank+1)
       end_index= start_index + number_of_examples_per_rank + 1
    else:
       start_index= hvd.rank() * number_of_examples_per_rank + remainder
       end_index= start_index + number_of_examples_per_rank 
    print('Rank''s, Start and End Index:',hvd.rank(),start_index,end_index)

   * start_index and end_index contains start and end index of images in training numpy array for each worker node.
      * e.g:- Let's say imgs_train.shape[0] = 5000 and worker nodes=10, 
         * worker node 1 , start_index=0, end_index=499
         * worker node 2 , start_index=500, end_index=999
         * worker node 10, start_index=4500, end_index=4999
         
   ### Callbacks
   ``` python
   callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]
    if hvd.rank() == 0:
       callbacks.append(keras.callbacks.ModelCheckpoint('/workspace/nddcheckpoint-{epoch}.h5',monitor='val_loss', save_best_only=True))
   ```    
   ### Keras fit method changes for distributed training
   ```python 
   model.fit(imgs_train[start_index:end_index], imgs_mask_train[start_index:end_index], batch_size=12,              
              epochs=resume_from_epoch+10,  shuffle=True, 
              validation_split=0.01,initial_epoch=resume_from_epoch, 
              callbacks=callbacks, 
              verbose=1 if hvd.rank() == 0 else 0)
   ```        
   ### Saving the model onto master node
   ``` python
   if hvd.rank() == 0:
       model.save('/workspace/unetmodelfdd.h5', include_optimizer=False)
   ```    
