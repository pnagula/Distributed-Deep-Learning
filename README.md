   # Distributed-Deep-Learning
 
   ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/DDL.jpg)
   
   ## Conceptually, the data-parallel distributed training paradigm is straightforward:

   1. Run multiple copies of the training script and each copy:
      1. Reads a chunk of the data
      1. Runs it through the model
      1. Computes model updates (gradients)

   1. Average gradients among those multiple copies

   1. Update the model

   1. Repeat (from Step 1.i)
   
   
   ## Horovod, MPI, Keras/Tensorflow Distributed Architeture 
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
   ## Performance comparison metrics between distributed vs non-distributed training
   
  ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/PM1.jpg)
  
  ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/PM2.jpg)
  
  ## Build Docker image
  * docker build -t hvd_tf_unet:1.0 -f Dockerfile_Horovod_MPI_Keras_TF.txt . 
  ## How to Start Training
  
  * Run following docker command on worker nodes first
  
      * docker run --name slave_node -v   <storage_path>/pivotal:/workspace  -v /root/.ssh:/root/.ssh  --network=host  --privileged --rm  hvd_tf_unet:1.0 /bin/bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
      
  * Run following docker command on master node
      
      * Keep the below command in a shell script – run_unet.sh
         * HOROVOD_FUSION_THRESHOLD=134217728 mpirun -np 8 --map-by ppr:2:socket:pe=10 --allow-run-as-root --mca plm_rsh_args "-p 12345" -mca btl_tcp_if_include bond0.123 -mca btl ^openib -mca pml ob1 -H 192.168.116.103:9999,192.168.116.104:9999  --oversubscribe --report-bindings -x LD_LIBRARY_PATH -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS=9 -x KMP_BLOCKTIME=1 -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 python3 -u   /workspace/unet/UnetImgseg_Horovod_DD.py 40 &> /workspace/unet/unetlog_DD.log  
      
      * docker run --name master_node -v /<storage_path>pivotal:/workspace -v /root/.ssh:/root/.ssh --network=host  --privileged --rm  hvd_tf_unet:1.0 /bin/bash -c "source /workspace/run_unet.sh"

# Distributed Inference using OpenVINO and Greenplum
## OpenVINO
OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNNs), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The OpenVINO™ toolkit includes the Deep Learning Deployment Toolkit (DLDT).

OpenVINO™ toolkit:

* Enables CNN-based deep learning inference on the edge
* Supports heterogeneous execution across an Intel® CPU, Intel® Integrated Graphics, Intel® FPGA, Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2 and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
* Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
* Includes optimized calls for computer vision standards, including OpenCV* and OpenCL

## Greenplum

Greenplum Database is a massively parallel processing (MPP) database server with an architecture specially designed to manage large-scale analytic data warehouses and business intelligence workloads.

MPP (also known as a shared nothing architecture) refers to systems with two or more processors that cooperate to carry out an operation, each processor with its own memory, operating system and disks. Greenplum uses this high-performance system architecture to distribute the load of multi-terabyte data warehouses, and can use all of a system's resources in parallel to process a query.
