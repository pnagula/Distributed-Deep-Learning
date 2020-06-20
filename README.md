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
   ### Keras fit method changes for distributed training
   ```python 
   model.fit(imgs_train[start_index:end_index], imgs_mask_train[start_index:end_index], batch_size=12,              
              epochs=resume_from_epoch+10,  shuffle=True, 
              validation_split=0.01,initial_epoch=resume_from_epoch, 
              callbacks=callbacks, 
              verbose=1 if hvd.rank() == 0 else 0)
   ```        
