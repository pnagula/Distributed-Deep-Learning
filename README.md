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
   ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/DDL.jpg)
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

   * Here, number_of examples per rank 
