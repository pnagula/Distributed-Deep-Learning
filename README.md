   # Distributed-Deep-Learning
 
   ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/DDL.jpg)
   
   ### Conceptually, the data-parallel distributed training paradigm is straightforward:

   1. Run multiple copies of the training script and each copy:
      1. Reads a chunk of the data
      1. Runs it through the model
      1. Computes model updates (gradients)

   1. Average gradients among those multiple copies

   1. Update the model

   1. Repeat (from Step 2.i)


