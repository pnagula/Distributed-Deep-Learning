   # Distributed-Deep-Learning
 
   ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/DDL.jpg)
   
   ##### Conceptually, the data-parallel distributed training paradigm is straightforward:

   ##### Run multiple copies of the training script and each copy:
   ###### a) reads a chunk of the data
   ###### b) runs it through the model
   ###### c) computes model updates (gradients)

   ###### Average gradients among those multiple copies

   ###### Update the model

   ###### Repeat (from Step 1a)


