   # Distributed-Deep-Learning
 
   ![](https://github.com/pnagula/Distributed-Deep-Learning/blob/master/DDL.jpg)
   
   ##### 1. Conceptually, the data-parallel distributed training paradigm is straightforward:

   ##### 2. Run multiple copies of the training script and each copy:
   #####  2.1) Reads a chunk of the data
   #####  2.2) Runs it through the model
   #####  2.3) Computes model updates (gradients)

   ##### 3. Average gradients among those multiple copies

   ##### 4. Update the model

   ##### 5. Repeat (from Step 1a)


