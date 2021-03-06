

HOROVOD_FUSION_THRESHOLD=134217728 mpirun -np 8 --map-by ppr:2:socket:pe=10 --allow-run-as-root --mca plm_rsh_args "-p 12345" -mca btl_tcp_if_include bond0.123 -mca btl ^openib -mca pml ob1 -H 192.168.116.103:9999,192.168.116.104:9999  --oversubscribe --report-bindings -x LD_LIBRARY_PATH -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS=9 -x KMP_BLOCKTIME=1 -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 python3 -u   /workspace/unet/UnetImgseg_Horovod_DD.py 40 &> /workspace/unet/unetlog_DD.log 

#HOROVOD_FUSION_THRESHOLD=134217728 mpirun -np 2 --map-by node:pe=40 --allow-run-as-root --mca plm_rsh_args "-p 12345" -mca btl_tcp_if_include bond0.123 -mca btl ^openib -mca pml ob1 -H 192.168.116.103:9999,192.168.116.104:9999  --oversubscribe --report-bindings -x LD_LIBRARY_PATH -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS=39 -x KMP_BLOCKTIME=1  -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 python3 -u /workspace/unet/UnetImgseg_Horovod.py
