# distributed TF

* single node multi gpus
```
export TENSORPACK_DATASET=/data/private/storage/tensorpack_data

python3 remote_feeder.py --batchsize 128 --dataset imagenet --service-code CONTACT_ME -p 15 --target tcp://localhost:1028
python3 train.py -c configs/imagenet.json --gpus 0 1 --port 1028 --name imagenet_batch128_gpu2
```

* multi node multi gpus
```
export TENSORPACK_DATASET=/data/private/storage/tensorpack_data

#ps node
python3 multinode_parameter_server.py -c configs/imagenet.json

#worker1 node
python3 remote_feeder.py --batchsize 128 --dataset imagenet --service-code CONTACT_ME -p 48 --target tcp://localhost:2222
python3 multinode_train.py -c configs/imagenet.json --port 2222 --name multinode --task-index 0

#worker2 node
python3 remote_feeder.py --batchsize 128 --dataset imagenet --service-code CONTACT_ME -p 48 --target tcp://localhost:2222
python3 multinode_train.py -c configs/imagenet.json --port 2222 --name multinode --task-index 1
```
