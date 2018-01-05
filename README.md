# distributed TF

```
export TENSORPACK_DATASET=/data/private/storage/tensorpack_data

python3 remote_feeder.py --dataset imagenet --service-code CONTACT_ME -p 48 --target tcp://localhost:1028
python3 train.py --dataset imagenet --port 1028 --num-gpus 8 --name imagenet_batch128_gpu8
```
