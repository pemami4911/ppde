#!/bin/bash

CURRENT_DIR=`pwd`
DATASETS_DIR="$CURRENT_DIR/data"
RESULTS_DIR="$CURRENT_DIR/results/mnist"

# Download and store binary mnist
if [[ ! -d "$DATASETS_DIR/MNIST_static" ]]
then
    mkdir -p "$DATASETS_DIR/MNIST_static"
fi

echo 'downloading binary mnist...'

wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat

mv binarized_mnist_train.amat $DATASETS_DIR/MNIST_static/
mv binarized_mnist_valid.amat $DATASETS_DIR/MNIST_static/
mv binarized_mnist_test.amat $DATASETS_DIR/MNIST_static/

echo 'training binary mnist EBM...'

cd "$CURRENT_DIR/ppde/third_party"
git clone git@github.com:wgrathwohl/GWG_release.git
cd GWG_release

python3 pcd_ebm_ema.py --save_dir $RESULTS_DIR \
    --sampler gwg --sampling_steps 100 --viz_every 100 \
    --model resnet-64 --print_every 10 --lr .0001 --warmup_iters 10000 --buffer_size 10000 --n_iters 50000 \
    --buffer_init mean --base_dist --reinit_freq 0.0 \
    --eval_every 5000 --eval_sampling_steps 10000

echo 'training denoising autoencoder...'

#cd "$CURRENT_DIR/experiments/mnist/"
python3 scripts/train_binary_mnist_dae.py --save_dir $RESULTS_DIR

# Train sum regression models
# N.b. there are no labels accompanying BinaryMNIST, so we have 
# to use a dynamically binarized version of standard MNIST here.
# MNIST will get downloaded to $DATASETS_DIR

echo 'training one-hot ensemble of size 3 for sums <= 10...'
for i in {0..3}
do
    python3 train_binary_mnist_regression.py --ensemble_id $i --data_path $DATASETS_DIR --save_dir $RESULTS_DIR --model_type one-hot --sumTo 10
done

echo 'training denoising autoencoder latent space ensemble of size 3 for sums <= 10...'
for i in {0..3}
do
    python3 train_binary_mnist_regression.py --data_path $DATASETS_DIR --save_dir $RESULTS_DIR --model_type latent --sumTo 10
done

echo 'training oracle predictor for sums <= 18'
python3 train_binary_mnist_regression.py --data_path $DATASETS_DIR --save_dir $RESULTS_DIR --model_type one-hot --sumTo 18