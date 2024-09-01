mkdir MNIST_ORG &&
cd MNIST_ORG &&
unzip ../MNIST_ORG.zip &&
mv t10k-images.idx3-ubyte t10k-images-idx3-ubyte &&
mv t10k-labels.idx1-ubyte t10k-labels-idx1-ubyte &&
mv train-images.idx3-ubyte train-images-idx3-ubyte &&
mv train-labels.idx1-ubyte train-labels-idx1-ubyte