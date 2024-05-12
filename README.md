*Secure machine learning on Internet Computer*

The goal is to create a prototype of a secure and decentralized machine learning application using Internet Computer blockchain and compare a similar solution with a traditional centralized approach.

./cycles-cost-tested/ic-mnist folder contains trials including tests with canbench on the cycles costs of loading MNIST training dataset (50k records for ~200B cycles).

./cycles-cost-tested/ic_djstorch folder contains test with instruction overflow (more than 40B instructions) with only 10 MNIST records using Azle.

./ic_linfa is a first working machine learning (train/test) prototype using [Linfa](https://github.com/rust-ml/linfa) ML toolkit on the Internet Computer and needs to be secured.

I'm new to Rust language, so this is also a learning project for me.