*Secure machine learning on Internet Computer*

The goal is to create a prototype of a secure and decentralized machine learning application (train and predict) using Internet Computer blockchain and compare similar solution locally and with traditional centralized approach.

**cycles-cost-tested/ic-mnist** folder contains trials including tests with canbench on the cycles costs of loading MNIST training dataset (50k records for ~200B cycles).

**cycles-cost-tested/ic_djstorch** folder contains test with instruction overflow (more than 40B instructions) with only 10 MNIST records using [Azle](https://demergent-labs.github.io/azle/).

**ic_linfa** is a first working machine learning (train/test) prototype using [Linfa](https://github.com/rust-ml/linfa) ML toolkit on the Internet Computer and is secured with encrypted message to and from the canister.

- You can try the prototype here https://zvrui-xqaaa-aaaao-a3pbq-cai.icp0.io/

**ic_split-learning** a prototype of neural net using split learning between client wasm and canister.

I'm new to Rust language, so this is also a learning project for me.