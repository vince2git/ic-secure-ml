# `ic_split-learning`

It's a prototype of neural net implementing split learning with two Conv2D on the client wasm and 2 Dense layer and a softmax on IC canister

## Running the project locally

If you want to test the project locally, you will need [MNIST csv files](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and a local web server to serve it at:
- http://localhost/mnist_train.csv
- http://localhost/mnist_test.csv

You can then use the following commands:

```bash
# Starts the replica, running in the background
dfx start --background

cd src/ic_usl_frontend

#install wasm-pack
npm i -g wasm-pack

# build client wasm and deploy canister locally
npm run setup
```
