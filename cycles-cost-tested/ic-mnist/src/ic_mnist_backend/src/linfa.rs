use linfa::{Dataset, DataLoader, optimizers};
use linfa_dataset::mnist::{self, MNIST};

////Load MNIST dataset

// let mnist = MNIST::default();
// let num_classes = 10;

////Split into train and test datasets
// let (train_set, test_set) = mnist.split_with_ratio(0.8); 


let (train_set, test_set) = linfa_datasets::winequality()
    .split_with_ratio(0.8);
// Create data loader
let mut train_loader = DataLoader::new(train_set);

// Model, loss function, optimizer
let model = linfa::nn::Sequential::default()
    .insert(linfa::nn::dense::Dense::new(
        num_classes,
        linfa::nn::dense::DenseConfig::default(),
    ))
    .insert(linfa::nn::log_softmax::LogSoftmax::default(), 0);

let loss = linfa::loss::SoftmaxCrossEntropy::default();
let mut optimizer = optimizers::SGD::default();

// Train
for epoch in 1..10 {
    model.train();
    for (batch_x, batch_y) in train_loader.iter_epoch() {
        let pred = model.forward(&batch_x);
        
        // Compute loss
        let batch_loss = loss
            .loss(&pred, &batch_y)
            .mean()
            .unwrap();
        
        // Update weights            
        optimizer.update(&model, batch_loss);
    }

    // Evaluate
    let acc = evaluate(&model, &test_set);
    println!("Epoch {}, Test accuracy {}", epoch, acc);
}

fn evaluate(model: &linfa::Model, test_set: &Dataset) -> f32 {
    let mut test_loader = DataLoader::new(test_set);
    let mut correct = 0;
    let mut count = 0;
    for (x, y) in test_loader.iter() {
        let pred = model.forward(&x)
            .argmax(1)
            .unwrap();
        correct += pred.eq(&y).sum();
        count += x.len();
    }
    correct as f32 / count as f32
}