#![cfg_attr(not(target_feature = "simd128"), feature(stdsimd))]

use std::cmp::Ordering;
use linfa::traits::{Fit, Transformer};

use csv::ReaderBuilder;
use serde_json::{json, Value};
use linfa::prelude::*;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Ix1, Ix2, OwnedRepr, s};

use std::error::Error;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use linfa_reduction::*;
use linfa_trees::{DecisionTree, SplitQuality};
use linfa_datasets::dataset;
pub(crate) fn run(csv_content: String,pca:usize,perp:f64,thres:f64,i:usize,compare_origin:bool) -> String {
    //ic_cdk::println!("conf {} {} {} {}", &pca,&perp,&thres,&i);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    // create a dataset from it
    let ds = from_csv(&csv_content).unwrap();
    // Split the dataset into training and test sets (80% train, 20% test)
    let (train_dataset, test_dataset) = ds.split_with_ratio(0.8);
    let test_labels = test_dataset.targets();

    let params = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(i));

    let model: DecisionTree<f64, usize> = params.fit(&train_dataset).unwrap();

    let pred_y = model.predict(&test_dataset);
    let cm = pred_y.confusion_matrix(test_labels).unwrap();

    json!({
        "pred_y": pred_y,
        "accuracy": cm.accuracy(),

    }).to_string()
}


fn format_tsne_results(ds: DatasetBase<Array2<f64>, ArrayBase<OwnedRepr<usize>, Ix1>>) -> Vec<Value> {

    let mut points = Vec::new();

    for (x, y) in ds.sample_iter() {
        let label = if let Some(&value) = y.iter().next() {
            value
        } else {
            *y.into_scalar()
        };
        points.push(json!({
            "x": x[0],
            "y": x[1],
            "label": label
        }));
    }

    points
}

fn from_csv(csv_content: &str) -> std::result::Result<DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>, Box<dyn Error>> {

    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(csv_content.as_bytes());
    let mut features = Vec::new();
    let mut targets = Vec::new();

    for result in rdr.deserialize() {

        let values: Vec<f64> = result?;

        // Assuming the first column is the target (label)
        targets.push(values[0] as usize);

        // Normalize features by dividing by 255
        features.push(values[1..].iter().map(|&x| x / 255.0).collect::<Vec<f64>>());
    }

    let nrows = features.len();
    let ncols = features.get(0).map_or(0, |r| r.len());

    let features_array = Array2::from_shape_vec((nrows, ncols), features.into_iter().flatten().collect())?;
    let targets_array = Array1::from_vec(targets);

    Ok(Dataset::new( features_array,targets_array))

}

fn log_dataset_summary(ds: &DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>, stage: &str) {


    let records = ds.records();
    let first_few = records.slice(s![0..5.min(records.nrows()), 0..5.min(records.ncols())]);
    ic_cdk::println!("{} - First few records: {:?}", stage, first_few);
    // Calculer la moyenne manuellement
    let mean: Array1<f64> = records.mean_axis(Axis(0)).unwrap();

    ic_cdk::println!("{} - Mean (first 5): {:?}", stage, mean.slice(s![0..5.min(mean.len())]));
    // Calculer l'écart-type manuellement
    let std: Array1<f64> = records.std_axis(Axis(0), 0.0);

    ic_cdk::println!("{} - Std (first 5): {:?}", stage, std.slice(s![0..5.min(std.len())]));

}
