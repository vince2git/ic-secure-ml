


use std::error::Error;

use wasm_bindgen::prelude::*;
use web_sys::console;


use linfa::traits::{Fit, Transformer};
use linfa_reduction::Pca;
use linfa_tsne::{Result, TSneParams};
use csv::ReaderBuilder;

use linfa::prelude::*;
use ndarray::{Array1, Array2, ArrayBase, Axis, Ix1, Ix2, OwnedRepr, s};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[wasm_bindgen]
pub fn tsne(csv_content: String,pca:usize,perp:f64,thres:f64,i:usize) -> String {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    //console::log_1(&JsValue::from(format!("{} {} {} {}", &pca,&perp,&thres,&i)));
    // create a dataset from it
    let ds = from_csv(&csv_content).unwrap();
    //log_dataset_summary(&ds, "After loading");
    // reduce to 50 dimension without whitening
    let ds = Pca::params(pca)
        .whiten(false)
        .fit(&ds)
        .unwrap()
        .transform(ds);
    //log_dataset_summary(&ds, "After PCA");
    // calculate a two-dimensional embedding with Barnes-Hut t-SNE
    let ds = TSneParams::embedding_size_with_rng(2,rng)
        .perplexity(perp)
        .approx_threshold(thres)
        .max_iter(i)
        .transform(ds).unwrap();
    //log_dataset_summary(&ds, "After t-SNE");

    format_tsne_results(ds).unwrap()
}
use serde_json::json;


fn format_tsne_results(ds: DatasetBase<Array2<f64>, ArrayBase<OwnedRepr<usize>, Ix1>>) -> Result<String> {

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

    Ok(serde_json::to_string(&points).unwrap())
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
    console::log_1(&JsValue::from(format!("{} - First few records: {:?}", stage, first_few)));

    // Calculer la moyenne manuellement
    let mean: Array1<f64> = records.mean_axis(Axis(0)).unwrap();
    console::log_1(&JsValue::from(format!("{} - Mean (first 5): {:?}", stage, mean.slice(s![0..5.min(mean.len())]))));

    // Calculer l'écart-type manuellement
    let std: Array1<f64> = records.std_axis(Axis(0), 0.0);
    console::log_1(&JsValue::from(format!("{} - Std (first 5): {:?}", stage, std.slice(s![0..5.min(std.len())]))));


}
