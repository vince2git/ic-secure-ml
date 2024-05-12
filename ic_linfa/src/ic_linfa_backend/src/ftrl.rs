use csv::ReaderBuilder;
use linfa::dataset::{AsSingleTargets, DatasetBase};
use linfa::prelude::*;
use linfa_ftrl::Ftrl;
use ndarray::{Array2, ArrayBase, Ix1, Ix2, OwnedRepr, s};
use rand::{rngs::SmallRng, SeedableRng};
use std::error::Error;

pub(crate) fn run(csv_content: String) -> Result<String, String> {
    let ds = load_csv_dataset(&csv_content).map_err(|e| e.to_string())?;
    let (train, valid) = ds
        .map_targets(|v| *v > 6)
        .split_with_ratio(0.9);

    let params = Ftrl::params()
        .alpha(0.005)
        .beta(1.0)
        .l1_ratio(0.005)
        .l2_ratio(1.0)
        .check_unwrap();

    let valid_params = params.clone();
    let mut model = Ftrl::new(valid_params, train.nfeatures());

    // Bootstrap each row from the train dataset to imitate online nature of the data flow
    let mut rng = SmallRng::seed_from_u64(42);
    let mut row_iter = train.bootstrap_samples(1, &mut rng);
    for _ in 0..train.nsamples() {
        let b_dataset = row_iter.next().unwrap();
        model = params.fit_with(Some(model), &b_dataset).map_err(|e| e.to_string())?;
    }
    let val_predictions = model.predict(&valid);
    let log_loss = val_predictions.log_loss(&valid.as_single_targets().to_vec());

    Ok(format!("valid log loss {:?}", log_loss))
}

fn load_csv_dataset(csv_content: &str) -> Result<DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>, Box<dyn Error>> {
    let array = from_csv(csv_content)?;

    let (data, targets) = (
        array.slice(s![.., 0..11]).to_owned(),
        array.column(11).to_owned().mapv(|x| x as usize),
    );

    let feature_names = vec![
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol",
    ];

    Ok(Dataset::new(data, targets).with_feature_names(feature_names))
}

fn from_csv(csv_content: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(csv_content.as_bytes());
    let mut records = Vec::new();

    for result in rdr.deserialize() {
        let record: Vec<f64> = result?;
        records.push(record);
    }

    let nrows = records.len();
    let ncols = records.get(0).map_or(0, |r| r.len());
    Array2::from_shape_vec((nrows, ncols), records.into_iter().flatten().collect()).map_err(Into::into)
}
