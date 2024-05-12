use linfa::prelude::*;
use linfa_elasticnet::{ MultiTaskElasticNet};
use ndarray::{Array, Array2, Axis};

pub(crate) fn run(csv_content: String) -> Result<String, String> {
    let dataset = parse_csv(&csv_content).map_err(|e| e.to_string())?;

    // Split dataset into features (X) and targets (Y) if applicable
    let (n_samples, _n_features) = dataset.dim();
    let x = dataset;
    let y = Array::from_elem((n_samples, 1), 1.0); // Dummy target variable, adjust as needed

    // Split dataset into training and validation sets
    let train_size = (n_samples as f64 * 0.8) as usize;
    let (train_x, valid_x) = x.view().split_at(Axis(0), train_size);
    let (train_y, valid_y) = y.view().split_at(Axis(0), train_size);

    // Train model
    let model = MultiTaskElasticNet::params()
        .penalty(0.1)
        .l1_ratio(1.0)
        .fit(&Dataset::new(train_x.to_owned(), train_y.to_owned()))
        .map_err(|e| e.to_string())?;

    let intercept = model.intercept();
    let params = model.hyperplane();
    let z = model.z_score().ok();

    // Validate
    let valid_dataset = Dataset::new(valid_x.to_owned(), valid_y.to_owned());
    let y_est = model.predict(&valid_dataset);
    let variance = y_est.r2(&valid_dataset).map_err(|e| e.to_string())?;

    Ok(format!("intercept:{}\nparams:{}\nz:{:?}\nvariance:{}", intercept, params, z,variance))
}

fn parse_csv(csv_content: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let lines: Vec<&str> = csv_content.trim().lines().collect();
    let mut data = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        if i == 0 { continue; } // Skip header
        let values: Result<Vec<f64>, _> = line.split(',').map(str::parse).collect();
        match values {
            Ok(v) => data.push(v),
            Err(e) => return Err(Box::new(e)),
        }
    }

    let n_samples = data.len();
    let n_features = if n_samples > 0 { data[0].len() } else { 0 };
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();
    let array = Array::from_shape_vec((n_samples, n_features), flat_data)?;

    Ok(array)
}