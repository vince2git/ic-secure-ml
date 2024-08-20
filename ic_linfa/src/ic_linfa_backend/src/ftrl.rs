use std::collections::{BTreeMap, HashMap};
use csv::ReaderBuilder;
use linfa::dataset::{AsSingleTargets, DatasetBase};
use linfa::prelude::*;
use linfa_ftrl::Ftrl;
use ndarray::{array, Array1, Array2, ArrayBase, Axis, Data, Dimension, Ix1, Ix2, OwnedRepr, s};
use rand::{rngs::SmallRng, SeedableRng};
use std::error::Error;
use itertools::iproduct;

use rand::seq::SliceRandom;
use serde_json::json;

use rand_chacha::ChaCha8Rng;

pub(crate) fn run(csv_content: String) ->  String {
    //ic_cdk::println!("start");
    let ds = load_csv_dataset(&csv_content).unwrap();
    let (train, valid) = split_dataset_preserving_classes(ds.map_targets(|v| *v > 6), 0.8, 42);

    let params = Ftrl::params()
        .alpha(0.1)
        .beta(0.5)
        .l1_ratio(0.05)
        .l2_ratio(1.0)
        .check_unwrap();

    let valid_params = params.clone();
    let mut model = Ftrl::new(valid_params, train.nfeatures());


    // Train the model

    model = params.fit_with(Some(model), &train).map_err(|e| e.to_string()).unwrap();
    // Make predictions once
    let val_predictions = model.predict(&valid);

    // Calculate log loss
    let log_loss = val_predictions.log_loss(&valid.as_single_targets().to_vec()).unwrap();

    // Calculate other metrics
    let pred_bool = val_predictions.map(|p| **p > 0.5);
    //let true_labels= valid.targets().map(|&x| if x { true } else  {false });
    // Vérifier la matrice de confusion
    let true_labels = valid.as_single_targets();
    let cm = pred_bool.confusion_matrix(&true_labels).unwrap();
    let roc = val_predictions.roc(&valid.as_single_targets().to_vec()).unwrap();
    //ic_cdk::println!("end");

    let accuracy = cm.accuracy();
    let precision = cm.precision();
    let recall = cm.recall();
    let f1_score = cm.f1_score();

    let auc = roc.area_under_curve();
    let class_distribution = train.targets().map(|&x| if x { 1 } else { 0 }).sum();

    let mut hyper_results = String::new();
    // Utilisation
    let (best_model, best_threshold) = hyperparameter_search(&train, &valid);
    hyper_results.push_str(&format!("Meilleur seuil : {}", best_threshold));
    hyper_results.push_str(&format!("Meilleurs hyperparamètres : {:?},{:?},{:?},{:?}", best_model.alpha(),best_model.beta(),best_model.l1_ratio(),best_model.l2_ratio()));

    // Évaluation finale
    let final_predictions = best_model.predict(&valid);
    let final_pred_bool = final_predictions.map(|p| **p > best_threshold as f32);
    let final_cm = final_pred_bool.confusion_matrix(&true_labels).unwrap();
    let final_roc =final_predictions.roc(&valid.as_single_targets().to_vec()).unwrap();
    hyper_results.push_str(&format!("AUC final : {}", final_roc.area_under_curve()));

    let debug=format!("Class distribution: {} positives out of {} samples,{:?}, {:?}, {:?}, {:?}", class_distribution, train.nsamples(), cm,
                &val_predictions.slice(s![0..5]),
                &valid.targets().slice(s![0..5]),
                &hyper_results
    );
    json!({
        "log_loss": log_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "AUC":auc,
        "debug":debug
    }).to_string()
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
    //ic_cdk::println!("len {}",records.len());
    let nrows = records.len();
    let ncols = records.get(0).map_or(0, |r| r.len());
    Array2::from_shape_vec((nrows, ncols), records.into_iter().flatten().collect()).map_err(Into::into)
}


fn test_confusion_matrix_cases() -> String{
    let test_cases = vec![
        (vec![0, 0, 0, 0], vec![0, 1, 0, 0]),
        (vec![1, 1, 1, 1], vec![1, 1, 0, 1]),
        (vec![1, 0, 1, 0], vec![1, 0, 1, 0]),
        (vec![0, 0, 1, 0], vec![1, 1, 1, 1]),
        (vec![1, 1, 1, 1], vec![0, 0, 0, 0]),
    ];
    let mut results = String::new();
    for (i, (pred, true_labels)) in test_cases.iter().enumerate() {
        let pred_array = Array1::from(pred.clone());
        let true_labels_array = Array1::from(true_labels.clone());


        let cm = pred_array.confusion_matrix(&true_labels_array).unwrap();
        results.push_str(&format!("Test Case {}: {:?}\n", i + 1, cm));

        let accuracy = safe_metric(cm.accuracy());
        let precision = safe_metric(cm.precision());
        let recall = safe_metric(cm.recall());
        let f1_score = safe_metric(cm.f1_score());

        results.push_str(&format!("Accuracy: {:?}\n", accuracy));
        results.push_str(&format!("Precision: {:?}\n", precision));
        results.push_str(&format!("Recall: {:?}\n", recall));
        results.push_str(&format!("F1 Score: {:?}\n\n", f1_score));
    }
    results
}


fn safe_metric(value: f32) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else {
        value.to_string()
    }
}


fn hyperparameter_search(train: &Dataset<f64, bool, Ix1>, valid: &Dataset<f64, bool, Ix1>) -> (Ftrl<f64>, f32) {
    let alphas = vec![0.001, 0.005, 0.01, 0.05, 0.1];
    let betas = vec![0.5, 1.0, 2.0];
    let l1_ratios = vec![0.001, 0.005, 0.01, 0.05];
    let l2_ratios = vec![0.1, 0.5,0.7, 1.0];
    let thresholds = vec![0.3, 0.4, 0.5, 0.6, 0.7];

    let mut rng = rand::thread_rng();
    let combinations: Vec<_> = (0..100).map(|_| {
        (
            *alphas.choose(&mut rng).unwrap(),
            *betas.choose(&mut rng).unwrap(),
            *l1_ratios.choose(&mut rng).unwrap(),
            *l2_ratios.choose(&mut rng).unwrap(),
            *thresholds.choose(&mut rng).unwrap(),
        )
    }).collect();

    let results: Vec<_> = iproduct!(alphas, betas, l1_ratios, l2_ratios, thresholds)
        .filter_map(|(alpha, beta, l1_ratio, l2_ratio, threshold)| {
            let params = Ftrl::params()
                .alpha(alpha)
                .beta(beta)
                .l1_ratio(l1_ratio)
                .l2_ratio(l2_ratio)
                .check_unwrap();

            let model = Ftrl::new(params.clone(), train.nfeatures());
            params.fit_with(Some(model), train).ok().and_then(|model| {
                let predictions = model.predict(valid);
                let pred_bool = predictions.map(|p| **p > threshold);

                // Vérifier s'il y a au moins une prédiction de chaque classe
                let has_true = pred_bool.iter().any(|&x| x);
                let has_false = pred_bool.iter().any(|&x| !x);

                if has_true && has_false {
                    predictions.roc(&valid.as_single_targets().to_vec()).ok().map(|roc|{
                        let auc=roc.area_under_curve();
                        (model, auc, alpha, beta, l1_ratio, l2_ratio, threshold)
                    })

                    /*pred_bool.confusion_matrix(valid.targets()).ok().map(|cm| {
                        let MCC = cm.mcc();
                        (model, MCC, alpha, beta, l1_ratio, l2_ratio, threshold)
                    })*/
                } else {
                    None
                }
            })
        })
        .collect();



    results.into_iter()
        .filter(|result| result.1.is_finite()) // Filtrer les résultats non valides (NaN ou infinis)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|r| (r.0, r.6)).unwrap()
}

pub fn split_dataset_preserving_classes<F, T, D>(
    dataset: Dataset<F, T, D>,
    ratio: f32,
    seed: u64
) -> (Dataset<F, T, D>, Dataset<F, T, D>)
where
    F: Float,
    T: Clone + std::hash::Hash + Eq+ std::cmp::Ord, D: Dimension + ndarray::RemoveAxis

{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Grouper les indices par classe
    let mut indices_by_class: BTreeMap<T, Vec<usize>> = BTreeMap::new();
    for (idx, target) in dataset.targets().iter().enumerate() {
        indices_by_class.entry(target.clone()).or_default().push(idx);
    }

    let mut train_indices = Vec::new();
    let mut valid_indices = Vec::new();

    // Pour chaque classe, diviser les indices en respectant le ratio
    for indices in indices_by_class.values_mut() {
        //indices.shuffle(&mut rng);
        let split_point = (indices.len() as f32 * ratio).round() as usize;
        train_indices.extend_from_slice(&indices[..split_point]);
        valid_indices.extend_from_slice(&indices[split_point..]);
    }

    // Créer les nouveaux datasets
    let train_data = dataset.records().select(Axis(0), &train_indices);
    let train_targets = dataset.targets().select(Axis(0), &train_indices);
    let valid_data = dataset.records().select(Axis(0), &valid_indices);
    let valid_targets = dataset.targets().select(Axis(0), &valid_indices);

    let train_weights = if !dataset.weights().is_none() {
        Array1::from(train_indices.iter().map(|&i| dataset.weights().unwrap()[i]).collect::<Vec<_>>())
    } else {
        Array1::zeros(0)
    };

    let valid_weights = if !dataset.weights().is_none() {
        Array1::from(valid_indices.iter().map(|&i| dataset.weights().unwrap()[i]).collect::<Vec<_>>())
    } else {
        Array1::zeros(0)
    };

    let train_dataset = Dataset::new(train_data, train_targets)
        .with_weights(train_weights)
        .with_feature_names(dataset.feature_names().to_vec());

    let valid_dataset = Dataset::new(valid_data, valid_targets)
        .with_weights(valid_weights)
        .with_feature_names(dataset.feature_names().to_vec());

    (train_dataset, valid_dataset)
}