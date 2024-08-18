

use linfa::traits::{Fit, Transformer};
use linfa_reduction::Pca;
use linfa_tsne::{Result, TSneParams};
use csv::ReaderBuilder;
use serde_json::{json, Value};
use linfa::prelude::*;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Ix1, Ix2, OwnedRepr, s};
use std::cmp::Ordering;
use std::collections::HashMap;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use wasm_bindgen::JsValue;
use web_sys::console;
use std::error::Error;

pub fn run(csv_content: String,pca:usize,perp:f64,thres:f64,i:usize) -> String {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    console::log_1(&JsValue::from(format!("{} {} {} {}", &pca,&perp,&thres,&i)));
    // create a dataset from it
    let ds = from_csv(&csv_content).unwrap();
    let ds_clone=ds.clone();
    //log_dataset_summary(&ds, "After loading");
    // reduce to 50 dimension without whitening
    let ds_pca = Pca::params(pca)
        .whiten(false)
        .fit(&ds)
        .unwrap()
        .transform(ds);
    //log_dataset_summary(&ds, "After PCA");
    // calculate a two-dimensional embedding with Barnes-Hut t-SNE
    let ds_tsne = TSneParams::embedding_size_with_rng(2,rng)
        .perplexity(perp)
        .approx_threshold(thres)
        .max_iter(i)
        .transform(ds_pca.clone()).unwrap();
    //log_dataset_summary(&ds, "After t-SNE");
    // Calculate Trustworthiness
    let k = 25; // Nombre de voisins à considérer

    let trustworthiness_score = trustworthiness(&ds_pca.records(), &ds_tsne.records(), k);

    let continuity_score = continuity(&ds_pca.records(), &ds_tsne.records(), k);
    let silhouette = silhouette_score(&ds_tsne.records(),&ds_clone.targets());
    let torigin= trustworthiness(&ds_clone.records(), &ds_tsne.records(), k);
    let corigin=continuity(&ds_clone.records(), &ds_tsne.records(), k);
    let density=average_clusters_density(&ds_tsne.records(),&ds_clone.targets(), k);
    let points =format_tsne_results(ds_tsne);
    let custom_score = synthetic_score(trustworthiness_score, continuity_score, silhouette, density,1.0, 1.0,1.0);
    json!({
        "points": points,
        "trust": trustworthiness_score,
        "torigin":torigin,
        "continuity": continuity_score,
        "corigin":corigin,
        "silhouette":silhouette,
        "score":custom_score,
    }).to_string()
}

fn synthetic_score(trustworthiness: f64, continuity: f64, silhouette: f64,density:f64, w1: f64, w2: f64, w3: f64) -> f64 {
    let normalized_silhouette = (silhouette + 1.0) / 2.0;
    let weighted_sum = w1 * trustworthiness + w2 * continuity + w3 * normalized_silhouette + density;
    let sum_weights = w1 + w2 + w3;

    weighted_sum / sum_weights
}
fn normalize_density(density: f64, min_density: f64, max_density: f64) -> f64 {
    (density - min_density) / (max_density - min_density)
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
    console::log_1(&JsValue::from(format!("{} - First few records: {:?}", stage, first_few)));

    // Calculer la moyenne manuellement
    let mean: Array1<f64> = records.mean_axis(Axis(0)).unwrap();
    console::log_1(&JsValue::from(format!("{} - Mean (first 5): {:?}", stage, mean.slice(s![0..5.min(mean.len())]))));

    // Calculer l'écart-type manuellement
    let std: Array1<f64> = records.std_axis(Axis(0), 0.0);
    console::log_1(&JsValue::from(format!("{} - Std (first 5): {:?}", stage, std.slice(s![0..5.min(std.len())]))));


}


fn trustworthiness(original_data: &Array2<f64>, projected_data: &Array2<f64>, k: usize) -> f64 {
    let n = original_data.nrows();
    let mut trustworthiness = 0.0;

    for i in 0..n {
        let mut original_distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&original_data.row(i), &original_data.row(j))))
            .collect();

        let mut projected_distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&projected_data.row(i), &projected_data.row(j))))
            .collect();

        original_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        projected_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let original_neighbors: Vec<usize> = original_distances
            .iter()
            .take(k)
            .map(|&(j, _)| j)
            .collect();

        let projected_neighbors: Vec<usize> = projected_distances
            .iter()
            .take(k)
            .map(|&(j, _)| j)
            .collect();

        for &j in &projected_neighbors {
            if !original_neighbors.contains(&j) {
                let r = original_distances
                    .iter()
                    .position(|&(idx, _)| idx == j)
                    .unwrap();
                trustworthiness += (r as f64 - k as f64);
            }
        }
    }

    1.0 - (2.0 / (n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0))) * trustworthiness
}
fn continuity(original_data: &Array2<f64>, projected_data: &Array2<f64>, k: usize) -> f64 {
    let n = original_data.nrows();
    let mut continuity = 0.0;

    for i in 0..n {
        let mut original_distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&original_data.row(i), &original_data.row(j))))
            .collect();

        let mut projected_distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&projected_data.row(i), &projected_data.row(j))))
            .collect();

        original_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        projected_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let original_neighbors: Vec<usize> = original_distances
            .iter()
            .take(k)
            .map(|&(j, _)| j)
            .collect();

        let projected_neighbors: Vec<usize> = projected_distances
            .iter()
            .take(k)
            .map(|&(j, _)| j)
            .collect();

        for &j in &original_neighbors {
            if !projected_neighbors.contains(&j) {
                let r = projected_distances
                    .iter()
                    .position(|&(idx, _)| idx == j)
                    .unwrap();
                continuity += (r as f64 - k as f64);
            }
        }
    }

    1.0 - (2.0 / (n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0))) * continuity
}

fn silhouette_score(data: &Array2<f64>, labels: &Array1<usize>) -> f64 {
    let n_samples = data.nrows();
    let mut silhouette_scores = Vec::with_capacity(n_samples);

    for (i, point) in data.rows().into_iter().enumerate() {
        let mut intra_cluster_distances = Vec::new();
        let mut inter_cluster_distances = vec![Vec::new(); 10]; // 10 classes for MNIST

        for (j, other_point) in data.rows().into_iter().enumerate() {
            if i != j {
                let distance = euclidean_distance(&point, &other_point);
                if labels[i] == labels[j] {
                    intra_cluster_distances.push(distance);
                } else {
                    inter_cluster_distances[labels[j]].push(distance);
                }
            }
        }

        let a = if intra_cluster_distances.is_empty() {
            0.0
        } else {
            intra_cluster_distances.iter().sum::<f64>() / intra_cluster_distances.len() as f64
        };

        let b = inter_cluster_distances
            .iter()
            .filter(|cluster| !cluster.is_empty())
            .map(|cluster| cluster.iter().sum::<f64>() / cluster.len() as f64)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        let s = if a == b {
            0.0
        } else {
            (b - a) / a.max(b)
        };

        silhouette_scores.push(s);
    }

    silhouette_scores.iter().sum::<f64>() / n_samples as f64
}
fn average_clusters_density(points: &Array2<f64>, labels: &Array1<usize>, k: usize) ->  f64 {
    let mut cluster_points: HashMap<usize, Vec<ArrayView1<f64>>> = HashMap::new();

    // Regrouper les points par cluster
    for (i, &label) in labels.iter().enumerate() {
        cluster_points.entry(label).or_insert_with(Vec::new).push(points.row(i));
    }

    let mut cluster_densities = HashMap::new();

    for (label, points) in cluster_points {
        let n_points = points.len();
        let mut densities = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let mut distances = Vec::with_capacity(n_points - 1);

            for j in 0..n_points {
                if i != j {
                    let distance = euclidean_distance(&points[i], &points[j]);
                    distances.push(distance);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let kth_distance = distances.get(k.min(distances.len() - 1)).unwrap_or(&f64::MAX);
            let density = 1.0 / kth_distance.max(f64::EPSILON);
            densities.push(density);
        }

        let avg_density = densities.iter().sum::<f64>() / n_points as f64;
        cluster_densities.insert(label, avg_density);
    }

    let total_density: f64 = cluster_densities.values().sum();
    total_density / cluster_densities.len() as f64
}
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}
