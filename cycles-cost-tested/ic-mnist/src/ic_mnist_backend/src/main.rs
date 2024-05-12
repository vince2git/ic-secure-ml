// This example is disabled for windows till mnist > 0.5 is released
// See https://github.com/davidMcneil/mnist/issues/10

#[cfg(not(target_family = "windows"))]
use linfa_tsne::Result;
use log::{info, warn};


use linfa::traits::Fit;
use linfa::traits::Transformer;
use candid::{CandidType, Principal};
use ic_cdk_macros::*;
use ic_cdk::{update};
use linfa::dataset::Dataset;
use linfa::prelude::*;
use ndarray::Array2;
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::BTreeMap;
use linfa::Dataset;
type WorkerStore = BTreeMap<Principal, Worker>;

thread_local! {
    static WORKERS: RefCell<WorkerStore> = RefCell::default();
}

#[derive(Clone, Debug, CandidType, Deserialize)]
struct TrainRequest {
    data_chunk: Dataset<Array2<f64>,Array2<f64>>,
    params: PcaParameters,
}

#[derive(Clone, Debug, CandidType, Deserialize)]
struct Worker {
    canister_id: Principal,
}

#[update]
fn register_worker(worker: Worker) {
    let caller_principal_id = ic_cdk::caller();
    WORKERS.with(|workers| {
        workers
            .borrow_mut()
            .insert(caller_principal_id, worker)
    });
}

#[update]
async fn train_distributed_pca(params: PcaParameters) -> PcaModel {
    let ds = /* Votre jeu de données d'origine */;
    let num_workers = WORKERS.with(|workers| workers.borrow().len());
    let chunk_size = ds.rows().len() / num_workers;
    let mut ds_chunks = Vec::with_capacity(num_workers);

    for i in 0..num_workers {
        let start = i * chunk_size;
        let end = if i == num_workers - 1 {
            ds.rows().len()
        } else {
            (i + 1) * chunk_size
        };
        let ds_chunk = ds.rows().split_at(start).1.split_at(end - start).0;
        ds_chunks.push(Dataset::from_rows(ds_chunk.to_owned()));
    }

    let mut pca_models = Vec::with_capacity(num_workers);
    WORKERS.with(|workers| {
        for (worker_id, ds_chunk) in workers.borrow().iter().zip(ds_chunks.iter()) {
            let train_request = TrainRequest {
                data_chunk: ds_chunk.clone(),
                params: params.clone(),
            };
            let _call_result: Result<PcaModel, _> =
                ic_cdk::call_with_payment(*worker_id, "train_pca_model", (train_request,), 0);
            if let Ok(pca_model) = _call_result {
                pca_models.push(pca_model);
            }
        }
    });

    // Combiner les modèles PCA entraînés sur chaque sous-ensemble
    // ...

    final_pca_model
}

#[cfg(not(target_family = "windows"))]
pub fn main() -> Result<()> {
    use linfa::traits::{Fit, Transformer};
    use linfa::Dataset;
    use linfa_reduction::Pca;
    use linfa_tsne::TSneParams;

    #[cfg(not(target_family = "windows"))]
    //use mnist::{Mnist, MnistBuilder};

    use ndarray::Array;
    use std::{io::Write, process::Command};
    use std::time::SystemTime;




    println!("{} {}", "start","time");
    // use 50k samples from the MNIST dataset


    // download and extract it into a dataset
    let ds = linfa_datasets::mnist();
    // Diviser le jeu de données en n parties égales
    let num_canisters = 1;
    let chunk_size = ds.rows().len() / num_canisters;
    let mut ds_chunks = Vec::with_capacity(num_canisters);


    for i in 0..num_canisters {
        let start = i * chunk_size;
        let end = if i == num_canisters - 1 {
            ds.rows().len()
        } else {
            (i + 1) * chunk_size
        };
        let ds_chunk = ds.rows().split_at(start).1.split_at(end - start).0;
        ds_chunks.push(Dataset::from_rows(ds_chunk.to_owned()));
    }
    println!("{} {}", "fit start","time");
    // reduce to 50 dimension without whitening
    /*
    let ds = Pca::params(5)
       .whiten(false)
       .fit(&ds)
       .unwrap()
       .transform(ds);
   println!("{} {}", "tsne start","time");
   // calculate a two-dimensional embedding with Barnes-Hut t-SNE
     let ds = TSneParams::embedding_size(2)
        .perplexity(10.0)
        .approx_threshold(0.9)
        .max_iter(10)
        .transform(ds)?;
    println!("{} {}", "done","time");
    // write out
   let mut f = std::fs::File::create("examples/mnist.dat").unwrap();
     println!(".dat");
     for (x, y) in ds.sample_iter() {
         f.write_all(format!("{} {} {}\n", x[0], x[1], y[0]).as_bytes())
             .unwrap();
     }
     println!("written");
     // and plot with gnuplot
     Command::new("gnuplot")
         .arg("-p")
         .arg("examples/mnist_plot.plt")
         .spawn()
         .expect(
             "Failed to launch gnuplot. Pleasure ensure that gnuplot is installed and on the $PATH.",
         );*/
    println!("plotted");
    Ok(())
}

#[cfg(target_family = "windows")]
fn main() {}