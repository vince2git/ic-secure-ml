// This example is disabled for windows till mnist > 0.5 is released
// See https://github.com/davidMcneil/mnist/issues/10

#[cfg(not(target_family = "windows"))]
use linfa_tsne::Result;
use log::{info, warn};


use linfa::traits::Fit;
use linfa::traits::Transformer;


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