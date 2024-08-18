mod ftrl;
mod tsne;


use wasm_bindgen::prelude::*;
use web_sys::console;


#[wasm_bindgen]
pub fn ftrl(csv_content: String)->String{
    ftrl::run(csv_content)
}

#[wasm_bindgen]
pub fn tsne(csv_content: String,pca:usize,perp:f64,thres:f64,i:usize)->String{
    tsne::run(csv_content,pca,perp,thres,i)
}
