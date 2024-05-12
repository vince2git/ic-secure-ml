mod elasticnet;
mod ftrl;

use ic_cdk_macros::{query, update};


#[update]
fn linnerud(csv_content: String) -> Result<String, String> {
    elasticnet::run(csv_content).map_err(|e| e.to_string())
}

#[query]
fn wines(csv_content: String) -> Result<String, String> {
    ftrl::run(csv_content).map_err(|e| e.to_string())
}
ic_cdk_macros::export_candid!();