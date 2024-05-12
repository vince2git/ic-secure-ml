use core::num::NonZeroU32;
use std::cell::RefCell;
use std::time::Duration;

use getrandom::Error;
use ic_cdk_macros::init;
use ic_cdk_timers::set_timer;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use candid::{CandidType, Principal};

// Some application-specific error code
const MY_CUSTOM_ERROR_CODE: u32 = Error::CUSTOM_START + 42;
pub fn always_fail(buf: &mut [u8]) -> Result<(), Error> {
    let code = NonZeroU32::new(MY_CUSTOM_ERROR_CODE).unwrap();
    Err(Error::from(code))
}


// Get a random number generator based on 'raw_rand'.
// Based on https://github.com/dfinity/internet-identity/blob/f76e36cc45e064b5e04b977de43698c13e7b55d9/src/internet_identity/src/main.rs#L683-L697
/*pub async fn get_ic_rand_raw<T: SeedableRng + CryptoRng>() -> T
    where
    // raw_rand returns 32 bytes
        T: SeedableRng<Seed = [u8; 32]>,
{
    let raw_rand: Vec<u8> = match call(Principal, "raw_rand", ()).await {
        Ok((res,)) => res,
        Err((_, err)) => trap(&format!("failed to get seed: {}", err)),
    };

    let seed: <T as SeedableRng>::Seed = raw_rand[..].try_into().unwrap_or_else(|_| {
        trap(&format!(
            "when creating seed from raw_rand output, expected raw randomness to be of length 32, got {}",
            raw_rand.len()
        ));
    });

    T::from_seed(seed)
}
*/
thread_local! {
    static RNG: RefCell<Option<StdRng>> = RefCell::new(None);
}
#[init]
fn init() {
    ic_cdk_timers::set_timer(Duration::ZERO, || ic_cdk::spawn(async {
        let (seed,): ([u8; 32],) = ic_cdk::call(Principal::management_canister(), "raw_rand", ()).await.unwrap();
        RNG.with(|rng| *rng.borrow_mut() = Some(StdRng::from_seed(seed)));
    }));
}
pub fn custom_getrandom(buf: &mut [u8]) -> Result<(), getrandom::Error> {
    RNG.with(|rng| rng.borrow_mut().as_mut().unwrap().fill_bytes(buf));
    Ok(())
}