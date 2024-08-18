mod elasticnet;
mod ftrl;
mod tsne;
mod trees;

use ic_cdk_macros::{query, update};


use std::cell::RefCell;
use std::collections::HashMap;


use anyhow::anyhow;

use getrandom::register_custom_getrandom;
use serde::{Deserialize, Serialize};
use crypto_box::{aead::{Aead, Payload}, SecretKey, PublicKey as PK, SalsaBox, Nonce};
use base64;
use base64::Engine;
use base64::engine::general_purpose;
use std::vec::Vec;
use crypto_box::aead::{AeadCore};

use ic_cdk::api::time;
use ic_cdk::id;
use rand_core::{CryptoRng, Error, RngCore};
use sha2::{Sha256, Digest};


fn custom_getrandom(_buf: &mut [u8]) -> Result<(), getrandom::Error> { Ok(()) }
register_custom_getrandom!(custom_getrandom);

thread_local! {
    static STATE: RefCell<CanisterState> = RefCell::new(CanisterState::default());
}
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
struct Session {
    sec: Vec<u8>,
    peer_public_key: Vec<u8>,
    last_active: u64,
}
#[derive(Serialize, Deserialize, Default)]
struct CanisterState {
    sessions: HashMap<String, Session>,
}

// Fonction pour créer ou mettre à jour une session
fn update_session(session_id: String, sec: Vec<u8>, peer_public_key: Vec<u8>) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let now = ic_cdk::api::time(); // Obtenir le timestamp actuel
        state.sessions.insert(session_id, Session {
            sec,
            peer_public_key,
            last_active: now,
        });
    });
}

// Fonction pour obtenir une session
fn get_session(session_id: &str) -> Option<Session> {
    STATE.with(|state| {
        let state = state.borrow();
        state.sessions.get(session_id).cloned()
    })
}

// Fonction pour nettoyer les sessions inactives
fn clean_inactive_sessions() {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let now = ic_cdk::api::time();
        state.sessions.retain(|_, session| now - session.last_active < 300000000000);
    });
}



struct CustomRng;

impl RngCore for CustomRng {
    fn next_u32(&mut self) -> u32 {
        let bytes = generate_random_bytes(4);
        u32::from_le_bytes(bytes.try_into().unwrap())
    }

    fn next_u64(&mut self) -> u64 {
        let bytes = generate_random_bytes(8);
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let random_bytes = generate_random_bytes(dest.len());
        dest.copy_from_slice(&random_bytes);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}
struct RandomState {
    counter: u64,
}

thread_local! {
    static RANDOM_STATE: std::cell::RefCell<RandomState> = std::cell::RefCell::new(RandomState { counter: 0 });
}

fn generate_random_bytes(num_bytes: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(num_bytes);

    while result.len() < num_bytes {
        let random_data = RANDOM_STATE.with(|state| {
            let mut state = state.borrow_mut();
            let bind=id();
            let canister_id = bind.as_slice();
            let timestamp = time();
            state.counter += 1;

            let mut hasher = Sha256::new();
            hasher.update(canister_id);
            hasher.update(&timestamp.to_be_bytes());
            hasher.update(&state.counter.to_be_bytes());
            hasher.finalize().to_vec()
        });

        result.extend_from_slice(&random_data);
    }

    result.truncate(num_bytes);
    result
}



// This is safe because our RNG is based on cryptographic primitives
impl CryptoRng for CustomRng {}
#[update]
fn generate_sk_pub(sess:String,client_key:Vec<u8>) -> Vec<u8> {
    //clean_inactive_sessions();
    ic_cdk::println!("start");
    let mut rng = CustomRng;
    let secret = SecretKey::generate(&mut rng);

    let public = secret.public_key();
    //ic_cdk::println!("pub {:?}",public);

    update_session(sess, secret.to_bytes().to_vec(), client_key);


    public.as_bytes().to_vec()
}
#[query]
fn decrypt_with_shared_secret(sess:String,ciphertext: String, nonce: String) -> (Vec<u8>, Vec<u8>) {
    match get_session(&sess) {
        Some(..) => {
            // Déchiffrement
            let decrypted = decrypt_internal(&sess, &ciphertext, &nonce);
            // Chiffrement
            encrypt(&sess, decrypted)

        },
        None => (Vec::new(),Vec::new())
    }

}
fn encrypt( sess: &String, message: String) -> (Vec<u8>, Vec<u8>) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let sec= state.sessions.get(sess).map(|session| session.sec.clone());
        let sk: [u8; 32] = sec.unwrap().try_into().map_err(|_| anyhow!("Invalid secret length")).unwrap();
        let secret_key = SecretKey::from(sk);
        let public = state.sessions.get(sess).map(|session| session.peer_public_key.clone());
        let pubk: [u8; 32] =public.unwrap().try_into().map_err(|_| anyhow!("Invalid secret length")).unwrap();
        let pub_key = PK::from(pubk);
        let mut rng = CustomRng;
        let nonce =SalsaBox::generate_nonce(&mut rng);
        let cypher = SalsaBox::new(&pub_key,&secret_key);
        let cyphertext= cypher.encrypt(&nonce,message.as_bytes());
        //ic_cdk::println!("{:?}  {:?}",&cyphertext,&nonce);
        let instructions = ic_cdk::api::performance_counter(0);
        ic_cdk::println!("{}",format!("i:{}", instructions));
        (cyphertext.unwrap(),nonce.to_vec())
    })

}

fn decrypt_internal(
    sess:&String,
    ciphertext: &str,
    nonce: &str
) -> String {
    let encrypted_data = general_purpose::STANDARD.decode(ciphertext).unwrap();
    let nonce_bytes = general_purpose::STANDARD.decode(nonce).unwrap();

    // Create Nonce from decoded bytes
    let nonce = Nonce::from_slice(&nonce_bytes);
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let sec= state.sessions.get(sess).map(|session| session.sec.clone());
        let sk: [u8; 32] = sec.unwrap().try_into().map_err(|_| anyhow!("Invalid secret length")).unwrap();
        let secret_key = SecretKey::from(sk);
        let public = state.sessions.get(sess).map(|session| session.peer_public_key.clone());
        let pubk: [u8; 32] =public.unwrap().try_into().map_err(|_| anyhow!("Invalid secret length")).unwrap();
        let pub_key = PK::from(pubk);
        // Create a CryptoBox instance
        let crypto_box = SalsaBox::new(&pub_key, &secret_key);
        // Decrypt the message

        let plaintext = crypto_box.decrypt(
            &nonce,
            Payload {
                msg: &encrypted_data[..],
                aad: &[], // Additional authenticated data, if any
            },
        );

        // Convert the plaintext to a UTF-8 string

        std::str::from_utf8(&plaintext.unwrap())
            .expect("Failed to convert decrypted data to string")
            .to_string()
    })

}


#[update]
fn clear_all(sess:String) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let now = ic_cdk::api::time(); // Obtenir le timestamp actuel
        let z:Vec<u8>=vec![0; 32];
        state.sessions.insert(sess, Session {
            sec:z.clone(),
            peer_public_key:z,
            last_active: now-300000000001,
        });
    });
    ic_cdk::println!("erase");
}


/*
#[update]
fn linnerud(csvcontent: String,nonce:String) -> Result<String, String> {
    let csv= decrypt_internal(&csvcontent,&nonce);
    elasticnet::run(csv).map_err(|e| e.to_string())
}
*/
#[query]
fn wines(sess:String,csvcontent: String,nonce:String) -> (Vec<u8>, Vec<u8>){
    ic_cdk::println!("wines");
    match get_session(&sess) {
        Some(..) => {
            let csv= decrypt_internal(&sess,&csvcontent,&nonce);
            encrypt(&sess,ftrl::run(csv))
        },
        None => (Vec::new(),Vec::new())
    }

}
#[query]
fn tsne(sess:String,csvcontent: String,nonce:String,pca:usize,perp:f64,thres:f64,i:usize) -> (Vec<u8>, Vec<u8>) {
    ic_cdk::println!("tsne");

    match get_session(&sess) {
        Some(..) => {
            let csv= decrypt_internal(&sess,&csvcontent,&nonce);

            encrypt(&sess,tsne::run(csv,pca,perp,thres,i,false))

        },
        None => (Vec::new(),Vec::new())
    }
}
#[update]
fn tsne_update(sess:String,csvcontent: String,nonce:String,pca:usize,perp:f64,thres:f64,i:usize) -> (Vec<u8>, Vec<u8>) {
    ic_cdk::println!("tsneU");
    match get_session(&sess) {
        Some(..) => {
            let csv= decrypt_internal(&sess,&csvcontent,&nonce);
            encrypt(&sess,tsne::run(csv,pca,perp,thres,i,true))

        },
        None => (Vec::new(),Vec::new())
    }
}

#[query]
fn trees(sess:String,csvcontent: String,nonce:String,pca:usize,perp:f64,thres:f64,i:usize) -> (Vec<u8>, Vec<u8>) {
    ic_cdk::println!("trees");

    match get_session(&sess) {
        Some(..) => {
            let csv= decrypt_internal(&sess,&csvcontent,&nonce);

            encrypt(&sess,trees::run(csv,pca,perp,thres,i,false))

        },
        None => (Vec::new(),Vec::new())
    }
}
#[update]
fn trees_update(sess:String,csvcontent: String,nonce:String,pca:usize,perp:f64,thres:f64,i:usize) -> (Vec<u8>, Vec<u8>) {
    ic_cdk::println!("treesU");

    match get_session(&sess) {
        Some(..) => {
            let csv= decrypt_internal(&sess,&csvcontent,&nonce);

            encrypt(&sess,trees::run(csv,pca,perp,thres,i,false))

        },
        None => (Vec::new(),Vec::new())
    }
}

ic_cdk_macros::export_candid!();