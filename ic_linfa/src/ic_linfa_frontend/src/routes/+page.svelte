
<script>
import "../index.scss";
import { backend } from "$lib/canisters";
import { mnistExtract,defaultWines } from "$lib/data.js";
import { onMount } from 'svelte';
import nacl from 'tweetnacl';
import { encodeBase64 } from 'tweetnacl-util';



import init,{ tsne} from 'wasm';

let canisterPublicKey = null;
let clientKeyPair = null;
let sharedSecret = null;
let tease="block";
const uuid=crypto.randomUUID();

let mobile='half';
onMount(async () => {
  await init();
  if(window.innerWidth<500)
      mobile='full';

  // Générer la paire de clés du client
  clientKeyPair = nacl.box.keyPair();
  //console.log(clientKeyPair.publicKey+" "+clientKeyPair.secretKey)
  // Obtenir la clé publique du canister
  canisterPublicKey = new Uint8Array(await getCanisterPublicKey(clientKeyPair.publicKey));
  //console.log(canisterPublicKey);
  // Calculer le secret partagé côté client
  //const clientSharedSecret = nacl.scalarMult(clientKeyPair.secretKey, canisterPublicKey);

  // Envoyer la clé publique du client au canister et obtenir le secret partagé
  //const canisterSharedSecret = await computeSharedSecret(clientKeyPair.publicKey);
  //console.log("c: "+clientSharedSecret+" s: "+canisterSharedSecret);
  let mess="test"
  //let enc=encrypt(mess,clientSharedSecret)

  let { encrypted, nonce } = encryptMessage(mess);
  //console.log(encrypted+" "+nonce)
  let response=await backend.decrypt_with_shared_secret(uuid,encrypted,nonce);

  let dec = decryptMessage(response[0],response[1]);
  if(dec===mess)
    console.log("encryption OK")
  else
    console.log("encryption KO")
  tease='none';
  /*console.log("mess: "+decrypt(enc,canisterSharedSecret))
  // Vérifier que les secrets partagés correspondent
  if (arrayBufferToHex(clientSharedSecret) === arrayBufferToHex(canisterSharedSecret)) {
    sharedSecret = arrayBufferToHex(clientSharedSecret);
    console.log("Secret partagé établi avec succès :", sharedSecret);

  } else {
    console.error("Erreur : Les secrets partagés ne correspondent pas");
  }*/
});

function encryptMessage(message) {
  const messageUint8 = new TextEncoder().encode(message);
  const nonce = nacl.randomBytes(nacl.box.nonceLength);
  const encrypted = nacl.box(
      messageUint8,
      nonce,
      canisterPublicKey,
      clientKeyPair.secretKey
  );

  return {
    encrypted: encodeBase64(encrypted),
    nonce: encodeBase64(nonce)
  };
}
function decryptMessage(encryptedMessage, nonce) {
  try {
    //console.log(encryptedMessage+ " "+nonce)

    const decrypted = nacl.box.open(
        encryptedMessage,
        nonce,
        canisterPublicKey,
        clientKeyPair.secretKey
    );

    if (!decrypted) {
      throw new Error("Échec du déchiffrement");
    }

    const decryptedText = new TextDecoder().decode(decrypted);
    return decryptedText;
  } catch (error) {
    console.error("Erreur lors du déchiffrement:", error);
    throw error;
  }
}



async function getCanisterPublicKey(pubk) {

  const publicKeyHex = await backend.generate_sk_pub(uuid,Array.from(pubk));

  return new Uint8Array(Object.values(publicKeyHex));
}




let result3="";
async function evalTsne(){
  icTsne('query')

}
async function evalTsneUpdate(){
  icTsne('update')

}

let err='';
async function icTsne(req){
    result3 ='Performing computations...'
  let startTime=performance.now()
  let { encrypted, nonce } = encryptMessage(mnist);
  let enc

  try{
    if(req=='query')
      enc = await backend.tsne(uuid,encrypted,nonce,pca,perp,thres,iter);
    else
      enc = await backend.tsne_update(uuid,encrypted,nonce,pca,perp,thres,iter);

    err='';
  }catch (e){
    err=e;
  }
  //console.log(enc)
  let response=decryptMessage(enc[0],enc[1]);
  console.log("caniste:"+response)
  drawPoints(JSON.parse(response))
  const endTime = performance.now();
  const totalTime = endTime - startTime;
  result3 = `Total duration on IC ( train with encryption/decryption forward-backward  print) : ${totalTime.toFixed(2)} ms`;

  return false;
}
async function clearKeys(){
   await backend.clear_all(uuid)

}
function evalLocalTsne(){
  result3 ='Performing computations...';
  let startTime=performance.now()
  let response
  try{
    response = tsne(mnist, pca, perp, thres, iter);
    err='';
  }catch (e){
    err=e;
  }
  console.log("local:"+response)
  drawPoints(JSON.parse(response))
  const endTime = performance.now();
  const totalTime = endTime - startTime;
  result3 = `Total duration for local wasm (train print) : ${totalTime.toFixed(2)} ms`;


  return false;

}
let ival=1;


let canvas;
let ctx;
// Fonction pour obtenir une couleur basée sur le label MNIST
function getColorForLabel(label) {
  const colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  return colors[label % colors.length];
}
// Fonction pour dessiner les points sur le canvas
function drawPoints(data) {

  const ctx = canvas.getContext('2d');

  // Effacer le canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Trouver les valeurs min et max pour la normalisation
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const labelCounts = {};
  data.forEach(point => {
    // Normaliser les coordonnées pour s'adapter au canvas
    const canvasX = ((point.x - minX) / (maxX - minX)) * canvas.width;
    const canvasY = ((point.y - minY) / (maxY - minY)) * canvas.height;
    const label = point.label;
    const color = getColorForLabel(label);
    labelCounts[label] = (labelCounts[label] || 0) + 1;
    // Dessiner le point
    ctx.beginPath();
    ctx.arc(canvasX, canvasY, 3, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  });
  // Dessiner la légende
  const legendX = 10;
  let legendY = 20;
  ctx.font = '12px Arial';
  for (let label in labelCounts) {
    const color = getColorForLabel(parseInt(label));
    ctx.fillStyle = color;
    ctx.fillRect(legendX, legendY, 15, 15);
    ctx.fillStyle = 'black';
    ctx.fillText(`Digit ${label}: ${labelCounts[label]} points`, legendX + 20, legendY + 12);
    legendY += 20;
  }

}

let result = "";
let result2 = "";

function onSubmit(event) {
  /*const exercices = event.target.exercices.value;
  backend.linnerud(exercices).then((response) => {
    result = JSON.stringify(response);
  });*/

  let { encrypted, nonce } = encryptMessage(wines);
 // console.log(encrypted)
  backend.wines(uuid,encrypted,nonce).then((response) => {
    result2 = JSON.stringify(response);
  });
  return false;
}
let wines=defaultWines;
let mnist=mnistExtract;

let pca=2;
let perp=2;
let thres=0.9;
let iter=20;
</script>



<main>

            <div class="overlay" style="display:{tease}"><h4 style="text-align:center">Encryption keys exchange...</h4></div>
        <h1 style="text-align:center">Prototype of secure Internet Computer vs local wasm training with <a href="https://github.com/rust-ml/linfa" target="_blank">Linfa.rs</a> machine learning benchmark</h1>
        <br />
        <div class="{mobile}">
            <h2>MNIST t-SNE (clustering, 300 handwritten digits)</h2>
            pca<input type="number" bind:value={pca}/>
            perplexity<input type="number" bind:value={perp}/>

            threshold<input type="number" bind:value={thres}/>
            max iter<input type="number" bind:value={iter}/>
        <br><br>
        <button on:click={evalTsne}>Eval on IC query (max 5B instructions)</button>
        <button on:click={evalTsneUpdate}>Eval on IC update (max 40B instructions)</button>
        <button on:click={evalLocalTsne}>Eval in local WASM</button> {err}
        <br><br>
        <section id="result3">{result3}</section>
        <br>


            </div>
        <div class="{mobile}">
            <canvas
                bind:this={canvas}
        width={700}
        height={700}
        />
        <br>
            <button on:click={clearKeys}>* Erase keys from canister *</button><div class="warn">* no more computation will be available from the canister, must reload the page to regenerate and exchange keys *</div>
            <label>CSV digits data used</label>
            <textarea id="tsne"  style="height:300px;width:100%" >{mnist}</textarea>

        </div>

        <br />

            <form action="#" on:submit|preventDefault={onSubmit}>
            <h2>Wines quality FTRL (recommandations) </h2>
        <!-- <textarea id="exercices" name="exercices" alt="Exercices" style="height:300px;width:100%" />-->

        <textarea id="wines" name="wines" alt="Vins" style="height:300px;width:100%" bind:value={wines}/>


        <button type="submit">Train/test!</button>
            </form>
            <section id="result">{result}</section>
            <section id="result2">{result2}</section>
             <br>
            <br>
             <a href="https://github.com/vince2git/ic-secure-ml">Source code here</a>
        {@html m}
        <style>
            canvas {
                border: 1px solid;
            }
        @media (max-width:500px){
            canvas{
                width:400px;
                height:400px;
            }
            h1{
             font-size: 18px !important;
              width: 80% !important;
            }
            h2{
            font-size: 22px;
            }
        }
        h1{
            font-weight: lighter;
            font-size: 32px;
              width: 50%;
              margin: auto;
              display: block;
              position: relative;
         }
        .half{
          display:inline-block;
          width:48%;
          text-align: center;
          vertical-align: top;
        }
        button{
          padding:5px;
          width: 260px;
        }
        .warn{
          font-size: 12px;
          color: #b81f1f;
        }
        input{
            width:50px;
        }
        .overlay {
          z-index: 1;
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          background-color: #f0f8ffcc;
          height: 100%;
        }

            </style>

</main>

