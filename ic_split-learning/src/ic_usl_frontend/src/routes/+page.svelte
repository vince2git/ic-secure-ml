<script>
  import "../index.scss";
  import { canister } from "$lib/canisters";
  import { onMount } from 'svelte';
  import init, { ClientModel } from 'wasm';

  let wasm;
  let n = 1;
  let accuracy = null;
  let canvas;
  let ctx;
  let status;
  let prediction = "";
  let isDrawing = false;
  let lr='0.01';
  let e='1';
  let recs='6000';
  onMount(async () => {
    await initializeWasm();
    setupCanvas();
  });
/*
  onMount(() => {
    init().then(() => {
      console.log('init wasm-pack');
      nn = ClientModel.new([1, 28, 28], 10); // Exemple de dimensions d'entrée pour MNIST
    });
  });*/
async function initializeWasm() {
  await init();
  wasm = new ClientModel();
  status="Prêt à l'entrainement";
}
  function setupCanvas() {
  ctx = canvas.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 35;
  ctx.lineCap = 'round';
}

  function startDrawing(event) {
  isDrawing = true;
  draw(event);
}

  function stopDrawing() {
  isDrawing = false;
  ctx.beginPath();
}

  function draw(event) {
  if (!isDrawing) return;

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

  function clearCanvas() {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  prediction = "";
}
function preprocessImage(ctx) {
  // Taille originale et cible
  const originalWidth = 280;
  const originalHeight = 280;
  const targetWidth = 28;
  const targetHeight = 28;

  // Canvas temporaire pour redimensionner l'image
  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');
  tempCanvas.width = targetWidth;
  tempCanvas.height = targetHeight;

  // Redimensionner l'image
  tempCtx.drawImage(ctx.canvas, 0, 0, originalWidth, originalHeight, 0, 0, targetWidth, targetHeight);

  // Obtenir les données d'image redimensionnées
  const imageData = tempCtx.getImageData(0, 0, targetWidth, targetHeight);
  let data = imageData.data;
  data = data.map(x => (255-x));
  console.log("Valeurs brutes des pixels:", data); // Affichez les premières valeurs des pixels pour le diagnostic

  // Convertir en niveaux de gris et normaliser
  const input = Array.from({ length: targetWidth * targetHeight }, (_, i) => {
    const pixelIndex = i * 4;
    const red = data[pixelIndex];
    const green = data[pixelIndex + 1];
    const blue = data[pixelIndex + 2];

    // Vérifiez les valeurs des pixels avant conversion
    if (isNaN(red) || isNaN(green) || isNaN(blue)) {
      console.error(`NaN détecté aux indices de pixel: ${pixelIndex}, valeurs: R=${red}, G=${green}, B=${blue}`);
    }
    const gray = (red * 0.299 + green * 0.587 + blue * 0.114);
    return Math.round(gray);
    /*
    // Conversion en niveaux de gris en utilisant une moyenne pondérée des canaux R, G, B
    const gray = (red * 0.299 + green * 0.587 + blue * 0.114) / 255;

    // Vérifiez la valeur de gris
    if (isNaN(gray)) {
      console.error(`NaN détecté lors de la conversion en niveaux de gris pour le pixel ${i}, valeurs: R=${red}, G=${green}, B=${blue}`);
    }

    // Inverser les valeurs de gris si nécessaire (1 - gray)
    const normalizedGray =  gray;

    // Vérifiez la valeur normalisée
    if (isNaN(normalizedGray)) {
      console.error(`NaN détecté lors de la normalisation pour le pixel ${i}, valeur de gris: ${gray}`);
    }

    return normalizedGray;
    */

  });

  console.log("Valeurs adaptées des pixels:", input); // Affichez les premières valeurs normalisées pour le diagnostic

  return data;
}

async function handlePredict() {
  const input = preprocessImage(ctx);
  const result = await predict(input);

  const predictedClass = result.class;
  const probabilities = result.probabilities;

  console.log("Predicted Class:", predictedClass);
  console.log("Probabilities:", probabilities);

  // Vous pouvez utiliser la classe prédite pour afficher ou effectuer d'autres actions dans votre application
  prediction = `Prediction: ${predictedClass}`;
}
async function predict(image) {
  if (!wasm) {
    console.error("WebAssembly model not initialized");
    return null;
  }
  //console.log(JSON.stringify("image: "+image))
  // Forward pass du modèle client (convolutions)
  const clientOutput = wasm.forward(image, 1);
  //console.log(JSON.stringify("client: "+clientOutput))
  // Envoyer les activations intermédiaires au serveur pour la propagation avant
  const serverOutput = await canister.predict_forward(clientOutput[0]);
  //console.log(JSON.stringify("server: "+serverOutput))
  // Appliquer softmax pour obtenir les probabilités
  const probabilities = softmax(serverOutput);

  // Trouver l'indice de la plus grande valeur (classe prédite)
  const predictedClass = probabilities.indexOf(Math.max(...probabilities));

  return {
    class: predictedClass,
    probabilities: probabilities
  };
}
function softmax(arr) {
  const maxVal = Math.max(...arr); // Trouver la valeur maximale
  const expArr = arr.map(x => Math.exp(x - maxVal)); // Soustraire la valeur maximale avant de calculer l'exponentielle
  const sumExpArr = expArr.reduce((a, b) => a + b, 0);
  return expArr.map(exp => exp / sumExpArr);
}
// Fonction softmax

/*
  async function predict(image) {
    const clientOutput = wasm.forward(image);
    const serverModelUpdate = await canister.get_model_update();
    const serverWeights = serverModelUpdate.dense2_weights;
    const serverBias = serverModelUpdate.dense2_bias;
    const finalOutput = new Array(10).fill(0);
    for (let i = 0; i < 10; i++) {
    for (let j = 0; j < clientOutput.length; j++) {
      finalOutput[i] += clientOutput[j] * serverWeights[i][j];
    }
    finalOutput[i] += serverBias[i];
  }

  const expValues = finalOutput.map(Math.exp);
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  return expValues.map(exp => exp / sumExp);
}

async function handlePredict() {
  const imageData = ctx.getImageData(0, 0, 28, 28);
  const input = new Float32Array(28 * 28);
  for (let i = 0; i < imageData.data.length; i += 4) {
    input[i / 4] = imageData.data[i] / 255;
  }
  const prediction = await predict(input);
 status = `Prediction: ${prediction.indexOf(Math.max(...prediction))}`;
}
*/
async function trainModel(batchImages, batchLabels, learningRate) {
  const batchSize = batchImages.length;
  const preprocessedBatch = wasm.preprocess_batch(batchImages.flat(), batchLabels.flat(), batchSize);
  console.log("Preprocessed:", JSON.stringify(preprocessedBatch));
  const loss = await canister.train(preprocessedBatch, learningRate);
  console.log(`Training loss: ${loss}`);

  const modelUpdate = await canister.get_model_update();

  // Prétraitement des poids
  let weights = [];
  if (modelUpdate.dense2_weights) {
    weights = modelUpdate.dense2_weights.flat().map(w => w === null ? 0 : w);
  }

  // Prétraitement des biais
  let bias = [];
  if (modelUpdate.dense2_bias) {
    bias = modelUpdate.dense2_bias.map(b => b === null ? 0 : b);
  }

  console.log("weights :", JSON.stringify(weights));
  console.log("bias:", JSON.stringify(bias));

  // Vérification que nous avons des données à charger
  if (weights.length > 0 && bias.length > 0) {
    await wasm.load_model(weights, bias);
    console.log("Model updated successfully");
  } else {
    console.warn("No valid data to update the model");
  }

  return loss;
}

async function trainModel2( batchImages, batchLabels, learningRate) {
  const batchSize = batchImages.length;

 const flattenedImagesBatch= new Float64Array(batchImages.flat());
   let tensor;
   try {
     tensor = wasm.forward(flattenedImagesBatch, batchSize);
   } catch (error) {
     console.error("Error in forward pass:", error);
     throw error;
   }
// Préparer l'objet ImagesBatch
  const imagesBatch = {
    tensor: tensor,
    labels: batchLabels
  };
  //console.log("Encoded: "+JSON.stringify(imagesBatch))
  //const flattenedEncodedBatch = Float64Array.from(encodedBatch.flat());
  //console.log("flat: "+JSON.stringify(flattenedEncodedBatch))
  // Train on server
  console.log("size KB:"+estimateSizeInKB(imagesBatch))
  const{ loss, server_gradients } = await canister.train2(imagesBatch, learningRate);
  //console.log("server: "+JSON.stringify(server_gradients))
  console.log("loss: "+loss)
  // Backward pass on client

   wasm.backward(flattenedImagesBatch, server_gradients.flat(), learningRate,batchSize);


 return loss;
}
/*

*/
async function run() {


  status = 'Training...';

  let NUM_EPOCHS = parseInt(e);
  const BATCH_SIZE = 300;
  let num=1;
  const trainingData = await getMnist(BATCH_SIZE);

  for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    console.log(`Epoch ${epoch + 1}/${NUM_EPOCHS}`);
    let epochLoss = 0;
    for (let batch of trainingData) {
      console.log("lot: "+ num++);
      const batchLoss = await trainModel2(batch.images, batch.labels, parseFloat(lr));
      epochLoss += batchLoss;
    }
    console.log(`Epoch ${epoch + 1} average loss: ${epochLoss / trainingData.length}`);
  }

  status = 'Training complete. Draw a digit and click "Predict"!';
}
async function getMnist( batchSize) {
  const response = await fetch('http://localhost/mnist_train.csv', {cache: "no-store"});
  const data = await response.text();
  const records = data.split('\n').map(row => row.split(',').map(Number));
  console.log('recs :'+recs+'/'+records.length);

  const batches = [];
  for (let i = 0; i+batchSize <= parseInt(recs); i += batchSize) {
    console.log('batch :'+batchSize);
    const recs = records.slice(i, i + batchSize);
    const images = recs.map(record => record.slice(1));
    const labels = recs.map(record => {
      let label = new Array(10).fill(0);
      label[record[0]] = 1;
     // console.log(label)
      return label;
    });
    batches.push({ images: images, labels: labels });
  }
  return batches;
}



async function handleEval() {
  status = "Evaluating model...";
  const response = await fetch('http://localhost/mnist_test.csv', {cache: "no-store"});
  const data = await response.text();
  const records = data.split('\n').map(row => row.split(',').map(Number));

  const images = records.map(record => record.slice(1));
  const labels = records.map(record => record[0]);

  accuracy = await evaluate(images, labels);
  status = `Evaluation complete. Accuracy: ${(accuracy * 100).toFixed(2)}%`;
}

async function evaluate(images, labels) {
  let correct = 0;
  for (let i = 0; i < images.length; i++) {
    const prediction = await predictTest(images[i]);
    //console.log("pred: "+prediction)
    const predictedLabel = prediction.indexOf(Math.max(...prediction));
    if (predictedLabel === labels[i]) {
      correct++;
    }
    if(i%100==0)
      console.log("correct 100 step: "+correct+" last pred: "+predictedLabel+" last real: "+labels[i]);
  }
  return correct / images.length;
}
async function predictTest(image) {
  //console.log("Input image:", image);

  const clientOutput = wasm.forward(image,1);
  //console.log("Client output:", clientOutput);

  // Envoyer les activations intermédiaires au serveur pour la propagation avant
  const serverOutput = await canister.predict_forward(clientOutput[0]);
  //console.log("server output:", serverOutput);
  // Apply softmax

  const softmaxProbabilities = softmax(serverOutput)
  //console.log("Softmax probabilities:", softmaxProbabilities);

  return softmaxProbabilities;
}
  function estimateSizeInKB(variable) {
    const jsonString = JSON.stringify(variable);
    const bytes = new TextEncoder().encode(jsonString).length;
    return bytes / 1024;
  }

  function saveModel() {
    const modelStr = wasm.save_model();
    localStorage.setItem("saved_model", modelStr);
    console.log('model sauvé');
  }

  function loadModel() {
    const modelStr = localStorage.getItem("saved_model");
    if (modelStr) {
      wasm.load_model(modelStr);
      console.log('model chargé');
    }
  }
</script>

<main>
  <img src="/logo2.svg" alt="DFINITY logo" />
  <br /><br />
    <div>{status}</div>
    <canvas
bind:this={canvas}
width={280}
height={280}
on:mousedown={startDrawing}
on:mousemove={draw}
on:mouseup={stopDrawing}
on:mouseleave={stopDrawing}
/>
<br />
<button on:click={clearCanvas}>Effacer</button>
    <button on:click={handlePredict}>Prédire</button>
    <div>{prediction}</div>
    Nb recs <input bind:value={recs} />
    Epoch <input bind:value={e} />
    Learning rate <input bind:value={lr} />
  <button on:click={run}>Commencer l'entraînement</button>
  <button on:click={saveModel}>Sauvegarder le modèle</button>
  <button on:click={loadModel}>Charger le modèle</button>
  <button on:click={handleEval}>Évaluer</button>

  {#if accuracy !== null}
    <p>Précision : {accuracy}</p>
  {/if}

</main>
<style>
    canvas {
    border: 1px solid black;
    cursor: crosshair;
    }
</style>
