import express from 'express';
import { torch } from "js-pytorch";
import { readFile} from 'fs/promises';
const nn = torch.nn;
const optim = torch.optim;


// Dimensions of the MNIST dataset images
const img_channels = 1; // Grayscale images, so 1 channel
const img_height = 28;
const img_width = 28;

// Size of the input layer (number of pixels in each image)
const input_size = img_channels * img_height * img_width;

// Size of the output layer (number of classes, which is 10 for MNIST)
const output_size = 10;

// Hyperparameters for the model
const hidden_size = 512; // Size of the hidden layers
const n_layers = 2; // Number of hidden layers
const n_heads = 8; // Number of attention heads in the Transformer
let dropout_p = 0.1; // Dropout probability

// Batch size and sequence length
const batch_size = 64;
const seq_len = 28; // Treat each row of pixels as a sequence
// Load the MNIST dataset from CSV
// @ts-ignore
async function loadMNISTFromCSV(filePath) {
    // @ts-ignore
    const response = (await readFile(filePath)).toString();
    const data = await response;
    const lines = data.trim().split('\n');
    const images = [];
    const labels = [];

    for (const line of lines) {
        const [labelStr, ...pixelValues] = line.split(',');
        const label = parseInt(labelStr, 10);
        const pixels = pixelValues.map(value => parseFloat(value) / 255); // Normalize pixel values to [0, 1]
        // @ts-ignore
        images.push(torch.tensor(pixels, [1, img_height, img_width, img_channels]));
        labels.push(label);
    }

    return [images, labels];
}

// Create a custom data loader
// @ts-ignore
function createDataLoader(images, labels, batchSize) {
    const dataLoader = {
        data: [],
        index: 0,
        batchSize: batchSize,
        reset: function() {
            this.index = 0;
            this.data = images.map((img, i) => [img, labels[i]]);
            this.data = this.data.sort(() => Math.random() - 0.5); // Shuffle the data
        },
        next() {
            const start = this.index;
            const end = this.index + this.batchSize;
            this.index = end;

            if (start >= this.data.length) {
                this.reset();
                return null;
            }

            const batch = this.data.slice(start, end);
            const batchImages = torch.zeros([batch.length, 1, img_height, img_width, img_channels]);
            // @ts-ignore
            const batchLabels = [];

            for (let i = 0; i < batch.length; i++) {
                // @ts-ignore
                const [img, label] = batch[i];
                // @ts-ignore
                batchImages.data[i] = img.data;
                batchLabels.push(label);
            }

            // @ts-ignore
            return [batchImages, torch.tensor(batchLabels)];

        }
    };

    dataLoader.reset();


    return dataLoader;
}

// Create the data loader
// @ts-ignore
async function createMNISTDataLoader(filePath, batchSize) {
    const [images, labels] = await loadMNISTFromCSV(filePath);
    return createDataLoader(images, labels, batchSize);
}



class Transformer extends nn.Module {
    constructor() {
        super();
        // Instantiate Transformer's Layers:
        this.embed = new nn.Linear(input_size, hidden_size);
        this.pos_embed = new nn.PositionalEmbedding(seq_len, hidden_size);
        this.blocks = Array.from({ length: n_layers }, () =>
            new nn.Block(
                hidden_size,
                hidden_size,
                n_heads,
                seq_len,
                (dropout_p = dropout_p)
            )
        );
        this.ln = new nn.LayerNorm(hidden_size);
        this.linear = new nn.Linear(hidden_size, output_size);
    }

    // @ts-ignore
    forward(x) {
        let z;
        z = x.view(x.size(0), -1); // Flatten the input images
        z = this.embed.forward(z);
        z = this.pos_embed.forward(z);
        for (const block of this.blocks) {
            z = block.forward(z);
        }
        z = this.ln.forward(z);
        z = this.linear.forward(z);
        return z;
    }
}
async function init() {
    const app = express();

    app.use(express.json());

    app.post('/', async (_req, res) => {
        try{
            const trainLoader = await createMNISTDataLoader('mnist.csv', batch_size);
            res.json(trainLoader);
    // Instantiate your custom nn.Module:
            const model = new Transformer();

    // Define loss function and optimizer:
            const loss_func = new nn.CrossEntropyLoss();
            const optimizer = new optim.Adam(model.parameters(), (lr = 5e-3), (reg = 0));

    // Training Loop:
            for (let epoch = 0; epoch < 40; epoch++) {
                // Get a batch of images and labels
                const [images, labels] = trainLoader.next();

                // Forward pass through the Transformer:
                const outputs = model.forward(images);

                // Get loss:
                const loss = loss_func.forward(outputs, labels);

                // Backpropagate the loss using torch.tensor's backward() method:
                loss.backward();

                // Update the weights:
                optimizer.step();

                // Reset the gradients to zero after each training step:
                optimizer.zero_grad();

                // Print the loss for this epoch
                console.log(`Epoch ${epoch + 1}/40, Loss: ${loss.item().toFixed(4)}`);
            }

        }catch(e){
            res.send(`error:`+e);
        }

    });

    app.listen();

}
init();