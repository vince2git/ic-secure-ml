import express from 'express';
import { ic } from 'azle';
import {MnistData} from './mnist/data';
import { torch } from "js-pytorch";
const nn = torch.nn;
const optim = torch.optim;


let data;
async function load() {
    data = new MnistData();
    await data.load();
}
// Load the MNIST dataset
// @ts-ignore
const mnistTrain = data.getTrainData();
// @ts-ignore
const mnistTest = data.getTestData();



class TransformerPart1 extends nn.Module {

    constructor(vocab_size, hidden_size, n_timesteps) {
        super();
        this.embed = new nn.Embedding(vocab_size, hidden_size);
        this.pos_embed = new nn.PositionalEmbedding(n_timesteps, hidden_size);
    }

    forward(x) {
        let z = torch.add(this.embed.forward(x), this.pos_embed.forward(x));
        return z;
    }
}

class TransformerPart2 extends nn.Module {
    constructor(hidden_size, n_timesteps, n_heads, p) {
        super();
        this.b1 = new nn.Block(
            hidden_size,
            hidden_size,
            n_heads,
            n_timesteps,
            (dropout_p = p)
        );
    }

    forward(z) {
        z = this.b1.forward(z);
        return z;
    }
}

class TransformerPart3 extends nn.Module {
    constructor(hidden_size, n_timesteps, n_heads, p) {
        super();
        this.b2 = new nn.Block(
            hidden_size,
            hidden_size,
            n_heads,
            n_timesteps,
            (dropout_p = p)
        );
        this.ln = new nn.LayerNorm(hidden_size);
        this.linear = new nn.Linear(hidden_size, vocab_size);
    }

    forward(z) {
        z = this.b2.forward(z);
        z = this.ln.forward(z);
        z = this.linear.forward(z);
        return z;
    }
}

async function init() {
    const app = express();

    app.use(express.json());

    app.get('/prediction', async (_req, res) => {
        // TODO Tokenization and prediction for this specific model have not yet been figured out
        // const prediction = model.predict(tf.tensor([]));

        res.send('Prediction not yet implemented');
    });

    app.listen();
// Instantiate parts of the custom nn.Module:
    const part1 = new TransformerPart1(vocab_size, hidden_size, n_timesteps);
    const part2 = new TransformerPart2(hidden_size, n_timesteps, n_heads, dropout_p);
    const part3 = new TransformerPart3(hidden_size, n_timesteps, n_heads, dropout_p);

// Define loss function and optimizer:
    const loss_func = new nn.CrossEntropyLoss();
    const optimizer1 = new optim.Adam(part1.parameters(), 5e-3, 0);
    const optimizer2 = new optim.Adam(part2.parameters(), 5e-3, 0);
    const optimizer3 = new optim.Adam(part3.parameters(),  5e-3, 0);

// Instantiate sample input and output:
    let x = torch.randint(0, vocab_size, [batch_size, n_timesteps, 1]);
    let y = torch.randint(0, vocab_size, [batch_size, n_timesteps]);
    let loss;

// Training Loop:
    for (let i = 0; i < 40; i++) {
        // Forward pass through the first part of the Transformer:
        let z = part1.forward(x);

        // Train the first part:
        optimizer1.zero_grad();
        loss = loss_func.forward(z, y);
        loss.backward();
        optimizer1.step();

        // Forward pass through the second part of the Transformer:
        z = part2.forward(z);

        // Train the second part:
        optimizer2.zero_grad();
        loss = loss_func.forward(z, y);
        loss.backward();
        optimizer2.step();

        // Forward pass through the third part of the Transformer:
        z = part3.forward(z);

        // Train the third part:
        optimizer3.zero_grad();
        loss = loss_func.forward(z, y);
        loss.backward();
        optimizer3.step();
    }
}
init();

