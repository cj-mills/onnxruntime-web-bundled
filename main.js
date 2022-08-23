// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// import { getImageTensorFromPath } from './imageHelper';
const ort = require('onnxruntime-web');
var session;

// use an async context to call onnxruntime functions.
async function main() {

    // var image = new Image();
    var image_src = './images/sailboat.jpg';

    var image = document.getElementById('image');

    var div = document.createElement("DIV");
    div.id = 'output_text';
    div.innerHTML = `Image Source: ${image.src}`;
    document.body.appendChild(div);

    try {
        // create a new session and load the specified model.
        session = await ort.InferenceSession.create('squeezenet1_1.onnx',
            { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });


        document.getElementById('output_text').innerHTML += 'Created inference session.';
    } catch (e) {
        document.getElementById('output_text').innerHTML += `failed to create inference session: ${e}.`;
    }

    try {
        // prepare inputs. a tensor need its corresponding TypedArray as data
        // const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        // const input_tensor = new ort.Tensor('float32', dataA, [3, 4]);

        // prepare feeds. use model input names as keys.
        // const feeds = { a: tensorA, b: tensorB };

        // feed inputs and run
        // const results = await session.run(feeds);

        // read from results
        // const dataC = results.c.data;
        // document.write(`data of result tensor 'c': ${dataC}`);
    } catch (e) {
        document.getElementById('output_text').innerHTML += `failed to perform inference: ${e}.`;
    }
}

main();
