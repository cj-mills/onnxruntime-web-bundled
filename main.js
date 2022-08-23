// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// import { getImageTensorFromPath } from './imageHelper';
const ort = require('onnxruntime-web');

// use an async context to call onnxruntime functions.
async function main() {

    // var image = new Image();
    var image_src = './images/sailboat.jpg';

    var image = document.getElementById('image');
    document.write(`Image Name: ${image.src}`);


    try {
        // window.alert("Hello World!");
        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
        // it has 1 output: 'c'(float32, 3x3)
        const session = await ort.InferenceSession.create('squeezenet1_1.onnx',
            { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });

        // // prepare inputs. a tensor need its corresponding TypedArray as data
        // const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        // const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
        // const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
        // const tensorB = new ort.Tensor('float32', dataB, [4, 3]);

        // // prepare feeds. use model input names as keys.
        // const feeds = { a: tensorA, b: tensorB };

        // // feed inputs and run
        // const results = await session.run(feeds);

        // // read from results
        // const dataC = results.c.data;
        // document.write(`data of result tensor 'c': ${dataC}`);
        document.write('Success!')

    } catch (e) {
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}

main();
