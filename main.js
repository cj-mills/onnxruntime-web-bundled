// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// import { getImageTensorFromPath } from './imageHelper';
const ort = require('onnxruntime-web');
var session;

async function init_session(model_path, exec_provider) {
    var return_msg;
    try {
        // create a new session and load the specified model.
        session = await ort.InferenceSession.create(model_path,
            { executionProviders: [exec_provider], graphOptimizationLevel: 'all' });
        return_msg = 'Created inference session.';
    } catch (e) {
        return_msg = `failed to create inference session: ${e}.`;
    }
    return return_msg;
}


// use an async context to call onnxruntime functions.
async function main() {

    // var image = new Image();
    var image_src = './images/sailboat.jpg';

    var image = document.getElementById('image');

    var div = document.createElement("DIV");
    div.id = 'output_text';
    div.innerHTML = `Image Source: ${image.src}`;
    document.body.appendChild(div);
    var model_path = 'squeezenet1_1.onnx';
    var exec_provider = 'wasm';
    var return_msg = init_session(model_path, exec_provider);

    document.getElementById('output_text').innerHTML += `<br>${return_msg}`;

    if (session = ! null) {
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
            document.getElementById('output_text').innerHTML += `<br>failed to perform inference: ${e}.`;
        }
    }

}

main();
