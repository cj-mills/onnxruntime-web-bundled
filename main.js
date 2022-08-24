// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// import { getImageTensorFromPath } from './imageHelper';
// import * as Jimp from 'jimp';
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

    var image = document.getElementById('image');
    var div = document.createElement("DIV");
    div.id = 'output_text';
    div.innerHTML = `Image Source: ${image.src}`;
    document.body.appendChild(div);

    var model_path = 'squeezenet1_1.onnx';
    var exec_provider = 'wasm';
    var return_msg = init_session(model_path, exec_provider);

    document.getElementById('output_text').innerHTML += `<br>${(await return_msg).toString()}`;

    if (session = ! null) {

        var canvas = document.createElement('canvas');
        var context = canvas.getContext('2d');
        canvas.width = image.width;
        canvas.height = image.height;

        var imageData = context.getImageData(0, 0, image.width, image.height);

        // 1. Get buffer data from image and create R, G, and B arrays.
        var imageBufferData = imageData.data;
        const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());

        // 2. Loop through the image buffer and extract the R, G, and B channels
        for (let i = 0; i < imageBufferData.length; i += 4) {
            redArray.push(imageBufferData[i]);
            greenArray.push(imageBufferData[i + 1]);
            blueArray.push(imageBufferData[i + 2]);
            // skip data[i + 3] to filter out the alpha channel
        }

        // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
        const transposedData = redArray.concat(greenArray).concat(blueArray);

        let i, l = transposedData.length;
        // create the Float32Array size 3 * 224 * 224 for these dimensions output
        const float32Data = new Float32Array(3 * image.height * image.width);
        for (i = 0; i < l; i++) {
            float32Data[i] = transposedData[i] / 255.0; // convert to float
        }
        // 5. create the tensor object from onnxruntime-web.
        const input_tensor = new ort.Tensor("float32", float32Data, [1, 3, image.width, image.height]);
        const feeds = ort.Record();
        feeds[session.inputNames[0]] = input_tensor;

        try {
            // feed inputs and run
            const results = await session.run(feeds);

            // read from results
            const dataC = results.c.data;
            document.write(`data of result tensor 'c': ${dataC}`);
        } catch (e) {
            document.getElementById('output_text').innerHTML += `<br>failed to perform inference: ${e}.`;
        }
    }
}

main();
