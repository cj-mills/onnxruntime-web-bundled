// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// import { getImageTensorFromPath } from './imageHelper';
// import * as Jimp from 'jimp';
const ort = require('onnxruntime-web');
var session = ort.InferenceSession;

// The mean of the ImageNet dataset used to train the model
const mean = [0.485, 0.456, 0.406];
// The standard deviation of the ImageNet dataset used to train the model
const std_dev = [0.229, 0.224, 0.225];

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

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

//The softmax transforms values to be between 0 and 1
function softmax(resultArray) {
    // Get the largest value in the array.
    const largestNumber = Math.max(...resultArray);
    // Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
    const sumOfExp = resultArray.map((resultItem) => Math.exp(resultItem - largestNumber)).reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
    //Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
    return resultArray.map((resultValue, index) => {
        return Math.exp(resultValue - largestNumber) / sumOfExp;
    });
}

// use an async context to call onnxruntime functions.
async function main() {

    // var image = document.getElementById('image');
    var div = document.createElement("DIV");
    div.id = 'output_text';

    document.body.appendChild(div);

    var model_dir = './models';
    var model_path = `${model_dir}/asl-and-some-words-resnet34.onnx`;
    var exec_provider = 'wasm';
    var return_msg = await init_session(model_path, exec_provider);
    console.log(`Input Name: ${session.inputNames[0]}`);

    document.getElementById('output_text').innerHTML += `<br>${(await return_msg).toString()}`;

    image = await new Image();
    image.src = 'https://cj-mills.github.io/onnxruntime-web-bundled/images/Play_1291.jpg';
    div.innerHTML = `Image Source: ${image.src}`;

    var canvas = document.createElement("CANVAS");
    var context = canvas.getContext('2d');
    document.body.appendChild(canvas);
    image.onload = () => {
        canvas.width = image.width;
        canvas.height = image.height;
        context.drawImage(image, 0, 0);
    }

    var imageData = context.getImageData(0, 0, image.width, image.height);

    // 1. Get buffer data from image.
    var imageBufferData = imageData.data;
    console.log(`Image Data: ${imageBufferData}`);


    var n_pixels = image.width * image.height;
    var n_channels = 3;
    const float32Data = new Float32Array(3 * image.height * image.width);

    for (p = 0; p < n_pixels; p++) {
        float32Data[0 * n_pixels + p] = ((imageBufferData[p * n_channels + 0] / 255.0) - mean[0]) / std_dev[0];
        float32Data[1 * n_pixels + p] = ((imageBufferData[p * n_channels + 1] / 255.0) - mean[1]) / std_dev[1];
        float32Data[2 * n_pixels + p] = ((imageBufferData[p * n_channels + 2] / 255.0) - mean[2]) / std_dev[2];
    }

    console.log(`Input Data: ${float32Data}`);

    // 5. create the tensor object from onnxruntime-web.
    const input_tensor = new ort.Tensor("float32", float32Data, [1, 3, image.height, image.width]);
    const feeds = {};
    feeds[session.inputNames[0]] = input_tensor;


    // feed inputs and run
    const outputData = await session.run(feeds);

    // read from results
    const output = outputData[session.outputNames[0]];
    var results = softmax(Array.prototype.slice.call(output.data));
    console.log(`Predictions: ${results}`);
    var index = argMax(results);
    document.getElementById('output_text').innerHTML += `<br>Predicted class index: ${index}`;
}

main();
