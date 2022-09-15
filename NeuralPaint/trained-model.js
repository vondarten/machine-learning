import {MnistData} from './assets/data.js';

document.querySelector('.btn-load-tf').addEventListener('click', run)
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data){
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(500);
    const testxs = testData.xs.reshape([500, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);
    testxs.dispose();
    return [preds, labels];
}

async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = {name: 'Accuracy', tab: 'Evaluation'};
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
    labels.dispose();
  }

async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
    tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

    labels.dispose();
}

async function getData() {
    const data = new MnistData();
    await data.load();
    return data
}

async function run() {
    
    const uploadJSONInput = document.getElementById('upload-json');
    const uploadWeightsInput = document.getElementById('upload-weights');

    const model = await tf.loadLayersModel(tf.io.browserFiles(
        [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));

    tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model)
    const data = await getData()
    await showAccuracy(model, data);
    await showConfusion(model, data);

}
