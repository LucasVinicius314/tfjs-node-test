import * as tf from '@tensorflow/tfjs'

import { Car } from '../typescript/index'
import { ChartJSNodeCanvas } from 'chartjs-node-canvas'
import fetch from 'node-fetch'

const trainModel = async (model: any, inputs: any, labels: any) => {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  })

  const batchSize = 32
  const epochs = 50

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
  })
}

const createModel = () => {
  // Create a sequential model
  const model = tf.sequential()
  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }))
  return model
}

const getData = async () => {
  const carsDataResponse = await fetch(
    'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
  )
  const carsData = (await carsDataResponse.json()) as Car[]
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null)

  return cleaned
}

const convertToTensor = (data) => {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data)

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horsepower)
    const labels = data.map((d) => d.mpg)

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max()
    const inputMin = inputTensor.min()
    const labelMax = labelTensor.max()
    const labelMin = labelTensor.min()

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin))

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  })
}

const testModel = (model: any, inputData: any, normalizationData: any) => {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100)
    const preds = model.predict(xs.reshape([100, 1]))

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin)

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin)

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()]
  })

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  })

  const originalPoints = inputData.map((d: any) => ({
    x: d.horsepower,
    y: d.mpg,
  }))

  console.log(originalPoints)
  console.log(predictedPoints)
}

const main = async () => {
  const model = createModel()

  const data = await getData()

  const tensorData = convertToTensor(data)
  const { inputs, labels } = tensorData

  // Train the model
  await trainModel(model, inputs, labels)
  console.log('Done Training')
  testModel(model, data, tensorData)

  const chartJSNodeCanvas = new ChartJSNodeCanvas({
    width: 512,
    height: 512,
    type: 'pdf',
  })

  console.log(
    await chartJSNodeCanvas.renderToDataURL({
      type: 'pie',
      data: {
        datasets: [{ data: [2] }],
        labels: ['a', 'b', 'c'],
      },
    })
  )

  console.log()
}

main()
