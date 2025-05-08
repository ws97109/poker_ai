// model.js
async function createModel() {
    const model = tf.sequential();
    
    // 第一層 Dense (7 -> 32)
    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
        inputShape: [7],
        weights: [
            tf.tensor2d(processedWeights.dense.kernel, [7, 32]),
            tf.tensor1d(processedWeights.dense.bias)
        ]
    }));
    
    // 第二層 Dense (32 -> 32)
    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
        weights: [
            tf.tensor2d(processedWeights.dense_1.kernel, [32, 32]),
            tf.tensor1d(processedWeights.dense_1.bias)
        ]
    }));
    
    // 輸出層 (32 -> 41)
    model.add(tf.layers.dense({
        units: 41,
        activation: 'linear',
        weights: [
            tf.tensor2d(processedWeights.dense_2.kernel, [32, 41]),
            tf.tensor1d(processedWeights.dense_2.bias)
        ]
    }));

    return model;
}