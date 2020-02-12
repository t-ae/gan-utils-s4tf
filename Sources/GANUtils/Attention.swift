import Foundation
import TensorFlow

// https://github.com/brain-research/self-attention-gan/blob/ad9612e60f6ba2b5ad3d3340ebae60f724636d75/non_local.py
public struct SelfAttention<Scalar: TensorFlowFloatingPoint>: Layer {
    
    public var thetaConv: SNConv2D<Scalar>
    public var phiConv: SNConv2D<Scalar>
    public var gConv: SNConv2D<Scalar>
    public var outputConv: SNConv2D<Scalar>
    public var sigma: Tensor<Scalar>
    
    public var maxPool: MaxPool2D<Scalar>
    
    public init(
        channels: Int,
        enableSpectralNormalization: Bool = false,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform()
    ) {
        precondition(channels.isMultiple(of: 8), "`channels` must be multiple of 8.")
        
        thetaConv = SNConv2D(Conv2D(filterShape: (1, 1, channels, channels / 8),
                                    filterInitializer: filterInitializer))
        phiConv = SNConv2D(Conv2D(filterShape: (1, 1, channels, channels / 8),
                                  filterInitializer: filterInitializer))
        gConv = SNConv2D(Conv2D(filterShape: (1, 1, channels, channels / 2),
                                filterInitializer: filterInitializer))
        outputConv = SNConv2D(Conv2D(filterShape: (1, 1, channels / 2, channels),
                                     filterInitializer: filterInitializer))
        
        sigma = Tensor(0)
        
        maxPool = MaxPool2D(poolSize: (2, 2), strides: (2, 2))
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let batchSize = input.shape[0]
        let spatialSize = input.shape[1] * input.shape[2]
        let downsampledSize = spatialSize / 4
        
        // [batchSize, spatialSize, downsampleSize]
        let attention = computeAttention(input)
        
        // [batchSize, downsampleSize, channels/2]
        var g = gConv(input)
        g = maxPool(g).reshaped(to: [batchSize, downsampledSize, -1])
        
        var x = matmul(attention, g) // [batchSize, spatialSize, channels/2]
        x = x.reshaped(to: input.shape.dropLast() + [g.shape[2]])
        x = outputConv(x)
        
        return x * sigma + input
    }
    
    @differentiable
    public func computeAttention(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let batchSize = input.shape[0]
        let spatialSize = input.shape[1] * input.shape[2]
        let downsampledSize = spatialSize / 4
        
        let theta = thetaConv(input).reshaped(to: [batchSize, spatialSize, -1])
        
        var phi = phiConv(input)
        phi = maxPool(phi).reshaped(to: [batchSize, downsampledSize, -1])
        
        
        // [batchSize, spatialSize, downsampleSize]
        let attention = softmax(matmul(theta, transposed: false, phi, transposed: true))
        
        return attention
    }
}

public struct ConvolutionalBlockAttention<Scalar: TensorFlowFloatingPoint>: Layer {
    
    public var dense1: Dense<Scalar>
    public var dense2: Dense<Scalar>
    
    public var conv: Conv2D<Scalar>
    
    public init(
        channels: Int,
        initializer: ParameterInitializer<Scalar> = glorotUniform()
    ) {
        self.init(channels: channels, hiddenChannels: channels / 16, initializer: initializer)
    }
    
    public init(
        channels: Int,
        hiddenChannels: Int,
        initializer: ParameterInitializer<Scalar> = glorotUniform()
    ) {
        dense1 = Dense(inputSize: channels, outputSize: hiddenChannels, weightInitializer: initializer)
        dense2 = Dense(inputSize: hiddenChannels, outputSize: channels, weightInitializer: initializer)
        
        conv = Conv2D(filterShape: (7, 7, 2, 1), padding: .same, filterInitializer: initializer)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        var x = input
        
        var max = x.max(squeezingAxes: 1, 2)
        var avg = x.mean(squeezingAxes: 1, 2)
        
        max = dense2(relu(dense1(max)))
        avg = dense2(relu(dense1(avg)))
        
        let channelAttention = sigmoid(max + avg)
        x = x * channelAttention.expandingShape(at: 1, 2)
        
        let pooled = Tensor(concatenating: [x.max(alongAxes: 3), x.mean(alongAxes: 3)], alongAxis: 3)
        let spatialAttention = sigmoid(conv(pooled))
        x = x * spatialAttention
        
        return x
    }
}
