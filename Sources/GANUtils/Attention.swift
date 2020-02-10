import Foundation
import TensorFlow

public struct SelfAttention<Scalar: TensorFlowFloatingPoint>: Layer {
    
    public var queryConv: Conv2D<Scalar>
    public var keyConv: Conv2D<Scalar>
    public var valueConv: Conv2D<Scalar>
    public var gamma: Tensor<Scalar>
    
    public init(
        channels: Int,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform()
    ) {
        self.init(channels: channels,
                  hiddenChannels: channels / 8,
                  filterInitializer: filterInitializer)
    }
    
    public init(
        channels: Int,
        hiddenChannels: Int,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform()
    ) {
        queryConv = Conv2D(filterShape: (1, 1, channels, hiddenChannels),
                           filterInitializer: filterInitializer)
        keyConv = Conv2D(filterShape: (1, 1, channels, hiddenChannels),
                         filterInitializer: filterInitializer)
        valueConv = Conv2D(filterShape: (1, 1, channels, channels),
                           filterInitializer: filterInitializer)
        gamma = Tensor(0)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        let batchSize = input.shape[0]
        let spatialSize = input.shape[1] * input.shape[2]
        
        let attention = computeAttention(input)
        
        // [batchSize, spatialSize, channels]
        let value = valueConv(input).reshaped(to: [batchSize, spatialSize, -1])
        
        return matmul(attention, value).reshaped(to: input.shape) * gamma + input
    }
    
    @differentiable
    public func computeAttention(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        let batchSize = input.shape[0]
        let spatialSize = input.shape[1] * input.shape[2]
        
        let query = queryConv(input).reshaped(to: [batchSize, spatialSize, -1])
        let key = keyConv(input).reshaped(to: [batchSize, spatialSize, -1])
        // [batchSize, spatialSize, spatialSize]
        let attention = softmax(matmul(query, transposed: false, key, transposed: true))
        
        return attention
    }
}
