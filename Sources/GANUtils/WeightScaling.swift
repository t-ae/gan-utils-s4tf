import Foundation
import TensorFlow

// Equalized lerning rate
// https://arxiv.org/abs/1710.10196

public struct WSDense<Scalar: TensorFlowFloatingPoint>: Layer {
    public var weight: Tensor<Scalar>
    @noDerivative
    public let scale: Scalar
    public var bias: Tensor<Scalar>
    @noDerivative
    public let activation: Activation
    @noDerivative
    internal let batched: Bool
    @noDerivative
    private let useBias: Bool

    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    public init(
        weight: Tensor<Scalar>,
        bias: Tensor<Scalar>? = nil,
        activation: @escaping Activation
    ) {
        precondition(weight.rank <= 3, "The rank of the 'weight' tensor must be less than 4.")
        precondition(bias == nil || bias!.rank <= 2, "The rank of the 'bias' tensor must be less than 3.")
        scale = weight.standardDeviation().scalarized()
        self.weight = weight / scale
        
        self.bias = bias ?? .zero
        self.activation = activation
        self.batched = weight.rank == 3
        useBias = (bias != nil)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        if batched {
            let hidden = matmul(input.expandingShape(at: 1), weight).squeezingShape(at: 1)
            return activation(useBias ? hidden + bias : hidden)
        }
        let weight = self.weight * scale
        return activation(useBias ? (matmul(input, weight) + bias) : matmul(input, weight))
    }
}

public extension WSDense {
    init(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        useBias: Bool = true,
        weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        self.init(
            weight: weightInitializer([inputSize, outputSize]),
            bias: useBias ? biasInitializer([outputSize]) : nil,
            activation: activation)
    }
}


public struct Conv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    public var filter: Tensor<Scalar>
    @noDerivative
    public let scale: Scalar
    public var bias: Tensor<Scalar>
    @noDerivative
    public let activation: Activation
    @noDerivative
    public let strides: (Int, Int)
    @noDerivative
    public let padding: Padding
    @noDerivative
    public let dilations: (Int, Int)
    @noDerivative
    private let useBias: Bool

    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>? = nil,
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1)
    ) {
        scale = filter.standardDeviation().scalarized()
        self.filter = filter / scale
        
        self.bias = bias ?? .zero
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        useBias = (bias != nil)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let conv = conv2D(
            input,
            filter: scale * filter,
            strides: (1, strides.0, strides.1, 1),
            padding: padding,
            dilations: (1, dilations.0, dilations.1, 1))
        return activation(useBias ? (conv + bias) : conv)
    }
}

public extension Conv2D {
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        activation: @escaping Activation = identity,
        useBias: Bool = true,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: useBias ? biasInitializer([filterShape.3]) : nil,
            activation: activation,
            strides: strides,
            padding: padding,
            dilations: dilations)
    }
}
