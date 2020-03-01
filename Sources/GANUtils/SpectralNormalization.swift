import TensorFlow

private func l2normalize<Scalar: TensorFlowFloatingPoint>(_ tensor: Tensor<Scalar>) -> Tensor<Scalar> {
    tensor * rsqrt(tensor.squared().sum() + 1e-8)
}

// https://github.com/pfnet-research/sngan_projection/blob/master/source/links/sn_convolution_2d.py
public struct SNDense<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The weight matrix.
    public var weight: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative
    public let activation: Activation
    /// Workaround optionals not being handled by AD
    @noDerivative
    private let useBias: Bool
    
    @noDerivative
    public var enabled: Bool
    
    @noDerivative
    public var numPowerIterations: Int
    
    @noDerivative
    public let v: Parameter<Scalar>

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    public init(
        weight: Tensor<Scalar>,
        bias: Tensor<Scalar>? = nil,
        activation: @escaping Activation,
        numPowerIterations: Int = 1,
        enabled: Bool = true
    ) {
        precondition(weight.rank == 2, "The rank of the 'weight' tensor must be 2.")
        self.weight = weight
        self.bias = bias ?? .zero
        self.activation = activation
        useBias = (bias != nil)
        
        self.numPowerIterations = numPowerIterations
        self.enabled = enabled
        self.v = Parameter(Tensor(randomNormal: [1, weight.shape[1]]))
    }
    
    @differentiable
    public func wBar() -> Tensor<Scalar> {
        guard enabled else {
            return weight
        }
        let outputDim = weight.shape[1]
        let mat = weight.reshaped(to: [-1, outputDim])
        
        var u = Tensor<Scalar>(0)
        var v = withoutDerivative(at: self.v.value)
        for _ in 0..<numPowerIterations {
            u = l2normalize(matmul(v, mat.transposed())) // [1, rows]
            v = l2normalize(matmul(u, mat)) // [1, cols]
        }
        
        let sigma = matmul(matmul(u, mat), v.transposed()) // [1, 1]
        
        if Context.local.learningPhase == .training {
            self.v.value = v
        }
        
        // Should detach sigma?
        return weight / sigma
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let weight = wBar()
        return activation(useBias ? (matmul(input, weight) + bias) : matmul(input, weight))
    }
}

public extension SNDense {
    /// Creates a `Dense` layer with the specified input size, output size, and element-wise
    /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
    /// the bias vector is created with shape `[outputSize]`.
    ///
    /// - Parameters:
    ///   - inputSize: The dimensionality of the input space.
    ///   - outputSize: The dimensionality of the output space.
    ///   - activation: The activation function to use. The default value is `identity(_:)`.
    ///   - weightInitializer: Initializer to use for `weight`.
    ///   - biasInitializer: Initializer to use for `bias`.
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


public struct SNConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    public var conv: Conv2D<Scalar>
    
    @noDerivative
    public var enabled: Bool
    
    @noDerivative
    public var numPowerIterations: Int
    
    @noDerivative
    public let v: Parameter<Scalar>
    
    public init(
        _ conv: Conv2D<Scalar>,
        numPowerIterations: Int = 1,
        enabled: Bool = true
    ) {
        self.conv = conv
        self.enabled = enabled
        self.numPowerIterations = numPowerIterations
        v = Parameter(Tensor(randomNormal: [1, conv.filter.shape[3]]))
    }
    
    @differentiable
    public func wBar() -> Tensor<Scalar> {
        guard enabled else {
            return conv.filter
        }
        let outputDim = conv.filter.shape[3]
        let mat = conv.filter.reshaped(to: [-1, outputDim])
        
        var u = Tensor<Scalar>(0)
        var v = withoutDerivative(at: self.v.value)
        for _ in 0..<numPowerIterations {
            u = l2normalize(matmul(v, mat.transposed())) // [1, rows]
            v = l2normalize(matmul(u, mat)) // [1, cols]
        }
        
        let sigma = matmul(matmul(u, mat), v.transposed()) // [1, 1]
        
        if Context.local.learningPhase == .training {
            self.v.value = v
        }
        
        // Should detach sigma?
        return conv.filter / sigma
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        conv.activation(conv2D(
            input,
            filter: wBar(),
            strides: (1, conv.strides.0, conv.strides.1, 1),
            padding: conv.padding,
            dilations: (1, conv.dilations.0, conv.dilations.1, 1)) + conv.bias)
    }
}
