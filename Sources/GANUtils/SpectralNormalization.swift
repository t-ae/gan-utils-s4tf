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
    public let useBias: Bool
    
    @noDerivative
    public var spectralNormalizationEnabled: Bool
    
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
        spectralNormalizationEnabled: Bool = true
    ) {
        precondition(weight.rank == 2, "The rank of the 'weight' tensor must be 2.")
        self.weight = weight
        self.bias = bias ?? .zero
        self.activation = activation
        useBias = (bias != nil)
        
        self.numPowerIterations = numPowerIterations
        self.spectralNormalizationEnabled = spectralNormalizationEnabled
        self.v = Parameter(Tensor(randomNormal: [1, weight.shape[1]]))
    }
    
    @differentiable
    public func wBar() -> Tensor<Scalar> {
        guard spectralNormalizationEnabled else {
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
        u = withoutDerivative(at: u)
        v = withoutDerivative(at: v)
        
        let sigma = matmul(matmul(u, mat), v.transposed()) // [1, 1]
        
        if Context.local.learningPhase == .training {
            self.v.value = v
        }
        
        return weight / sigma
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let weight = wBar()
        return activation(useBias ? (matmul(input, weight) + bias) : matmul(input, weight))
    }
}

public extension SNDense {
    init(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        useBias: Bool = true,
        numPowerIterations: Int = 1,
        spectralNormalizationEnabled: Bool = true,
        weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        self.init(
            weight: weightInitializer([inputSize, outputSize]),
            bias: useBias ? biasInitializer([outputSize]) : nil,
            activation: activation,
            numPowerIterations: numPowerIterations,
            spectralNormalizationEnabled: spectralNormalizationEnabled)
    }
}

public struct SNConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The 4-D convolution filter.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative
    public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative
    public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative
    public let padding: Padding
    /// Note: `useBias` is a workaround for TF-1153: optional differentiation support.
    @noDerivative
    public let useBias: Bool
    
    @noDerivative
    public var spectralNormalizationEnabled: Bool
    
    @noDerivative
    public var numPowerIterations: Int
    
    @noDerivative
    public let v: Parameter<Scalar>

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>? = nil,
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        numPowerIterations: Int = 1,
        spectralNormalizationEnabled: Bool = true
    ) {
        self.filter = filter
        self.bias = bias ?? .zero
        self.activation = activation
        self.strides = strides
        self.padding = padding
        useBias = (bias != nil)
        
        self.numPowerIterations = numPowerIterations
        self.spectralNormalizationEnabled = spectralNormalizationEnabled
        v = Parameter(Tensor(randomNormal: [1, filter.shape[3]]))
    }
    
    @differentiable
    public func wBar() -> Tensor<Scalar> {
        guard spectralNormalizationEnabled else {
            return filter
        }
        let outputDim = filter.shape[3]
        let mat = filter.reshaped(to: [-1, outputDim])
        
        var u = Tensor<Scalar>(0)
        var v = withoutDerivative(at: self.v.value)
        for _ in 0..<numPowerIterations {
            u = l2normalize(matmul(v, mat.transposed())) // [1, rows]
            v = l2normalize(matmul(u, mat)) // [1, cols]
        }
        u = withoutDerivative(at: u)
        v = withoutDerivative(at: v)
        
        let sigma = matmul(matmul(u, mat), v.transposed()) // [1, 1]
        
        if Context.local.learningPhase == .training {
            self.v.value = v
        }
        
        return filter / sigma
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let filter = wBar()
        let conv = conv2D(
            input,
            filter: filter,
            strides: (1, strides.0, strides.1, 1),
            padding: padding,
            dilations: (1, 1, 1, 1))
        return activation(useBias ? (conv + bias) : conv)
    }
}

public extension SNConv2D {
    init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        useBias: Bool = true,
        numPowerIterations: Int = 1,
        spectralNormalizationEnabled: Bool = true,
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
            numPowerIterations: numPowerIterations,
            spectralNormalizationEnabled: spectralNormalizationEnabled)
    }
}

public struct SNTransposedConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    public var filter: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    @noDerivative public let activation: Activation
    @noDerivative public let strides: (Int, Int)
    @noDerivative public let padding: Padding
    @noDerivative public let paddingIndex: Int
    @noDerivative private let useBias: Bool
    @noDerivative
    public var spectralNormalizationEnabled: Bool
    @noDerivative
    public var numPowerIterations: Int
    @noDerivative
    public let v: Parameter<Scalar>
    
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>? = nil,
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        numPowerIterations: Int = 1,
        spectralNormalizationEnabled: Bool = true
    ) {
        self.filter = filter
        self.bias = bias ?? .zero
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.paddingIndex = padding == .same ? 0 : 1
        useBias = (bias != nil)
        self.numPowerIterations = numPowerIterations
        self.spectralNormalizationEnabled = spectralNormalizationEnabled
        v = Parameter(Tensor(randomNormal: [1, filter.shape[3]]))
    }
    
    @differentiable
    public func wBar() -> Tensor<Scalar> {
        guard spectralNormalizationEnabled else {
            return filter
        }
        let outputDim = filter.shape[3]
        let mat = filter.reshaped(to: [-1, outputDim])
        
        var u = Tensor<Scalar>(0)
        var v = withoutDerivative(at: self.v.value)
        for _ in 0..<numPowerIterations {
            u = l2normalize(matmul(v, mat.transposed())) // [1, rows]
            v = l2normalize(matmul(u, mat)) // [1, cols]
        }
        u = withoutDerivative(at: u)
        v = withoutDerivative(at: v)
        
        let sigma = matmul(matmul(u, mat), v.transposed()) // [1, 1]
        
        if Context.local.learningPhase == .training {
            self.v.value = v
        }
        
        return filter / sigma
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let batchSize = input.shape[0]
        let h = (input.shape[1] - (1 * paddingIndex)) * strides.0 + (filter.shape[0] * paddingIndex)
        let w = (input.shape[2] - (1 * paddingIndex)) * strides.1 + (filter.shape[1] * paddingIndex)
        let c = filter.shape[2]
//        let newShape = [Int64(batchSize), Int64(h), Int64(w), Int64(c)]
        let newShape = Tensor<Int32>([Int32(batchSize), Int32(h), Int32(w), Int32(c)])
        let conv = transposedConv2D(
            input,
            shape: newShape,
            filter: wBar(),
            strides: (1, strides.0, strides.1, 1),
            padding: padding)
        return activation(useBias ? (conv + bias) : conv)
    }
}

extension SNTransposedConv2D {
    public init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        activation: @escaping Activation = identity,
        useBias: Bool = true,
        numPowerIterations: Int = 1,
        spectralNormalizationEnabled: Bool = true,
        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros()
    ) {
        let filterTensorShape = TensorShape([
            filterShape.0, filterShape.1, filterShape.2, filterShape.3,
        ])
        self.init(
            filter: filterInitializer(filterTensorShape),
            bias: useBias ? biasInitializer([filterShape.2]) : nil,
            activation: activation,
            strides: strides,
            padding: padding,
            numPowerIterations: numPowerIterations,
            spectralNormalizationEnabled: spectralNormalizationEnabled)
    }
}
