import Foundation
import TensorFlow

// https://arxiv.org/abs/1710.10196
@differentiable(wrt: x)
public func pixelNormalization<Scalar: TensorFlowFloatingPoint>(
    _ x: Tensor<Scalar>,
    epsilon: Scalar = 1e-8
) -> Tensor<Scalar> {
    // FIXME: mean(alongAxes: -1) cause crash.
    let mean = x.squared().mean(alongAxes: x.rank-1)
    return x * rsqrt(mean + epsilon)
}

public struct PixelNorm<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative
    private var epsilon: Scalar
    
    public init(epsilon: Scalar = 1e-8) {
        self.epsilon = epsilon
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        pixelNormalization(input, epsilon: epsilon)
    }
}

public struct InstanceNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    public var scale: Tensor<Scalar>
    public var offset: Tensor<Scalar>
    
    @noDerivative
    public let epsilon: Scalar
    
    public init(featureCount: Int, epsilon: Scalar = 1e-8) {
        scale = Tensor(ones: [featureCount])
        offset = Tensor(zeros: [featureCount])
        self.epsilon = epsilon
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank >= 3)
        let normalizationAxes = [Int](1..<input.rank-1)
        let moment = input.moments(alongAxes: normalizationAxes)
        let normalized = (input - moment.mean) * rsqrt(moment.variance + epsilon)
        
        return scale * normalized + offset
    }
}

public struct ConditionalBatchNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    public struct Input: Differentiable {
        public var feature: Tensor<Scalar>
        @noDerivative
        public var label: Tensor<Int32>
        
        @differentiable
        public init(feature: Tensor<Scalar>, label: Tensor<Int32>) {
            self.feature = feature
            self.label = label
        }
    }
    
    @noDerivative
    public let featureCount: Int
    
    public var gammaEmb: Embedding<Scalar>
    public var betaEmb: Embedding<Scalar>
    
    @noDerivative
    public let momentum: Scalar
    @noDerivative
    public let epsilon: Scalar
    
    @noDerivative
    public var runningMean: Parameter<Scalar>
    @noDerivative
    public var runningVariance: Parameter<Scalar>
    
    public init(
        numClass: Int,
        featureCount: Int,
        momentum: Scalar = 0.99,
        epsilon: Scalar = 1e-3
    ) {
        self.featureCount = featureCount
        self.gammaEmb = Embedding(embeddings: Tensor<Scalar>(ones: [numClass, featureCount]))
        self.betaEmb = Embedding(embeddings: Tensor<Scalar>(zeros: [numClass, featureCount]))
        
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.runningMean = Parameter(Tensor(0))
        self.runningVariance = Parameter(Tensor(1))
    }
    
    @differentiable
    public func callAsFunction(_ input: Input) -> Tensor<Scalar> {
        let gamma = gammaEmb(input.label).expandingShape(at: 1, 2) // [batchSize, featureCount]
        let beta = betaEmb(input.label).expandingShape(at: 1, 2) // [batchSize, featureCount]
        
        let x = input.feature
        switch Context.local.learningPhase {
        case .training:
          let normalizedAxes = Array(0..<x.rank-1) // Exclude last feature axis
          let moments = x.moments(alongAxes: normalizedAxes)
          let decayMomentum = Tensor(1 - momentum, on: x.device)
          runningMean.value += (moments.mean - runningMean.value) * decayMomentum
          runningVariance.value += (moments.variance - runningVariance.value) * decayMomentum
          let inv = rsqrt(moments.variance + Tensor(epsilon, on: x.device)) * gamma
          return (x - moments.mean) * inv + beta
        case .inference:
          let inv = rsqrt(runningVariance.value + Tensor(epsilon, on: x.device)) * gamma
          return (x - runningMean.value) * inv + beta
        }
    }
}
