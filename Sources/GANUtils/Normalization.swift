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
    
    public var bn: BatchNorm<Scalar>
    
    public var gammaEmb: Embedding<Scalar>
    public var betaEmb: Embedding<Scalar>
    
    public init(featureCount: Int) {
        self.featureCount = featureCount
        self.bn = BatchNorm(featureCount: featureCount)
        self.gammaEmb = Embedding(embeddings: Tensor<Scalar>(ones: [10, featureCount]))
        self.betaEmb = Embedding(embeddings: Tensor<Scalar>(zeros: [10, featureCount]))
    }
    
    @differentiable
    public func callAsFunction(_ input: Input) -> Tensor<Scalar> {
        let x = bn(input.feature)
        
        let gamma = gammaEmb(input.label).expandingShape(at: 1, 2)
        let beta = betaEmb(input.label).expandingShape(at: 1, 2)
        
        return x  * gamma + beta
    }
}
