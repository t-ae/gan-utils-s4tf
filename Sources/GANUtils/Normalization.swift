import Foundation
import TensorFlow

// https://arxiv.org/abs/1710.10196
@differentiable(wrt: x)
public func pixelNormalization(_ x: Tensor<Float>, epsilon: Float = 1e-8) -> Tensor<Float> {
    // FIXME: mean(alongAxes: -1) cause crash.
    let mean = x.squared().mean(alongAxes: x.rank-1)
    return x * rsqrt(mean + epsilon)
}

public struct InstanceNorm<Scalar: TensorFlowFloatingPoint>: Layer {
    public var scale: Tensor<Scalar>
    public var offset: Tensor<Scalar>
    
    public init(featureCount: Int) {
        scale = Tensor(ones: [featureCount])
        offset = Tensor(zeros: [featureCount])
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let normalizationAxes = [Int](1..<input.rank)
        let mean = input.mean(alongAxes: normalizationAxes)
        let variance = squaredDifference(input, mean).mean(alongAxes: normalizationAxes)
        let normalized = (input - mean) * rsqrt(variance + 1e-8)
        
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
