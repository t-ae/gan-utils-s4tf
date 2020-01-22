import Foundation
import TensorFlow

// https://arxiv.org/abs/1710.10196
@differentiable(wrt: x)
public func pixelNormalization(_ x: Tensor<Float>, epsilon: Float = 1e-8) -> Tensor<Float> {
    let mean = x.squared().mean(alongAxes: -1)
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
