import Foundation
import TensorFlow

@differentiable(wrt: (a, b))
public func lerp(_ a: Tensor<Float>, _ b: Tensor<Float>, rate: Float) -> Tensor<Float> {
    let rate = min(max(rate, 0), 1)
    return a + rate * (b - a)
}
