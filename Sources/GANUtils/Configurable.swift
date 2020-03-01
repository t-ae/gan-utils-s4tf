import Foundation
import TensorFlow

public struct Configurable<L: Layer>: Layer where L.Input == L.Output {
    public var layer: L
    
    @noDerivative
    public var enabled: Bool
    
    public init(_ layer: L, enabled: Bool) {
        self.layer = layer
        self.enabled = enabled
    }

    @differentiable
    public func callAsFunction(_ input: L.Input) -> L.Output {
        if enabled {
            return layer(input)
        } else {
            return input
        }
    }
}
