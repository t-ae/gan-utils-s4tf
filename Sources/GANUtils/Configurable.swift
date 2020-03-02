import Foundation
import TensorFlow

public protocol ConfigurableLayer: Layer {}

extension BatchNorm: ConfigurableLayer {}
extension LayerNorm: ConfigurableLayer {}

extension PixelNorm: ConfigurableLayer {}
extension ConditionalBatchNorm: ConfigurableLayer {}
extension InstanceNorm: ConfigurableLayer {}

public struct Configurable<L: ConfigurableLayer>: Layer where L.Input == L.Output {
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
