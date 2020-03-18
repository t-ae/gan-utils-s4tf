import Foundation
import TensorFlow

// https://arxiv.org/abs/1806.04498
public class ModelAveraging<Model: Layer> {
    public var average: Model
    
    public let beta: Float
    
    public init(average: Model, beta: Float = 0.95) {
        self.average = average
        self.beta = beta
        
        // Separate `Parameter`s from base model
        for kp in self.average.recursivelyAllWritableKeyPaths(to: Parameter<Float>.self) {
            self.average[keyPath: kp] = Parameter(average[keyPath: kp].value)
        }
        for kp in average.recursivelyAllWritableKeyPaths(to: Parameter<Double>.self) {
            self.average[keyPath: kp] = Parameter(average[keyPath: kp].value)
        }
    }
    
    public func update(model: Model) {
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            average[keyPath: kp] = lerp(model[keyPath: kp], average[keyPath: kp], rate: beta)
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
            average[keyPath: kp] = lerp(model[keyPath: kp], average[keyPath: kp], rate: Double(beta))
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Parameter<Float>.self) {
            average[keyPath: kp].value = lerp(model[keyPath: kp].value,
                                              average[keyPath: kp].value,
                                              rate: beta)
        }
        for kp in model.recursivelyAllWritableKeyPaths(to: Parameter<Double>.self) {
            average[keyPath: kp].value = lerp(model[keyPath: kp].value,
                                              average[keyPath: kp].value,
                                              rate: Double(beta))
        }
    }
}
