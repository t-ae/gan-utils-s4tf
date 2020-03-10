import Foundation
import TensorFlow

public class ModelAveraging<Model: Layer> where Model.TangentVector.VectorSpaceScalar == Float {
    public var average: Model
    
    public let beta: Float
    
    public init(average: Model, beta: Float = 0.95) {
        self.average = average
        self.beta = beta
    }
    
    public func update(model: Model) {
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            average[keyPath: kp] = lerp(model[keyPath: kp], average[keyPath: kp], rate: beta)
        }
    }
}
