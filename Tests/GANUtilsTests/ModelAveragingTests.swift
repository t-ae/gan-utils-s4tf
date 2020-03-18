import XCTest
import TensorFlow
import GANUtils

class ModelAveragingTests: XCTestCase {

    func testAverage() {
        struct Model: Layer {
            var dense1 = Dense<Float>(weight: Tensor<Float>(zeros: [10, 10]),
                                      activation: identity)
            var dense2 = Dense<Float>(weight: Tensor<Float>(zeros: [10, 1]),
                                      activation: identity)
            var param = Parameter(Tensor<Float>(zeros: [3, 3]))
            
            @differentiable
            func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
                dense2(dense1(input)) * param.value
            }
        }
        
        var model = Model()
        
        let avg = ModelAveraging(average: model, beta: 0.95)
        XCTAssertEqual(avg.average.dense1.weight, model.dense1.weight)
        XCTAssertEqual(avg.average.dense2.weight, model.dense2.weight)
        XCTAssertEqual(avg.average.param.value, model.param.value)
        
        model.dense1.weight = Tensor<Float>(ones: [10, 10])
        model.dense2.weight = Tensor<Float>(ones: [10, 1])
        model.param.value = Tensor<Float>(ones: [3, 3])
        avg.update(model: model)
        
        XCTAssert(avg.average.dense1.weight.isAlmostEqual(to: Tensor(repeating: 0.05, shape: [10, 10])))
        XCTAssert(avg.average.dense2.weight.isAlmostEqual(to: Tensor(repeating: 0.05, shape: [10, 1])))
        print(model.param)
        print(avg.average.param.value)
        XCTAssert(avg.average.param.value.isAlmostEqual(to: Tensor(repeating: 0.05, shape: [3, 3])))
    }
}
