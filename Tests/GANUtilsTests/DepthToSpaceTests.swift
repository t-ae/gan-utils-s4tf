import XCTest
import TensorFlow
import GANUtils

class DepthToSpaceTests: XCTestCase {
    func testDepthToSpace() {
        let tensor = Tensor<Float>([1, 2, 3, 4]).reshaped(to: [1, 1, 1, 4])
        let result = depthToSpace(tensor, blockSize: 2)
        XCTAssertEqual(result, tensor.reshaped(to: [1, 2, 2, 1]))
    }
    
    func testDepthToSpaceGrad() {
        let tensor = Tensor<Float>([1, 2, 3, 4]).reshaped(to: [1, 1, 1, 4])
        let g = gradient(at: tensor) { tensor in
            depthToSpace(tensor, blockSize: 2).sum()
        }
        XCTAssertEqual(g, Tensor(ones: tensor.shape))
    }

    func testSpaceToDepth() {
        let tensor = Tensor<Float>([1, 2, 3, 4]).reshaped(to: [1, 2, 2, 1])
        let result = spaceToDepth(tensor, blockSize: 2)
        XCTAssertEqual(result, tensor.reshaped(to: [1, 1, 1, 4]))
    }
    
    func testSpaceToDepthGrad() {
        let tensor = Tensor<Float>([1, 2, 3, 4]).reshaped(to: [1, 2, 2, 1])
        let g = gradient(at: tensor) { tensor in
            spaceToDepth(tensor, blockSize: 2).sum()
        }
        XCTAssertEqual(g, Tensor(ones: tensor.shape))
    }
}
