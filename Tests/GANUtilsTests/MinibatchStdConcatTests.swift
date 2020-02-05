import XCTest
import TensorFlow
import GANUtils

class MinibatchStdConcatTests: XCTestCase {

    func testMinibatchStdConcat() {
        let layer = MinibatchStdConcat<Float>(groupSize: 4)
        
        var images = Tensor<Float>(zeros: [8, 1, 1, 3])
        images[0] = Tensor(repeating: 100, shape: [1, 1, 3])
        images[1] = Tensor(repeating: 10, shape: [1, 1, 3])
        images[3] = Tensor(repeating: 10, shape: [1, 1, 3])
        images[4] = Tensor(repeating: 100, shape: [1, 1, 3])
        
        let out = layer(images)
        XCTAssertEqual(out.shape, [8, 1, 1, 4])
        XCTAssertEqual(out[0, 0, 0, 3], Tensor(50))
        XCTAssertEqual(out[1, 0, 0, 3], Tensor(5))
        XCTAssertEqual(out[2, 0, 0, 3], Tensor(50))
        XCTAssertEqual(out[3, 0, 0, 3], Tensor(5))
        XCTAssertEqual(out[4, 0, 0, 3], Tensor(50))
        XCTAssertEqual(out[5, 0, 0, 3], Tensor(5))
        XCTAssertEqual(out[6, 0, 0, 3], Tensor(50))
        XCTAssertEqual(out[7, 0, 0, 3], Tensor(5))
    }

}
