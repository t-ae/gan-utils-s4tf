import XCTest
import TensorFlow
import GANUtils

class AttentionTests: XCTestCase {

    func testSelfAttentionForward() {
        let layer = SelfAttention<Float>(channels: 16)
        var input = Tensor<Float>(ones: [1, 4, 4, 16])
        input[0, 0, 0] *= 10
        input[0, 1, 1] *= 10
        
        let output = layer(input)
        XCTAssertEqual(output.shape, input.shape)
    }
    
    func testConvolutionalBlockAttentionForward() {
        let layer = ConvolutionalBlockAttention<Float>(channels: 32)
        var input = Tensor<Float>(ones: [1, 4, 4, 32])
        input[0, 0, 0] *= 10
        input[0, 1, 1] *= 10
        
        let output = layer(input)
        XCTAssertEqual(output.shape, input.shape)
    }

}
