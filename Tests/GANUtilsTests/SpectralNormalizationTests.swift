import XCTest
import TensorFlow
import GANUtils

class SpectralNormalizationTests: XCTestCase {

    func testSpectralNorm() {
        let dense = SNDense<Float>(Dense(inputSize: 10, outputSize: 8))
        
        let input = Tensor<Float>(randomNormal: [8, 10])
        
        let output = dense(input)
        XCTAssertEqual(output.shape, [8, 8])
        
        Context.local.learningPhase = .training
        for _ in 0..<100 {
            // Update v
            let input = Tensor<Float>(randomNormal: [8, 10])
            _ = dense(input)
        }
        let weight = dense.wBar()
        let svd = _Raw.svd(weight)
        
        XCTAssertEqual(svd.s[0].scalarized(), 1, accuracy: 1e-3)
        
        Context.local.learningPhase = .inference
        
    }

}
