import XCTest
import TensorFlow
import GANUtils

class SpectralNormalizationTests: XCTestCase {

    func testSNDense() {
        let dense = SNDense<Float>(inputSize: 10, outputSize: 8)
        
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
    
    func testDifferentiability() {
        let dense = SNDense<Float>(inputSize: 10, outputSize: 8)
        let input = Tensor<Float>(randomNormal: [8, 10])
        _ = gradient(at: dense, input) { $0($1).sum() }
    }
}
