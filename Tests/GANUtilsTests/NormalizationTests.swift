import XCTest
import TensorFlow
import GANUtils

class NormalizationTests: XCTestCase {
    func testPixelNorm() {
        do {
            let length = 32
            let tensor = Tensor<Float>(randomNormal: [1, length*length])
            
            let norm = pixelNormalization(tensor)
            
            let len = sqrt(norm.squared().sum())
            
            XCTAssert(len.isAlmostEqual(to: Tensor(Float(length))))
        }
        do {
            let length = 32
            let tensor = Tensor<Float>(randomNormal: [4, 8, 8, length*length])
            
            let norm = pixelNormalization(tensor)
            
            let len = sqrt(norm.squared().sum(alongAxes: -1))
            
            XCTAssert(len.isAlmostEqual(to: Tensor(repeating: Float(length), shape: len.shape)))
        }
    }
    func testPixelNormGrad() {
        let length = 32
        let tensor = Tensor<Float>(randomNormal: [1, length*length])
        
        let g = gradient(at: tensor) { tensor -> Tensor<Float> in
            pixelNormalization(tensor).sum()
        }
        print(g)
    }
}
