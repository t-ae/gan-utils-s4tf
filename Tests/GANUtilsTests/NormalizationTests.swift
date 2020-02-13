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
    
    func testInstanceNorm() {
        let norm = InstanceNorm<Float>(featureCount: 8)
        let features = Tensor<Float>(randomNormal: [2, 4, 4, 8])
        
        let output = norm(features)
        
        XCTAssertEqual(output.shape, features.shape)
    }
    
    func testConditionalBatchNorm() {
        let norm = ConditionalBatchNorm<Float>(numClass: 3, featureCount: 8)
        let features = Tensor<Float>(randomNormal: [2, 4, 4, 8])
        let labels = Tensor<Int32>([0, 1])
        
        let output = norm(.init(feature: features, label: labels))
        
        XCTAssertEqual(output.shape, features.shape)
    }
}
