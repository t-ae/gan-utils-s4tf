import Foundation
import TensorFlow

@differentiable(wrt: (a, b))
public func lerp<Scalar: TensorFlowFloatingPoint>(
    _ a: Tensor<Scalar>,
    _ b: Tensor<Scalar>,
    rate: Scalar
) -> Tensor<Scalar> {
    let rate = min(max(rate, 0), 1)
    return a + rate * (b - a)
}

/// Create grid by interpolating given 4 tensors.
///
/// - Parameters:
///   - corners: Tensor with shape [4, ...]. `tensor[i]` will be corner of grid.
///   - gridSize: width/height of output grid.
///   - flatten: IF true, grid dims will be flattened.
/// - Returns: Grid tensor with shape [gridSize, gridSize, ...], or [gridSize*gridSize, ...] if `flatten` is `true`.
public func makeGrid<Scalar: TensorFlowFloatingPoint>(
    corners tensor: Tensor<Scalar>,
    gridSize: Int,
    flatten: Bool = false
) -> Tensor<Scalar> {
    precondition(tensor.shape[0] == 4, "`tensor.shape[0]` must be 4.")
    let (z0, z1, z2, z3) = (tensor[0], tensor[1], tensor[2], tensor[3])
    
    var zs = [Tensor<Scalar>]()
    zs.reserveCapacity(gridSize*gridSize)
    
    for y in 0..<gridSize {
        let rate = Scalar(y) / Scalar(gridSize)
        let z02 = lerp(z0, z2, rate: rate)
        let z13 = lerp(z1, z3, rate: rate)
        
        for x in 0..<gridSize {
            let rate = Scalar(x) / Scalar(gridSize)
            let z = lerp(z02, z13, rate: rate)
            zs.append(z)
        }
    }
    
    var grid = Tensor(stacking: zs)
    
    if !flatten {
        let outputShape = TensorShape([gridSize, gridSize] + tensor.shape.dropFirst())
        grid = grid.reshaped(to: outputShape)
    }
    
    return grid
}
