import Foundation
import TensorFlow

@differentiable(wrt: tensor)
public func depthToSpace<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> Tensor<Scalar> {
    _Raw.depthToSpace(tensor, blockSize: Int64(blockSize))
}

@inlinable
@derivative(of: depthToSpace)
func vjpDepthToSpace<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>)->Tensor<Scalar>) {
    let result = depthToSpace(tensor, blockSize: blockSize)
    return (result, { v in
        spaceToDepth(v, blockSize: blockSize)
    })
}

@differentiable(wrt: tensor)
public func spaceToDepth<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> Tensor<Scalar> {
    _Raw.spaceToDepth(tensor, blockSize: Int64(blockSize))
}

@inlinable
@derivative(of: spaceToDepth)
func vjpSpaceToDepth<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>)->Tensor<Scalar>) {
    let result = spaceToDepth(tensor, blockSize: blockSize)
    return (result, { v in
        depthToSpace(v, blockSize: blockSize)
    })
}
