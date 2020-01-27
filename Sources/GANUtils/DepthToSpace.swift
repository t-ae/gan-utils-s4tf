import Foundation
import TensorFlow

@differentiable(wrt: tensor, vjp: vjpDepthToSpace)
public func depthToSpace<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> Tensor<Scalar> {
    _Raw.depthToSpace(tensor, blockSize: Int64(blockSize))
}

@usableFromInline
func vjpDepthToSpace<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>)->Tensor<Scalar>) {
    let result = depthToSpace(tensor, blockSize: blockSize)
    return (result, { v in
        spaceToDepth(v, blockSize: blockSize)
    })
}

@differentiable(wrt: tensor, vjp: vjpSpaceToDepth)
public func spaceToDepth<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> Tensor<Scalar> {
    _Raw.spaceToDepth(tensor, blockSize: Int64(blockSize))
}

@usableFromInline
func vjpSpaceToDepth<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>)->Tensor<Scalar>) {
    let result = spaceToDepth(tensor, blockSize: blockSize)
    return (result, { v in
        depthToSpace(v, blockSize: blockSize)
    })
}
