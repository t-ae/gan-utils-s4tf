import TensorFlow

public struct ResizeLayer: ParameterlessLayer {
    public enum Method {
        case nearestNeighbor, bilinear, bicubic
    }
    
    @noDerivative
    public var method: Method
    
    @noDerivative
    public var width: Int
    @noDerivative
    public var height: Int
    
    @noDerivative
    public var alignCorners: Bool
    @noDerivative
    public var halfPixelCenters: Bool
    
    public init(
        _ method: Method,
        width: Int,
        height: Int,
        alignCorners: Bool = false,
        halfPixelCenters: Bool = false
    ) {
        self.method = method
        self.width = width
        self.height = height
        self.alignCorners = alignCorners
        self.halfPixelCenters = halfPixelCenters
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        switch method {
        case .nearestNeighbor:
            return resizeNearestNeighbor(
                images: input,
                width: width,
                height: height,
                alignCorners: alignCorners,
                halfPixelCenters: halfPixelCenters
            )
        case .bilinear:
            return resizeBilinear(
                images: input,
                width: width,
                height: height,
                alignCorners: alignCorners,
                halfPixelCenters: halfPixelCenters
            )
        case .bicubic:
            return resizeBicubic(
                images: input,
                width: width,
                height: height,
                alignCorners: alignCorners,
                halfPixelCenters: halfPixelCenters
            )
        }
    }
}

@differentiable(wrt: images, vjp: vjpResizeNearestNeighbor)
public func resizeNearestNeighbor(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<Float> {
    _Raw.resizeNearestNeighbor(
        images: images,
        size: Tensor([Int32(height), Int32(width)]),
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
}

//@derivative(of: resizeNN)
@usableFromInline
func vjpResizeNearestNeighbor(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool,
    halfPixelCenters: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resizeNearestNeighbor(
        images: images,
        width: width,
        height: height,
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
    return (resized, { v in
        _Raw.resizeNearestNeighborGrad(
            grads: v,
            size: Tensor([Int32(height), Int32(width)]),
            alignCorners: alignCorners,
            halfPixelCenters: halfPixelCenters
        )
    })
}


@differentiable(wrt: images, vjp: vjpResizeBilinear)
public func resizeBilinear(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<Float> {
    _Raw.resizeBilinear(
        images: images,
        size: Tensor([Int32(height), Int32(width)]),
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
}

//@derivative(of: resizeBL)
@usableFromInline
func vjpResizeBilinear(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool,
    halfPixelCenters: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resizeBilinear(
        images: images,
        width: width,
        height: height,
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
    return (resized, { v in
        _Raw.resizeBilinearGrad(
            grads: v,
            originalImage: images,
            alignCorners: alignCorners,
            halfPixelCenters: halfPixelCenters
        )
    })
}

@differentiable(wrt: images, vjp: vjpResizeBicubic)
public func resizeBicubic(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool = false,
    halfPixelCenters: Bool = false
) -> Tensor<Float> {
    _Raw.resizeBicubic(
        images: images,
        size: Tensor([Int32(height), Int32(width)]),
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
}

@usableFromInline
func vjpResizeBicubic(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool,
    halfPixelCenters: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resizeBicubic(
        images: images,
        width: width,
        height: height,
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
    )
    return (resized, { v in
        _Raw.resizeBicubicGrad(
            grads: v,
            originalImage: images,
            alignCorners: alignCorners,
            halfPixelCenters: halfPixelCenters
        )
    })
}
