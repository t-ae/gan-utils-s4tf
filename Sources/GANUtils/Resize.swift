import TensorFlow

public struct Resize: ParameterlessLayer {
    public enum Method: String, Codable {
        case nearestNeighbor, bilinear, bicubic
    }
    public enum OutputSize {
        case constant(width: Int, height: Int)
        case factor(x: Int, y: Int)
    }
    
    @noDerivative
    public var method: Method
    
    @noDerivative
    public var outputSize: OutputSize
    
    @noDerivative
    public var alignCorners: Bool
    @noDerivative
    public var halfPixelCenters: Bool
    
    public init(_ method: Method,
                outputSize: OutputSize,
                alignCorners: Bool = false,
                halfPixelCenters: Bool = false) {
        self.method = method
        self.outputSize = outputSize
        self.alignCorners = alignCorners
        self.halfPixelCenters = halfPixelCenters
    }
    
    public init(
        _ method: Method,
        width: Int,
        height: Int,
        alignCorners: Bool = false,
        halfPixelCenters: Bool = false
    ) {
        self.init(method, outputSize: .constant(width: width, height: height),
                  alignCorners: alignCorners, halfPixelCenters: halfPixelCenters)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        assert(input.rank == 4, "Input must be rank 4 image tensor.")
        
        let (width, height): (Int, Int)
        switch outputSize {
        case let .constant(width: w, height: h):
            (width, height) = (w, h)
        case let .factor(x: x, y: y):
            width = input.shape[2] * x
            height = input.shape[1] * y
        }
        
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

public func resizeArea<Scalar: Numeric>(
    images: Tensor<Scalar>,
    width: Int,
    height: Int,
    alignCorners: Bool = false
) -> Tensor<Float> {
    _Raw.resizeArea(images: images,
                    size: Tensor([Int32(height), Int32(width)]),
                    alignCorners: alignCorners)
}

@differentiable(wrt: images)
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

@inlinable
@derivative(of: resizeNearestNeighbor)
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
            size: Tensor([Int32(images.shape[1]), Int32(images.shape[2])]),
            alignCorners: alignCorners,
            halfPixelCenters: halfPixelCenters
        )
    })
}


@differentiable(wrt: images)
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

@inlinable
@derivative(of: resizeBilinear)
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

@differentiable(wrt: images)
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

@inlinable
@derivative(of: resizeBicubic)
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
