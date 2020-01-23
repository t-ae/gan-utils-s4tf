import TensorFlow

public enum GANLossType: String, Codable {
    case nonSaturating, lsgan, hinge
}

public struct GANLoss {
    public var type: GANLossType
    
    public init(_ type: GANLossType) {
        self.type = type
    }
    
    @differentiable
    public func lossG(_ tensor: Tensor<Float>) -> Tensor<Float> {
        switch type {
        case .nonSaturating:
            return softplus(-tensor).mean()
        case .lsgan:
            return pow(tensor - 1, 2).mean()
        case .hinge:
            return -tensor.mean()
        }
    }
    
    @differentiable
    public func lossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        switch type {
        case .nonSaturating:
            return softplus(-real).mean() + softplus(fake).mean()
        case .lsgan:
            return pow(real-1, 2).mean() + pow(fake, 2).mean()
        case .hinge:
            return relu(1 - real).mean() + relu(1 + fake).mean()
        }
    }
}

public enum ReconstructionLossType: String, Codable {
    case meanSquaredError
}

public struct ReconstructionLoss {
    public let type: ReconstructionLossType
    
    public init(_ type: ReconstructionLossType) {
        self.type = type
    }
    
    @differentiable(wrt: fake)
    public func callAsFunction(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        switch type {
        case .meanSquaredError:
            return meanSquaredError(predicted: fake, expected: real)
        }
    }
}
