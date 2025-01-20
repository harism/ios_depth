import AVFoundation
import CoreML
import SwiftUI

fileprivate extension CIImage {
    var image: CGImage? {
        let ciContext = CIContext()
        guard let cgImage = ciContext.createCGImage(self, from: self.extent) else { return nil }
        return cgImage
        //return Image(decorative: cgImage, scale: 1, orientation: .up)
    }

    /// Returns a resized image.
    func resized(to size: CGSize) -> CIImage {
        let outputScaleX = size.width / extent.width
        let outputScaleY = size.height / extent.height
        var outputImage = self.transformed(by: CGAffineTransform(scaleX: outputScaleX, y: outputScaleY))
        outputImage = outputImage.transformed(
            by: CGAffineTransform(translationX: -outputImage.extent.origin.x, y: -outputImage.extent.origin.y)
        )
        return outputImage
    }
}

final class CameraStream : ObservableObject {
    let camera = Camera()
    let renderView : RenderView
    var depthAnything : DepthAnything

    init(_ renderView : RenderView) {
        self.renderView = renderView

        do {
            let mlModelConfiguration = MLModelConfiguration();
            mlModelConfiguration.computeUnits = .cpuAndNeuralEngine
            depthAnything = try DepthAnything(configuration: mlModelConfiguration)
        } catch let error {
            fatalError(error.localizedDescription)
        }

        Task {
            await handleCameraPreviews()
        }
        Task {
            await camera.start()
        }
    }

    func handleCameraPreviews() async {
        let imageStream = camera.previewStream
            .map { $0.image }
        for await image in imageStream {
            if image != nil {
                let bitsPerComponent = image!.bitsPerComponent
                let bytesPerRow = image!.bytesPerRow
                guard let colorSpace = image!.colorSpace else { continue }
                let bitmapInfo = image!.bitmapInfo

                guard let context = CGContext(data: nil, width: Int(1920), height: Int(1080), bitsPerComponent: 8, bytesPerRow: 4 * 1920, space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else { continue }
                context.interpolationQuality = .high
                context.draw(image!, in: CGRect(origin: CGPoint.zero, size: CGSize(width: 1920, height: 1080)))

                let resizedImage = context.makeImage()
                var imBuffer : CVImageBuffer? = nil
                let options : NSDictionary = [:]
                let dataFromImageDataProvider = CFDataCreateMutableCopy(kCFAllocatorDefault, 0, resizedImage!.dataProvider!.data)

                CVPixelBufferCreateWithBytes(
                    kCFAllocatorDefault,
                    1920,
                    1080,
                    kCVPixelFormatType_32BGRA,
                    CFDataGetMutableBytePtr(dataFromImageDataProvider),
                    4 * 1920,
                    nil,
                    nil,
                    options,
                    &imBuffer)

                do {
                    let depthOutput = try depthAnything.prediction(fromInput_1: imBuffer!)
                    await renderView.setDepthImage(depthOutput.var_1258)
                    await renderView.setVideoImage(imBuffer!)
                } catch let error {
                    print(error.localizedDescription)
                }
            }
        }
    }

}
