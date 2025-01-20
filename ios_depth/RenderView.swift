import CoreML
import Foundation
import MetalKit
import SwiftUI

fileprivate extension MLMultiArray {
    func copy(to: UnsafeMutableRawPointer, size: Int) {
        to.copyMemory(from: self.dataPointer, byteCount: size)
    }
}

struct RenderView : UIViewRepresentable {
    var coordinator : RenderView.Coordinator?
    var view : MTKView = MTKView()

    init() {
        self.coordinator = RenderView.Coordinator(self)
    }

    func makeUIView(context: Context) -> MTKView {
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
    }

    func makeCoordinator() -> Coordinator {
        return coordinator!
    }

    func setVideoImage(_ image : CVImageBuffer) {
        coordinator!.setVideoImage(image)
    }

    func setDepthImage(_ image : MLMultiArray) {
        coordinator!.setDepthImage(image)
    }

    class Coordinator : NSObject, MTKViewDelegate {
        var parent : RenderView
        var metalDevice : MTLDevice!
        var metalCommandQueue : MTLCommandQueue!

        var metalCopyPipelineState: MTLRenderPipelineState!
        var metalRenderParticlesPipelineState: MTLRenderPipelineState!

        var metalDepthImageTexture: MTLTexture!
        var metalDepthImageTextureData: UnsafeMutableRawPointer!
        var metalVideoImageTexture: MTLTexture!
        var metalVideoImageTextureData: UnsafeMutableRawPointer!
        var metalRenderTexture: MTLTexture!

        var metalSortInitPipelineState: MTLComputePipelineState!
        var metalSortRadixPipelineState: MTLComputePipelineState!
        var metalSortReorderPipelineState: MTLComputePipelineState!
        var metalSortNeighboursPipelineState: MTLComputePipelineState!
        var metalSortReorderOidxPipelineState: MTLComputePipelineState!
        var metalDensityPipelineState: MTLComputePipelineState!
        var metalForcesPipelineState: MTLComputePipelineState!
        var metalEulerPipelineState: MTLComputePipelineState!

        var metalSortBuffer: MTLBuffer!
        var metalSortBufferTmp: MTLBuffer!
        var metalParticleBuffer: MTLBuffer!
        var metalParticleBufferTmp: MTLBuffer!
        var metalNeighboursBuffer: MTLBuffer!
        var metalVarsBuffer: MTLBuffer!

        var renderTimeStep: Float = 0.04

        init(_ parent : RenderView) {
            self.parent = parent

            if let metalDevice = MTLCreateSystemDefaultDevice() {
                self.metalDevice = metalDevice
            }

            self.metalCommandQueue = metalDevice.makeCommandQueue()
            let metalLibrary = metalDevice.makeDefaultLibrary();

            do {
                let copyDescriptor = MTLRenderPipelineDescriptor()
                copyDescriptor.vertexFunction = metalLibrary?.makeFunction(name: "copyVertexShader")
                copyDescriptor.fragmentFunction = metalLibrary?.makeFunction(name: "copyFragmentShader")
                copyDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
                self.metalCopyPipelineState = try metalDevice.makeRenderPipelineState(descriptor: copyDescriptor)

               let renderParticlesDescriptor = MTLRenderPipelineDescriptor()
                renderParticlesDescriptor.vertexFunction = metalLibrary?.makeFunction(name: "renderParticleVertexShader")
                renderParticlesDescriptor.fragmentFunction = metalLibrary?.makeFunction(name: "renderParticleFragmentShader")
                renderParticlesDescriptor.depthAttachmentPixelFormat = .invalid // .depth32Float
                renderParticlesDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
                renderParticlesDescriptor.colorAttachments[0].isBlendingEnabled = true
                renderParticlesDescriptor.colorAttachments[0].alphaBlendOperation = .max
                renderParticlesDescriptor.colorAttachments[0].rgbBlendOperation = .max
                renderParticlesDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
                renderParticlesDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
                renderParticlesDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
                renderParticlesDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
                self.metalRenderParticlesPipelineState = try metalDevice.makeRenderPipelineState(descriptor: renderParticlesDescriptor)

                let depthImageDescriptor = MTLTextureDescriptor()
                depthImageDescriptor.width = 512
                depthImageDescriptor.height = 512
                depthImageDescriptor.depth = 1
                depthImageDescriptor.pixelFormat = .r16Float
                depthImageDescriptor.storageMode = .shared
                depthImageDescriptor.textureType = .type2D
                depthImageDescriptor.usage = .shaderRead
                self.metalDepthImageTexture = metalDevice.makeTexture(descriptor: depthImageDescriptor)
                self.metalDepthImageTextureData = UnsafeMutableRawPointer.allocate(byteCount: 512 * 512 * MemoryLayout<Float16>.stride, alignment: 1)

                let videoImageTextureDescriptor = MTLTextureDescriptor()
                videoImageTextureDescriptor.width = 1920
                videoImageTextureDescriptor.height = 1080
                videoImageTextureDescriptor.depth = 1
                videoImageTextureDescriptor.pixelFormat = .bgra8Unorm
                videoImageTextureDescriptor.storageMode = .shared
                videoImageTextureDescriptor.textureType = .type2D
                videoImageTextureDescriptor.usage = .shaderRead
                self.metalVideoImageTexture = metalDevice.makeTexture(descriptor: videoImageTextureDescriptor)
                self.metalVideoImageTextureData = UnsafeMutableRawPointer.allocate(byteCount: 1920 * 1080 * MemoryLayout<UInt32>.stride, alignment: 1)

                let renderTextureDescriptor = MTLTextureDescriptor()
                renderTextureDescriptor.width = 1080
                renderTextureDescriptor.height = 1920
                renderTextureDescriptor.depth = 1
                renderTextureDescriptor.pixelFormat = .rgba16Float
                renderTextureDescriptor.usage = [.shaderRead, .renderTarget]
                renderTextureDescriptor.storageMode = .private
                renderTextureDescriptor.mipmapLevelCount = 1
                self.metalRenderTexture = metalDevice.makeTexture(descriptor: renderTextureDescriptor)
            } catch let error {
                print(error.localizedDescription)
            }

            do {
                self.metalSortInitPipelineState = try metalDevice.makeComputePipelineState(function: (metalLibrary?.makeFunction(name: "sortInit"))!)
                self.metalSortRadixPipelineState = try metalDevice.makeComputePipelineState(function: (metalLibrary?.makeFunction(name: "sortRadix"))!)
                self.metalSortReorderPipelineState = try metalDevice.makeComputePipelineState(function: (metalLibrary?.makeFunction(name: "sortReorder"))!)
                self.metalSortNeighboursPipelineState = try metalDevice.makeComputePipelineState(function: (metalLibrary?.makeFunction(name: "sortNeighbours"))!)
                self.metalSortReorderOidxPipelineState = try metalDevice.makeComputePipelineState(function: (metalLibrary?.makeFunction(name: "sortReorderOidx"))!)
                self.metalDensityPipelineState = try metalDevice.makeComputePipelineState(function: (metalLibrary?.makeFunction(name: "density"))!)
                self.metalForcesPipelineState = try metalDevice.makeComputePipelineState(function: (metalLibrary?.makeFunction(name: "forces"))!)
                self.metalEulerPipelineState = try metalDevice.makeComputePipelineState(function: (metalLibrary?.makeFunction(name: "euler"))!)
            } catch let error {
                print(error.localizedDescription)
            }

            let particles = UnsafeMutablePointer<Particle>.allocate(capacity: Int(VAR_count))
            for index in 0..<Int(VAR_count) {
                let x = Float((index >> 1) & 0x01FF) / 512.0
                let y = Float((index >> 1) / 0x01FF) / 512.0
                let pos = simd_make_float3(Float.random(in: -0.99...0.99), Float.random(in: -1.99...1.99), Float.random(in: -0.99...0.99))
                let vel = simd_make_float3(0.0, 0.0, 0.0)
                particles[index] = Particle(
                    oidx: Int32(index),
                    pos: pos,
                    vel: vel,
                    force: simd_make_float3(0.0, 0.0, 0.0),
                    density: 0.0,
                    pressure: 0.0,
                    color: simd_make_float4(-1.0))
            }
            
            self.metalParticleBuffer = metalDevice.makeBuffer(bytes: particles, length: Int(VAR_count) * MemoryLayout<Particle>.stride)
            self.metalParticleBufferTmp = metalDevice.makeBuffer(length: Int(VAR_count) * MemoryLayout<Particle>.stride,
                                                                 options: MTLResourceOptions.storageModePrivate)
            self.metalNeighboursBuffer = metalDevice.makeBuffer(length: 9 * Int(VAR_count) * MemoryLayout<simd_int2>.stride, options: .storageModePrivate)
            self.metalSortBuffer = metalDevice.makeBuffer(length: Int(VAR_count) * MemoryLayout<KeyIndex>.stride,
                                                          options: MTLResourceOptions.storageModePrivate)
            self.metalSortBufferTmp = metalDevice.makeBuffer(length: Int(VAR_count) * MemoryLayout<KeyIndex>.stride,
                                                             options: MTLResourceOptions.storageModePrivate)
            particles.deallocate()

            var vars = Vars()
            vars.simWidth = 4.0
            vars.d = 0.01
            vars.h = 4.0 * vars.d
            vars.m = 0.15
            vars.p0 = 998.29
            vars.gasConstant = 1.3145
            vars.viscosityConstant = 3.252
            vars.poly6 = Float(315.0 / (64.0 * .pi * pow(Double(vars.h), 9.0)))
            vars.poly6Lap = Float(315.0 / (64.0 * .pi * pow(Double(vars.h), 9.0)))
            vars.poly6Grad = Float(315.0 / (64.0 * .pi * pow(Double(vars.h), 9.0)))
            vars.spikyLap = Float(-45.0 / (.pi * pow(Double(vars.h), 6.0)))
            vars.viscGrad = Float(45.0 / (.pi * pow(Double(vars.h), 6.0)))
            vars.maxKey = Int32(ceil(min(1023.0, (vars.simWidth / vars.h) - 1.0)))
            self.metalVarsBuffer = metalDevice.makeBuffer(bytes: &vars, length: MemoryLayout<Vars>.stride)

            super.init()
        }

        func setVideoImage(_ image : CVImageBuffer) {
            CVPixelBufferLockBaseAddress(image, .readOnly)
            let imageAddress = CVPixelBufferGetBaseAddress(image)
            memcpy(metalVideoImageTextureData, imageAddress, 1920 * 1080 * MemoryLayout<UInt32>.stride)
            CVPixelBufferUnlockBaseAddress(image, .readOnly)
        }

        func setDepthImage(_ image : MLMultiArray) {
            image.copy(to: metalDepthImageTextureData, size: 512 * 512 * MemoryLayout<Float16>.stride)
        }

        func mtkView(_ view : MTKView, drawableSizeWillChange size : CGSize) {
        }

        func draw(in view : MTKView) {
            guard let drawable = view.currentDrawable else {
                return
            }
            let commandBuffer = metalCommandQueue.makeCommandBuffer()
            var renderCount = Int(VAR_count)

            // Sort
            if true {
                 let sortInitCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
                sortInitCommandEncoder!.setComputePipelineState(metalSortInitPipelineState)
                sortInitCommandEncoder!.setBuffer(metalVarsBuffer, offset: 0, index: 0)
                sortInitCommandEncoder!.setBuffer(metalParticleBuffer, offset: 0, index: 1)
                sortInitCommandEncoder!.setBuffer(metalSortBuffer, offset: 0, index: 2)
                sortInitCommandEncoder!.dispatchThreads(MTLSizeMake(renderCount, 1, 1), threadsPerThreadgroup:
                                                        MTLSizeMake(min(renderCount, metalSortInitPipelineState.maxTotalThreadsPerThreadgroup), 1, 1))
                sortInitCommandEncoder!.endEncoding()

                let sortCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
                sortCommandEncoder!.setComputePipelineState(metalSortRadixPipelineState)
                sortCommandEncoder!.setBuffer(metalVarsBuffer, offset: 0, index: 0)
                sortCommandEncoder!.setBuffer(metalSortBuffer, offset: 0, index: 1)
                sortCommandEncoder!.setBuffer(metalSortBufferTmp, offset: 0, index: 2)
                sortCommandEncoder!.setBytes(&renderCount, length: MemoryLayout<Int>.stride, index: 3)
                sortCommandEncoder!.setThreadgroupMemoryLength(4 * 4 * metalSortRadixPipelineState.maxTotalThreadsPerThreadgroup, index: 0)
                sortCommandEncoder!.dispatchThreadgroups(MTLSizeMake(1, 1, 1), threadsPerThreadgroup:
                            MTLSizeMake(metalSortRadixPipelineState.maxTotalThreadsPerThreadgroup, 1, 1))
                sortCommandEncoder!.endEncoding()

                let sortReorderCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
                sortReorderCommandEncoder!.setComputePipelineState(metalSortReorderPipelineState)
                sortReorderCommandEncoder!.setBuffer(metalVarsBuffer, offset: 0, index: 0)
                sortReorderCommandEncoder!.setBuffer(metalSortBuffer, offset: 0, index: 1)
                sortReorderCommandEncoder!.setBuffer(metalParticleBuffer, offset: 0, index: 2)
                sortReorderCommandEncoder!.setBuffer(metalParticleBufferTmp, offset: 0, index: 3)
                sortReorderCommandEncoder!.dispatchThreads(MTLSizeMake(renderCount, 1, 1), threadsPerThreadgroup: MTLSizeMake(min(renderCount, metalSortReorderPipelineState.maxTotalThreadsPerThreadgroup), 1, 1))
                sortReorderCommandEncoder!.endEncoding()

                let sortNeighboursCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
                sortNeighboursCommandEncoder!.setComputePipelineState(metalSortNeighboursPipelineState)
                sortNeighboursCommandEncoder!.setBuffer(metalVarsBuffer, offset: 0, index: 0)
                sortNeighboursCommandEncoder!.setBuffer(metalSortBuffer, offset: 0, index: 1)
                sortNeighboursCommandEncoder!.setBuffer(metalParticleBufferTmp, offset: 0, index: 2)
                sortNeighboursCommandEncoder!.setBuffer(metalNeighboursBuffer, offset: 0, index: 3)
                sortNeighboursCommandEncoder!.setBytes(&renderCount, length: MemoryLayout<Int>.stride, index: 4)
                sortNeighboursCommandEncoder!.dispatchThreadgroups(MTLSizeMake(3, 3, 1), threadsPerThreadgroup: MTLSizeMake(metalSortNeighboursPipelineState.maxTotalThreadsPerThreadgroup, 1, 1))
                sortNeighboursCommandEncoder!.endEncoding()

                let particleBufferBlitCommandEncoder = commandBuffer!.makeBlitCommandEncoder()
                particleBufferBlitCommandEncoder!.copy(from: metalParticleBufferTmp, sourceOffset: 0, to: metalParticleBuffer, destinationOffset: 0, size: renderCount * MemoryLayout<Particle>.stride)
                particleBufferBlitCommandEncoder!.endEncoding()

                let sortReorderOidxCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
                sortReorderOidxCommandEncoder!.setComputePipelineState(metalSortReorderOidxPipelineState)
                sortReorderOidxCommandEncoder!.setBuffer(metalVarsBuffer, offset: 0, index: 0)
                sortReorderOidxCommandEncoder!.setBuffer(metalParticleBuffer, offset: 0, index: 1)
                sortReorderOidxCommandEncoder!.setBuffer(metalParticleBufferTmp, offset: 0, index: 2)
                sortReorderOidxCommandEncoder!.dispatchThreads(MTLSizeMake(renderCount, 1, 1), threadsPerThreadgroup: MTLSizeMake(min(renderCount, metalSortReorderPipelineState.maxTotalThreadsPerThreadgroup), 1, 1))
                sortReorderOidxCommandEncoder!.endEncoding()
            }

            // Density
            if true {
                let densityCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
                densityCommandEncoder!.setComputePipelineState(metalDensityPipelineState)
                densityCommandEncoder!.setBuffer(metalVarsBuffer, offset: 0, index: 0)
                densityCommandEncoder!.setBuffer(metalNeighboursBuffer, offset: 0, index: 1)
                densityCommandEncoder!.setBuffer(metalParticleBuffer, offset: 0, index: 2)
                densityCommandEncoder!.dispatchThreads(MTLSizeMake(renderCount, 1, 1), threadsPerThreadgroup:
                                       MTLSizeMake(min(renderCount, metalDensityPipelineState.maxTotalThreadsPerThreadgroup), 1, 1))
                densityCommandEncoder!.endEncoding()
            }

            // Forces
            if true {
                let forcesCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
                forcesCommandEncoder!.setComputePipelineState(metalForcesPipelineState)
                forcesCommandEncoder!.setBuffer(metalVarsBuffer, offset: 0, index: 0)
                forcesCommandEncoder!.setBuffer(metalNeighboursBuffer, offset: 0, index: 1)
                forcesCommandEncoder!.setBuffer(metalParticleBuffer, offset: 0, index: 2)
                forcesCommandEncoder!.dispatchThreads(MTLSizeMake(renderCount, 1, 1), threadsPerThreadgroup:
                                                        MTLSizeMake(min(renderCount, metalForcesPipelineState.maxTotalThreadsPerThreadgroup), 1, 1))
                forcesCommandEncoder!.endEncoding()
            }

            // Integrate
            if true {
                //var g = gravity
                let eulerCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
                eulerCommandEncoder!.setComputePipelineState(metalEulerPipelineState)
                eulerCommandEncoder!.setBuffer(metalVarsBuffer, offset: 0, index: 0)
                eulerCommandEncoder!.setBuffer(metalParticleBuffer, offset: 0, index: 1)
                eulerCommandEncoder!.setBytes(&renderTimeStep, length: MemoryLayout<Float>.stride, index: 2)
                //eulerCommandEncoder!.setBytes(&g, length: MemoryLayout<simd_float4>.stride, index: 3)
                //eulerCommandEncoder!.setBytes(&rmsIter, length: MemoryLayout<Float>.stride, index: 4)
                //eulerCommandEncoder!.setBytes(&fft, length: MemoryLayout<Float>.stride * 512, index: 4)
                eulerCommandEncoder!.setTexture(metalVideoImageTexture, index: 0)
                eulerCommandEncoder!.setTexture(metalDepthImageTexture, index: 1)
                eulerCommandEncoder!.dispatchThreads(MTLSizeMake(renderCount, 1, 1), threadsPerThreadgroup:
                                                        MTLSizeMake(min(renderCount, metalEulerPipelineState.maxTotalThreadsPerThreadgroup), 1, 1))
                eulerCommandEncoder!.endEncoding()
            }





            metalDepthImageTexture.replace(region: MTLRegionMake2D(0, 0, 512, 512), mipmapLevel: 0, withBytes: metalDepthImageTextureData, bytesPerRow: 512 *  MemoryLayout<Float16>.stride)
            metalVideoImageTexture.replace(region: MTLRegionMake2D(0, 0, 1920, 1080), mipmapLevel: 0, withBytes: metalVideoImageTextureData, bytesPerRow: 1920 * MemoryLayout<UInt32>.stride)


            let particlesRenderPassDescriptor = MTLRenderPassDescriptor()
            particlesRenderPassDescriptor.depthAttachment.clearDepth = 1.0
            particlesRenderPassDescriptor.depthAttachment.loadAction = .dontCare
            particlesRenderPassDescriptor.depthAttachment.storeAction = .dontCare
            particlesRenderPassDescriptor.colorAttachments[0].texture = metalRenderTexture
            particlesRenderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.23, 0.28, 0.4, 0.0)
            particlesRenderPassDescriptor.colorAttachments[0].loadAction =  .clear
            particlesRenderPassDescriptor.colorAttachments[0].storeAction = .store

            let renderParticlesCommandEncoder = commandBuffer!.makeRenderCommandEncoder(descriptor: particlesRenderPassDescriptor)
            renderParticlesCommandEncoder!.setRenderPipelineState(metalRenderParticlesPipelineState)
            renderParticlesCommandEncoder!.setCullMode(.none)
            renderParticlesCommandEncoder!.setVertexBuffer(metalVarsBuffer, offset: 0, index: 0)
            renderParticlesCommandEncoder!.setVertexBuffer(metalParticleBufferTmp, offset: 0, index: 1)
            //renderParticlesCommandEncoder!.setVertexBytes(&renderCount, length: MemoryLayout<Int>.stride, index: 2)
            //renderParticlesCommandEncoder!.setVertexBytes(&projMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 3)
            //renderParticlesCommandEncoder!.setVertexBytes(&viewModelMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 4)
            //renderParticlesCommandEncoder!.setVertexTexture(metalRenderTexture, index: 0)
            //renderParticlesCommandEncoder!.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: renderCount / 2 - 1)
            //renderParticlesCommandEncoder!.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 2, instanceCount: 262144) // renderCount / 2)
            renderParticlesCommandEncoder!.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: renderCount)
            renderParticlesCommandEncoder!.endEncoding()


            let copyPassDescriptor = view.currentRenderPassDescriptor
            copyPassDescriptor!.colorAttachments[0].loadAction = .dontCare
            copyPassDescriptor!.colorAttachments[0].storeAction = .store

            let copyCommandEncoder = commandBuffer!.makeRenderCommandEncoder(descriptor: copyPassDescriptor!)
            copyCommandEncoder!.setRenderPipelineState(metalCopyPipelineState)
            copyCommandEncoder!.setFragmentTexture(metalRenderTexture, index: 0)
            copyCommandEncoder!.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            copyCommandEncoder!.endEncoding()

            commandBuffer!.present(drawable)
            commandBuffer!.commit()

        }
    }
}
