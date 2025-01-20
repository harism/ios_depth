import AVFoundation
import SwiftUI

struct ContentView: View {
    var cameraStream : CameraStream
    @State var renderView : RenderView

    init() {
        let renderView = RenderView()
        self.renderView = renderView
        self.cameraStream = CameraStream(renderView)
    }

    var body: some View {
        VStack {
            renderView

            /*
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
            */
        }
        // .padding()
    }
}
