import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { Bounds, OrbitControls, useGLTF, Environment } from "@react-three/drei";

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return <primitive object={scene} />;
}

function ModelPane({ url, label }: { url: string; label: string }) {
  return (
    <div className="flex-1 min-w-0">
      <p className="text-xs text-ink-400 text-center mb-1 truncate">{label}</p>
      <div className="w-full aspect-square bg-ink-900 rounded-lg overflow-hidden">
        <Canvas camera={{ position: [3, 3, 3], fov: 45 }}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[5, 5, 5]} intensity={1.2} />
          <Suspense fallback={null}>
            {/* Bounds auto-frames the camera to the model's real bounding
                box — models range from a few mm to hundreds of mm and a
                fixed camera distance either clips or looks empty. */}
            <Bounds fit clip observe margin={1.3}>
              <Model url={url} />
            </Bounds>
          </Suspense>
          <OrbitControls makeDefault />
          <Environment preset="city" />
        </Canvas>
      </div>
    </div>
  );
}

interface Props {
  queryGltfUrl?: string;
  resultGltfUrl?: string;
  resultName: string;
}

export default function Viewer3D({ queryGltfUrl, resultGltfUrl, resultName }: Props) {
  if (!queryGltfUrl && !resultGltfUrl) return null;

  return (
    <div className="flex gap-3">
      {queryGltfUrl && <ModelPane url={queryGltfUrl} label="Your upload" />}
      {resultGltfUrl && <ModelPane url={resultGltfUrl} label={resultName} />}
    </div>
  );
}
