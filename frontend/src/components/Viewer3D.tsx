import { Suspense, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Center, Environment } from "@react-three/drei";

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return (
    <Center>
      <primitive object={scene} />
    </Center>
  );
}

function ModelPane({ url, label }: { url: string; label: string }) {
  return (
    <div className="flex-1 min-w-0">
      <p className="text-xs text-gray-500 text-center mb-1">{label}</p>
      <div className="w-full aspect-square bg-gray-900 rounded-lg overflow-hidden">
        <Canvas camera={{ position: [0, 0, 3], fov: 50 }}>
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 5, 5]} intensity={1.2} />
          <Suspense fallback={null}>
            <Model url={url} />
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
      {queryGltfUrl && <ModelPane url={queryGltfUrl} label="Query" />}
      {resultGltfUrl && <ModelPane url={resultGltfUrl} label={resultName} />}
    </div>
  );
}
