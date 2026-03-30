import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Environment, Text } from '@react-three/drei';
import * as THREE from 'three';

function SkinSurface({ imageUrl }) {
  const meshRef = useRef();
  const texture = useMemo(() => {
    if (!imageUrl) return null;
    const loader = new THREE.TextureLoader();
    const tex = loader.load(imageUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }, [imageUrl]);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.15;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.08;
    }
  });

  // Create a slightly curved surface to simulate skin curvature
  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(3, 3, 64, 64);
    const positions = geo.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      // Slight dome shape
      const dist = Math.sqrt(x * x + y * y);
      const z = Math.cos(dist * 0.6) * 0.3;
      positions.setZ(i, z);
    }
    geo.computeVertexNormals();
    return geo;
  }, []);

  return (
    <mesh ref={meshRef} geometry={geometry}>
      {texture ? (
        <meshStandardMaterial
          map={texture}
          roughness={0.7}
          metalness={0.05}
          side={THREE.DoubleSide}
        />
      ) : (
        <meshStandardMaterial
          color="#e8b4a0"
          roughness={0.8}
          metalness={0.02}
          side={THREE.DoubleSide}
        />
      )}
    </mesh>
  );
}

function GridFloor() {
  return (
    <gridHelper
      args={[10, 20, '#1e3a5f', '#0f1f33']}
      position={[0, -1.8, 0]}
      rotation={[0, 0, 0]}
    />
  );
}

function AxisLabels() {
  return (
    <group>
      <Text position={[2.2, -1.7, 0]} fontSize={0.15} color="#38bdf8" anchorX="center">
        X
      </Text>
      <Text position={[0, 0.3, 0]} fontSize={0.15} color="#38bdf8" anchorX="center">
        Y
      </Text>
      <Text position={[0, -1.7, 2.2]} fontSize={0.15} color="#38bdf8" anchorX="center">
        Z
      </Text>
    </group>
  );
}

export default function ThreeDViewer({ imageUrl }) {
  return (
    <div className="w-full aspect-square rounded-2xl overflow-hidden bg-gray-900/50 border border-white/10">
      <Canvas
        camera={{ position: [0, 1, 4], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={['#0a0f1a']} />
        <fog attach="fog" args={['#0a0f1a', 5, 15]} />

        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight position={[3, 5, 2]} intensity={1.2} color="#ffffff" />
        <directionalLight position={[-2, 3, -1]} intensity={0.4} color="#38bdf8" />
        <pointLight position={[0, 2, 3]} intensity={0.6} color="#f0f0ff" />

        {/* Scene */}
        <SkinSurface imageUrl={imageUrl} />
        <GridFloor />
        <AxisLabels />

        {/* Controls */}
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          minDistance={2}
          maxDistance={8}
          autoRotate={!imageUrl}
          autoRotateSpeed={1}
        />
      </Canvas>
    </div>
  );
}
