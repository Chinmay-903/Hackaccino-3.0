import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { ParticleSystem } from './components/ParticleSystem';
import { ParticleBucket } from './components/ParticleBucket';
import { AtomicInfo } from './components/AtomicInfo';
import { PeriodicTable } from './components/PeriodicTable';

function App() {
  return (
    <div className="w-full h-screen bg-gray-950">
      <Canvas
        camera={{ position: [0, 0, 15], fov: 75 }}
        className="w-full h-full"
      >
        <color attach="background" args={['#050510']} />
        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <Stars
          radius={100}
          depth={50}
          count={5000}
          factor={4}
          saturation={0}
          fade
          speed={1}
        />
        <ParticleSystem />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={true}
          autoRotateSpeed={0.5}
        />
      </Canvas>
      
      <AtomicInfo />
      <ParticleBucket />
      <PeriodicTable />
    </div>
  );
}

export default App;