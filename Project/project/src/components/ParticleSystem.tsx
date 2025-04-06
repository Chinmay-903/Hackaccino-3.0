import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Trail } from '@react-three/drei';
import { useAtomStore } from '../store/atomStore';
import * as THREE from 'three';

const PARTICLE_COLORS = {
  proton: '#ff4444',
  neutron: '#4444ff',
  electron: '#44ff44',
};

const PARTICLE_SIZES = {
  proton: 0.5,
  neutron: 0.5,
  electron: 0.2,
};

const SHELL_CONFIGURATION = [2, 8, 8, 18, 18, 32];
const SHELL_RADII = [2, 3.5, 5, 6.5, 8, 9.5];

// Generate random rotation axis
const generateRandomAxis = () => {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.random() * Math.PI;
  return new THREE.Vector3(
    Math.sin(phi) * Math.cos(theta),
    Math.sin(phi) * Math.sin(theta),
    Math.cos(phi)
  ).normalize();
};

const calculateNuclearPosition = (index: number, total: number) => {
  const phi = Math.acos(-1 + (2 * index) / total);
  const theta = Math.sqrt(total * Math.PI) * phi;
  const radius = 0.8;

  return {
    x: radius * Math.cos(theta) * Math.sin(phi),
    y: radius * Math.sin(theta) * Math.sin(phi),
    z: radius * Math.cos(phi),
  };
};

const getElectronShellInfo = (electronIndex: number) => {
  let shellIndex = 0;
  let electronCount = 0;
  
  while (electronIndex >= electronCount + SHELL_CONFIGURATION[shellIndex]) {
    electronCount += SHELL_CONFIGURATION[shellIndex];
    shellIndex++;
  }

  const positionInShell = electronIndex - electronCount;
  const totalInShell = SHELL_CONFIGURATION[shellIndex];
  const shellRadius = SHELL_RADII[shellIndex];

  return { shellIndex, positionInShell, totalInShell, shellRadius };
};

export const ParticleSystem = () => {
  const { protons, neutrons, electrons } = useAtomStore();
  const groupRef = useRef<THREE.Group>();
  const electronRefs = useRef<THREE.Group[]>([]);
  const time = useRef(0);
  
  // Generate random rotation axes for each shell
  const shellRotationAxes = useMemo(() => 
    SHELL_CONFIGURATION.map(() => generateRandomAxis()),
  []);

  const nucleusParticles = useMemo(() => {
    const total = protons.length + neutrons.length;
    return [...protons, ...neutrons].map((particle, index) => ({
      ...particle,
      ...calculateNuclearPosition(index, total),
    }));
  }, [protons, neutrons]);

  useFrame((state, delta) => {
    time.current += delta;

    if (groupRef.current) {
      groupRef.current.rotation.y += 0.001;
    }

    // Animate electrons in their shells with random orbital planes
    electrons.forEach((electron, index) => {
      if (electronRefs.current[index]) {
        const { shellRadius, positionInShell, totalInShell, shellIndex } = getElectronShellInfo(index);
        const baseSpeed = 0.5 / (shellIndex + 1);
        const angle = time.current * baseSpeed + (2 * Math.PI * positionInShell) / totalInShell;

        // Get the rotation axis for this shell
        const rotationAxis = shellRotationAxes[shellIndex];
        
        // Calculate position in the orbital plane
        const position = new THREE.Vector3(
          Math.cos(angle) * shellRadius,
          0,
          Math.sin(angle) * shellRadius
        );

        // Apply rotation to the position vector
        position.applyAxisAngle(rotationAxis, Math.PI / 2);

        electronRefs.current[index].position.copy(position);
        electronRefs.current[index].lookAt(position.clone().multiplyScalar(2));
      }
    });
  });

  // Generate orbital spheres for each shell
  const orbitalSpheres = useMemo(() => {
    const spheres = [];
    let electronCount = 0;

    for (let shellIndex = 0; shellIndex < SHELL_CONFIGURATION.length; shellIndex++) {
      const shellElectrons = Math.min(
        SHELL_CONFIGURATION[shellIndex],
        Math.max(0, electrons.length - electronCount)
      );

      if (shellElectrons > 0) {
        spheres.push({
          radius: SHELL_RADII[shellIndex],
          shellIndex,
          rotationAxis: shellRotationAxes[shellIndex],
        });
      }

      electronCount += SHELL_CONFIGURATION[shellIndex];
      if (electronCount >= electrons.length) break;
    }

    return spheres;
  }, [electrons.length, shellRotationAxes]);

  return (
    <group ref={groupRef}>
      {/* Nucleus */}
      <group>
        {nucleusParticles.map((particle) => (
          <Sphere
            key={particle.id}
            position={[particle.x, particle.y, particle.z]}
            args={[PARTICLE_SIZES[particle.type], 32, 32]}
          >
            <meshStandardMaterial
              color={PARTICLE_COLORS[particle.type]}
              emissive={PARTICLE_COLORS[particle.type]}
              emissiveIntensity={0.5}
              roughness={0.2}
              metalness={0.8}
            />
          </Sphere>
        ))}
      </group>

      {/* Orbital Spheres */}
      {orbitalSpheres.map(({ radius, shellIndex, rotationAxis }) => (
        <group key={`orbital-${shellIndex}`} rotation={[
          Math.acos(rotationAxis.z),
          Math.atan2(rotationAxis.y, rotationAxis.x),
          0
        ]}>
          <mesh>
            <sphereGeometry args={[radius, 32, 32]} />
            <meshBasicMaterial
              color={PARTICLE_COLORS.electron}
              transparent
              opacity={0.03}
              side={THREE.DoubleSide}
              wireframe
            />
          </mesh>
        </group>
      ))}

      {/* Electrons */}
      {electrons.map((electron, index) => (
        <group
          key={electron.id}
          ref={(el) => (electronRefs.current[index] = el)}
          position={[0, 0, 0]}
        >
          <Trail
            width={2}
            length={5}
            color={PARTICLE_COLORS.electron}
            attenuation={(t) => t * t}
          >
            <Sphere args={[PARTICLE_SIZES.electron, 32, 32]}>
              <meshStandardMaterial
                color={PARTICLE_COLORS.electron}
                emissive={PARTICLE_COLORS.electron}
                emissiveIntensity={1}
                transparent
                opacity={0.8}
              />
            </Sphere>
          </Trail>
        </group>
      ))}
    </group>
  );
};