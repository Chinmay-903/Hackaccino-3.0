import { create } from 'zustand';

interface Particle {
  id: string;
  type: 'proton' | 'neutron' | 'electron';
  position: { x: number; y: number; z: number };
}

interface AtomState {
  protons: Particle[];
  neutrons: Particle[];
  electrons: Particle[];
  selectedElement: number | null;
  addParticle: (type: Particle['type']) => void;
  removeParticle: (id: string, type: Particle['type']) => void;
  updateParticlePosition: (id: string, type: Particle['type'], position: { x: number; y: number; z: number }) => void;
  setAtomConfiguration: (protons: number, neutrons?: number, electrons?: number) => void;
}

export const useAtomStore = create<AtomState>((set) => ({
  protons: [],
  neutrons: [],
  electrons: [],
  selectedElement: null,
  
  addParticle: (type) => set((state) => {
    const newParticle: Particle = {
      id: Math.random().toString(36).substr(2, 9),
      type,
      position: { x: 0, y: 0, z: 0 },
    };
    
    return {
      [type + 's']: [...state[type + 's'], newParticle],
    };
  }),
  
  removeParticle: (id, type) => set((state) => ({
    [type + 's']: state[type + 's'].filter((p: Particle) => p.id !== id),
  })),
  
  updateParticlePosition: (id, type, position) => set((state) => ({
    [type + 's']: state[type + 's'].map((p: Particle) =>
      p.id === id ? { ...p, position } : p
    ),
  })),

  setAtomConfiguration: (protons, neutrons = protons, electrons = protons) => set(() => {
    const createParticles = (count: number, type: Particle['type']) => 
      Array.from({ length: count }, () => ({
        id: Math.random().toString(36).substr(2, 9),
        type,
        position: { x: 0, y: 0, z: 0 },
      }));

    return {
      protons: createParticles(protons, 'proton'),
      neutrons: createParticles(neutrons, 'neutron'),
      electrons: createParticles(electrons, 'electron'),
      selectedElement: protons,
    };
  }),
}));