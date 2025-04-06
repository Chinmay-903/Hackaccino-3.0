import { motion } from 'framer-motion';
import { Atom, BellElectric as Electron, Disc } from 'lucide-react';
import { useAtomStore } from '../store/atomStore';

export const ParticleBucket = () => {
  const addParticle = useAtomStore((state) => state.addParticle);

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gray-900 bg-opacity-80 p-4">
      <div className="max-w-4xl mx-auto flex justify-around items-center">
        <motion.div
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          className="flex flex-col items-center cursor-pointer"
          onClick={() => addParticle('proton')}
        >
          <Atom className="w-12 h-12 text-red-500" />
          <span className="text-red-500 mt-2">Proton</span>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          className="flex flex-col items-center cursor-pointer"
          onClick={() => addParticle('neutron')}
        >
          <Disc className="w-12 h-12 text-blue-500" />
          <span className="text-blue-500 mt-2">Neutron</span>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          className="flex flex-col items-center cursor-pointer"
          onClick={() => addParticle('electron')}
        >
          <Electron className="w-12 h-12 text-green-500" />
          <span className="text-green-500 mt-2">Electron</span>
        </motion.div>
      </div>
    </div>
  );
};