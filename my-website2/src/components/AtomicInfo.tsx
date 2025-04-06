import { motion } from 'framer-motion';
import { useAtomStore } from '../store/atomStore';

export const AtomicInfo = () => {
  const { protons, neutrons, electrons } = useAtomStore();

  const atomicNumber = protons.length;
  const atomicMass = protons.length + neutrons.length;
  
  const getElementName = (protonCount: number) => {
    const elements = [
      'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
      'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon'
    ];
    return elements[protonCount - 1] || 'Unknown';
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="fixed left-4 top-4 bg-gray-900 bg-opacity-80 p-6 rounded-lg text-white"
    >
      <h2 className="text-2xl font-bold mb-4">Atomic Information</h2>
      <div className="space-y-2">
        <p>Element: {getElementName(atomicNumber)}</p>
        <p>Atomic Number: {atomicNumber}</p>
        <p>Mass Number: {atomicMass}</p>
        <p>Electrons: {electrons.length}</p>
        <p>Charge: {protons.length - electrons.length}</p>
      </div>
    </motion.div>
  );
};