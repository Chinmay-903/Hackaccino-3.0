import { motion } from 'framer-motion';
import { useAtomStore } from '../store/atomStore';

export const AtomicInfo = () => {
  const { protons, neutrons, electrons } = useAtomStore();

  const atomicNumber = protons.length;
  const atomicMass = protons.length + neutrons.length;
  
  const getElementName = (protonCount: number) => {
    const elements = [
      'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
      'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon',
      'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus',
      'Sulfur', 'Chlorine', 'Argon', 'Potassium', 'Calcium',
      'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese',
      'Iron', 'Cobalt', 'Nickel', 'Copper', 'Zinc',
      'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine',
      'Krypton', 'Rubidium', 'Strontium', 'Yttrium', 'Zirconium',
      'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium', 'Rhodium',
      'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin',
      'Antimony', 'Tellurium', 'Iodine', 'Xenon'
    ];
    return elements[protonCount - 1] || 'Unknown';
  };

  const getElectronicConfiguration = () => {
    const SHELLS = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s'];
    const MAX_ELECTRONS = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2];
    
    let remainingElectrons = electrons.length;
    let config = [];
    
    for (let i = 0; i < SHELLS.length && remainingElectrons > 0; i++) {
      const electronsInShell = Math.min(MAX_ELECTRONS[i], remainingElectrons);
      if (electronsInShell > 0) {
        config.push(`${SHELLS[i]}${electronsInShell}`);
      }
      remainingElectrons -= electronsInShell;
    }
    
    return config.join(', ');
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
        <p>Configuration: {getElectronicConfiguration()}</p>
      </div>
    </motion.div>
  );
};