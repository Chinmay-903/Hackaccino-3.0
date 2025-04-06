import { motion } from 'framer-motion';
import { useAtomStore } from '../store/atomStore';

const ELEMENTS = [
  ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
  ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
  ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
  ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
  ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
];

export const PeriodicTable = () => {
  const { protons } = useAtomStore();
  const currentElement = protons.length;

  return (
    <motion.div
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      className="fixed right-4 top-4 bg-gray-900 bg-opacity-80 p-4 rounded-lg"
    >
      <h3 className="text-white text-lg font-bold mb-3">Periodic Table</h3>
      <div className="grid gap-1">
        {ELEMENTS.map((row, rowIndex) => (
          <div key={rowIndex} className="flex gap-1">
            {row.map((element, colIndex) => (
              element ? (
                <motion.div
                  key={`${rowIndex}-${colIndex}`}
                  className={`w-8 h-8 flex items-center justify-center rounded text-xs
                    ${element.length > 2 ? 'text-[10px]' : ''}
                    ${ELEMENTS[rowIndex][colIndex] === element && currentElement === rowIndex * 18 + colIndex + 1
                      ? 'bg-blue-500 shadow-lg shadow-blue-500/50'
                      : 'bg-gray-800'}`}
                  whileHover={{ scale: 1.1 }}
                >
                  {element}
                </motion.div>
              ) : (
                <div key={`${rowIndex}-${colIndex}`} className="w-8 h-8" />
              )
            ))}
          </div>
        ))}
      </div>
    </motion.div>
  );
};