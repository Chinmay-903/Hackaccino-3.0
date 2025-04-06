import { motion } from 'framer-motion';
import { useAtomStore } from '../store/atomStore';

// Correct periodic table layout with proper atomic numbers
const ELEMENTS = [
  // Period 1
  ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'], 
  // Period 2
  ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
  // Period 3
  ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
  // Period 4
  ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
  // Period 5
  ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
];

// Map element symbols to atomic numbers
const ELEMENT_TO_ATOMIC_NUMBER: Record<string, number> = {
  'H': 1, 'He': 2,
  'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
  'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
  'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 
  'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 
  'Br': 35, 'Kr': 36,
  'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
  'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52,
  'I': 53, 'Xe': 54
};

export const PeriodicTable = () => {
  const { protons, setAtomConfiguration } = useAtomStore();
  const currentElement = protons.length;

  const handleElementClick = (elementSymbol: string) => {
    const atomicNumber = ELEMENT_TO_ATOMIC_NUMBER[elementSymbol];
    if (atomicNumber) {
      setAtomConfiguration(atomicNumber);
    }
  };

  // Function to check if an element should be highlighted
  const shouldHighlight = (elementSymbol: string) => {
    return ELEMENT_TO_ATOMIC_NUMBER[elementSymbol] === currentElement;
  };

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
                  className={`w-8 h-8 flex items-center justify-center rounded text-xs cursor-pointer
                    ${element.length > 2 ? 'text-[10px]' : ''}
                    ${shouldHighlight(element)
                      ? 'bg-blue-500 shadow-lg shadow-blue-500/50'
                      : 'bg-gray-800'}`}
                  whileHover={{ scale: 1.1 }}
                  onClick={() => handleElementClick(element)}
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