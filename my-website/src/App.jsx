import { useState, useRef, useEffect } from 'react';
import './App.css';
import boatImage from '/boat.png';

export default function App() {
  // Simulation parameters
  const [riverWidth, setRiverWidth] = useState(400);
  const [riverDepth, setRiverDepth] = useState(300);
  const [boatSpeed, setBoatSpeed] = useState(5);
  const [riverSpeed, setRiverSpeed] = useState(3);
  const [angle, setAngle] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [time, setTime] = useState(0);
  const [drift, setDrift] = useState(0);
  const [showVectors, setShowVectors] = useState(true);
  const [showGrid, setShowGrid] = useState(false);
  const [waterDisturbance, setWaterDisturbance] = useState(true);
  const [preset, setPreset] = useState('custom');
  const [showControls, setShowControls] = useState(true);

  // Boat position and animation
  const [boatPosition, setBoatPosition] = useState({ x: 50, y: 0 });
  const [boatWake, setBoatWake] = useState([]);
  const animationRef = useRef();
  const lastTimeRef = useRef();
  const hasReachedEnd = useRef(false);

  // Calculate derived values
  const angleRad = angle * Math.PI / 180;
  const boatVelocity = {
    x: boatSpeed * Math.cos(angleRad),
    y: boatSpeed * Math.sin(angleRad)
  };
  const resultantVelocity = {
    x: boatVelocity.x + riverSpeed,
    y: boatVelocity.y
  };
  
  const resultantSpeed = Math.sqrt(resultantVelocity.x**2 + resultantVelocity.y**2);
  const timeToCross = resultantVelocity.y !== 0 ? riverDepth / Math.abs(resultantVelocity.y) : Infinity;
  const theoreticalDrift = timeToCross * resultantVelocity.x;
  let optimalAngle = riverSpeed !== 0 ? Math.acos(-riverSpeed / boatSpeed) * 180 / Math.PI : 0;
  
  // Ensure the optimal angle stays within 0 to 180 degrees
  optimalAngle = Math.max(0, Math.min(180, optimalAngle));
  
  // Presets
  const applyPreset = (preset) => {
    switch(preset) {
      case 'shortestTime':
        setAngle(90); // Straight across
        setPreset('shortestTime');
        break;
      case 'noDrift':
        setAngle(optimalAngle);
        setPreset('noDrift');
        break;
      case 'maxDrift':
        setAngle(riverSpeed > 0 ? Math.min(180, 45) : Math.max(0, 135)); // Ensure within bounds
        setPreset('maxDrift');
        break;
      default:
        setPreset('custom');
    }
  };

  // Start/stop the simulation
  const toggleSimulation = () => {
    if (isRunning) {
      cancelAnimationFrame(animationRef.current);
      setIsRunning(false);
      hasReachedEnd.current = false;
    } else {
      lastTimeRef.current = performance.now();
      setBoatPosition({ x: 50, y: 0 });
      setBoatWake([]);
      setTime(0);
      setDrift(0);
      hasReachedEnd.current = false;
      animate();
      setIsRunning(true);
    }
  };

  // Reset simulation
  const resetSimulation = () => {
    cancelAnimationFrame(animationRef.current);
    setIsRunning(false);
    hasReachedEnd.current = false;
    setBoatPosition({ x: 50, y: 0 });
    setBoatWake([]);
    setTime(0);
    setDrift(0);
  };

  // Animation loop
  const animate = () => {
    const now = performance.now();
    const deltaTime = (now - lastTimeRef.current) / 1000;
    lastTimeRef.current = now;

    if (!hasReachedEnd.current) {
      setTime(prevTime => prevTime + deltaTime);
    }
    
    setBoatPosition(prevPos => {
      if (hasReachedEnd.current) {
        return { x: prevPos.x, y: 100 };
      }

      const newY = prevPos.y + resultantVelocity.y * deltaTime * 0.5;
      const newX = prevPos.x + resultantVelocity.x * deltaTime * 0.5;
      
      // Add wake effect
      if (waterDisturbance && deltaTime > 0) {
        setBoatWake(prev => [
          ...prev.slice(-20),
          {
            x: prevPos.x,
            y: prevPos.y,
            time: 0,
            id: Date.now()
          }
        ]);
      }

      if (newY >= 100) {
        hasReachedEnd.current = true;
        setIsRunning(false);
        setDrift(newX - 50);
        return { x: newX, y: 100 };
      }
      
      return { x: newX, y: newY };
    });

    // Update wake animation
    setBoatWake(prev => 
      prev.map(point => ({
        ...point,
        time: point.time + deltaTime,
        size: Math.min(point.time * 5, 30)
      })).filter(point => point.time < 2)
    );

    if (!hasReachedEnd.current && boatPosition.y < 100) {
      animationRef.current = requestAnimationFrame(animate);
    }
  };

  // Handle parameter changes
  useEffect(() => {
    resetSimulation();
  }, [riverWidth, riverDepth, boatSpeed, riverSpeed, angle]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === ' ') toggleSimulation();
      if (e.key === 'r') resetSimulation();
      if (e.key === 'ArrowLeft') setAngle(prev => Math.max(prev - 2, -45));
      if (e.key === 'ArrowRight') setAngle(prev => Math.min(prev + 2, 45));
      if (e.key === 'c') setShowControls(prev => !prev);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Get river direction in degrees (0 for right, 180 for left)
  const riverDirection = riverSpeed <= 0 ? 180 : 0;

  return (
    <div className="app-container">
      <button 
        className={`toggle-controls ${showControls ? 'active' : ''}`}
        onClick={() => setShowControls(prev => !prev)}
      >
        {showControls ? '◄ Hide Controls' : '► Show Controls'}
      </button>
      
      {showControls && (
        <div className="controls-panel">
          <div className="controls">
            <h2>River Boat Simulation</h2>
            
            <div className="control-section">
              <h3>Presets</h3>
              <div className="preset-buttons">
                <button 
                  className={preset === 'shortestTime' ? 'active' : ''}
                  onClick={() => applyPreset('shortestTime')}
                >
                  Shortest Time
                </button>
                <button 
                  className={preset === 'noDrift' ? 'active' : ''}
                  onClick={() => applyPreset('noDrift')}
                  disabled={boatSpeed <= riverSpeed}
                >
                  No Drift {boatSpeed <= riverSpeed && '(Not possible)'}
                </button>
                <button 
                  className={preset === 'maxDrift' ? 'active' : ''}
                  onClick={() => applyPreset('maxDrift')}
                >
                  Max Drift
                </button>
              </div>
            </div>
            
            <div className="control-section">
              <h3>Environment</h3>
              <div className="control-group">
                <label>River Width: {riverWidth}m</label>
                <input 
                  type="range" 
                  min="100" 
                  max="800" 
                  value={riverWidth}
                  onChange={(e) => setRiverWidth(Number(e.target.value))}
                />
              </div>
              <div className="control-group">
                <label>River Depth: {riverDepth}m</label>
                <input 
                  type="range" 
                  min="100" 
                  max="600" 
                  value={riverDepth}
                  onChange={(e) => setRiverDepth(Number(e.target.value))}
                />
              </div>
              <div className="control-group">
                <label>River Speed: {riverSpeed}m/s</label>
                <input 
                  type="range" 
                  min="-10" 
                  max="10" 
                  step="0.1"
                  value={riverSpeed}
                  onChange={(e) => setRiverSpeed(Number(e.target.value))}
                />
              </div>
            </div>
            
            <div className="control-section">
              <h3>Boat Configuration</h3>
              <div className="control-group">
                <label>Boat Speed: {boatSpeed}m/s</label>
                <input 
                  type="range" 
                  min="0.5" 
                  max="15" 
                  step="0.1"
                  value={boatSpeed}
                  onChange={(e) => setBoatSpeed(Number(e.target.value))}
                />
              </div>
              <div className="control-group">
                <label>Boat Angle: {angle}°</label>
                <input 
                  type="range" 
                  min="0" 
                  max="90" 
                  value={angle}
                  onChange={(e) => {
                    setAngle(Number(e.target.value));
                    setPreset('custom');
                  }}
                />
              </div>
            </div>
            
            <div className="control-section">
              <h3>Visualization</h3>
              <div className="control-group">
                <label>
                  <input 
                    type="checkbox" 
                    checked={showVectors}
                    onChange={() => setShowVectors(!showVectors)}
                  />
                  Show Velocity Vectors
                </label>
              </div>
              <div className="control-group">
                <label>
                  <input 
                    type="checkbox" 
                    checked={showGrid}
                    onChange={() => setShowGrid(!showGrid)}
                  />
                  Show Grid
                </label>
              </div>
              <div className="control-group">
                <label>
                  <input 
                    type="checkbox" 
                    checked={waterDisturbance}
                    onChange={() => setWaterDisturbance(!waterDisturbance)}
                  />
                  Water Disturbance
                </label>
              </div>
            </div>
            
            <div className="action-buttons">
              <button 
                className={`simulate-button ${isRunning ? 'stop' : 'start'}`}
                onClick={toggleSimulation}
              >
                {isRunning ? '⏹ Stop' : '▶ Start'} Simulation
              </button>
              <button className="reset-button" onClick={resetSimulation}>
                ↻ Reset
              </button>
            </div>
            
            <div className="keyboard-hint">
              <p>Keyboard shortcuts: Space=Start/Stop, R=Reset, Arrows=Adjust Angle, C=Toggle Controls</p>
            </div>
          </div>
        </div>
      )}
      
      <div className="simulation-area">
        <div 
          className="river" 
          style={{ 
            width: `${riverWidth}px`,
            height: `${riverDepth}px`
          }}
        >
          {/* Horizontal grass corners */}
          <div className="river-top-grass"></div>
          <div className="river-bottom-grass"></div>

          {/* River background elements */}
          {showGrid && (
            <div className="river-grid">
              {Array.from({ length: Math.floor(riverWidth/50) }).map((_, i) => (
                <div key={`col-${i}`} className="grid-line vertical" style={{ left: `${i * 50}px` }}></div>
              ))}
              {Array.from({ length: Math.floor(riverDepth/50) }).map((_, i) => (
                <div key={`row-${i}`} className="grid-line horizontal" style={{ top: `${i * 50}px` }}></div>
              ))}
            </div>
          )}
          
          {/* River flow animation */}
          <div className="river-flow">
            {Array.from({ length: 30 }).map((_, i) => (
              <div 
                key={`flow-${i}`}
                className="flow-particle"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 5}s`,
                  opacity: 0.3 + Math.random() * 0.4
                }}
              ></div>
            ))}
          </div>
          
          {/* River banks with grass */}
          <div className="river-bank left">
            <div className="grass-overlay"></div>
          </div>
          <div className="river-bank right">
            <div className="grass-overlay"></div>
          </div>
          
          {/* Boat wake effect */}
          {waterDisturbance && boatWake.map((point) => (
            <div 
              key={point.id}
              className="wake-point"
              style={{
                left: `${point.x}%`,
                top: `${point.y}%`,
                width: `${point.size}px`,
                height: `${point.size}px`,
                opacity: 1 - (point.time / 2)
              }}
            ></div>
          ))}
          
          {/* Boat */}
          <div 
            className="boat"
            style={{
              left: `${boatPosition.x}%`,
              top: `${boatPosition.y}%`,
              transform: `rotate(${angle}deg)`,
              transition: isRunning ? 'none' : 'transform 0.3s ease'
            }}
          >
            <img src={boatImage} alt="Boat" className="boat-image" />
            {isRunning && (
              <div className="boat-propell"></div>
            )}
          </div>
          
          {/* Velocity vectors */}
          {showVectors && (
            <>
              <div 
                className="vector boat-vector"
                style={{
                  left: `${boatPosition.x}%`,
                  top: `${boatPosition.y}%`,
                  width: `${boatSpeed * 15}px`,
                  transform: `rotate(${angle}deg)`
                }}
              >
                <div className="vector-label">V<sub>boat</sub> ({boatSpeed}m/s)</div>
              </div>
              
              <div 
                className="vector river-vector"
                style={{
                  left: `${boatPosition.x}%`,
                  top: `${boatPosition.y}%`,
                  width: `${Math.abs(riverSpeed) * 15}px`,
                  transform: `rotate(${riverDirection}deg)`,
                  backgroundColor:  '#4dabf7'
                }}
              >
                <div className="vector-label">V<sub>river</sub> ({Math.abs(riverSpeed).toFixed(1)}m/s)</div>
              </div>
              
              <div 
                className="vector resultant-vector"
                style={{
                  left: `${boatPosition.x}%`,
                  top: `${boatPosition.y}%`,
                  width: `${resultantSpeed * 15}px`,
                  transform: `rotate(${Math.atan2(resultantVelocity.y, resultantVelocity.x) * 180/Math.PI}deg)`
                }}
              >
                <div className="vector-label">V<sub>result</sub> ({resultantSpeed.toFixed(1)}m/s)</div>
              </div>
            </>
          )}
        </div>
        
        <div className="simulation-stats">
          <div className="stats-card">
            <h3>Real-time Data</h3>
            <div className="stat-item">
              <span className="stat-label">Time Elapsed:</span>
              <span className="stat-value">{time.toFixed(2)}s</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Current Drift:</span>
              <span className="stat-value">
                {((boatPosition.x - 50) * riverWidth / 100).toFixed(1)}m
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Boat Position:</span>
              <span className="stat-value">
                ({(boatPosition.x * riverWidth / 100).toFixed(1)}m, {(boatPosition.y * riverDepth / 100).toFixed(1)}m)
              </span>
            </div>
          </div>
          
          <div className="stats-card">
            <h3>Theoretical Values</h3>
            <div className="stat-item">
              <span className="stat-label">Time to Cross:</span>
              <span className="stat-value">
                {timeToCross !== Infinity ? timeToCross.toFixed(2) + 's' : 'Will not cross'}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Final Drift:</span>
              <span className="stat-value">{theoreticalDrift.toFixed(1)}m</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Optimal Angle:</span>
              <span className="stat-value">
                {boatSpeed > riverSpeed ? optimalAngle.toFixed(1) + '°' : 'Not possible'}
              </span>
            </div>
          </div>
          
          <div className="theory-card">
            <h3>Physics Breakdown</h3>
            <div className="formula">
              V<sub>x</sub> = V<sub>b</sub>·cosθ - V<sub>r</sub> = {resultantVelocity.x.toFixed(2)}m/s
            </div>
            <div className="formula">
              V<sub>y</sub> = V<sub>b</sub>·sinθ = {resultantVelocity.y.toFixed(2)}m/s
            </div>
            <div className="formula">
              |V| = √(V<sub>x</sub>² + V<sub>y</sub>²) = {resultantSpeed.toFixed(2)}m/s
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}