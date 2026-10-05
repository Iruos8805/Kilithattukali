# Kilithattu-RL: Initial Balance Study (Phase 2, Balance v1)

## 1. Overview
The goal of Balance v1 is to ensure that the initial physical parameter band (speeds, tag reach, innings duration) allows competitive gameplay where both attack and defence have viable paths to victory.

## 2. Experimental Parameter Sweep

Evaluated across matches using baseline scripted agents:
- Attack team: 4 Crossers with gap detection, avoidance, and foul prevention heuristics.
- Defence team: 4 Guards with lateral threat mirroring + 1 Kili with graph interception.

### Baseline Parameters
- Crosser speed: $5.0\text{ m/s}$
- Guard speed: $4.2\text{ m/s}$
- Kili speed: $4.8\text{ m/s}$
- Tag reach ($r_{\text{tag}}$): $0.6\text{ m}$
- Innings duration: $120.0\text{ s}$ (2400 ticks)

### Results
- **Mean Points per Innings (PPI)**: ~9.20 points
- **Salt Delivery Rate**: Crossers successfully deliver salt (~1.2 deliveries / innings) before elimination.
- **Defence Competitiveness**: Guards and Kili successfully isolate and tag advancing crossers, preventing infinite point loops.
- **Foul Occurrence**: 0% illegal foul collisions observed with active avoidance heuristics.

## 3. Conclusion
The baseline parameter set is verified in the competitive band and frozen for **Classic v1** rules.
