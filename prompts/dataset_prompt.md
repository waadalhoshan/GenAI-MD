# Materials Discovery Lab — Synthetic Dataset

This CSV contains 500 synthetic material candidates for the Mesa-based "Materials Discovery Lab" simulation.

## Columns
- **id**: unique material ID (M0001–M0500)
- **density**: g/cm³, sampled U(2.0, 15.0)
- **hardness**: Mohs-like, sampled U(1.0, 10.0)
- **conductivity**: relative 0–100, sampled U(0, 100)
- **cost**: $/kg, sampled U(5, 200)

## Suggested scoring (compute inside the simulation)
Normalize each property to [0,1] using dataset min/max or fixed bounds, then:

score = 0.35·norm(hardness) + 0.35·norm(conductivity) + 0.20·(1 - norm(density)) + 0.10·max(0, 1 - cost/50)

Where a lower density and lower cost are better; higher hardness and conductivity are better.

## Typical bounds (clip after mutation)
- density ∈ [2.0, 15.0]
- hardness ∈ [1.0, 10.0]
- conductivity ∈ [0.0, 100.0]
- cost ∈ [5.0, 200.0]

## Notes
- Keep mutation steps small (±5–10%) and use imitation from better neighbors to see convergence.
- Do NOT store `score` in the CSV; compute it per-step to reflect evolving objectives or noise.
