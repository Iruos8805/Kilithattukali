# Kilithattu Kali — Canonical Rules Specification (RULES.md)

> **Document status:** Canonical specification (v1.0.0)  
> **Source of truth:** All simulation environments (`kilithattu-core`), reference models, and game logic must adhere to this document.

---

## 1. Overview & Terminology

**Kilithattu Kali** (also known as *Uppukali*) is a traditional team tag game of Kerala, historically played during Onam.
Two teams compete in alternating attack and defence innings:
- **Defence**: 1 **Kili** (the roaming line leader) + $(n_B - 1)$ **Guards** (line-bound defenders).
- **Attack**: $n_a$ **Crossers** (outbound = *Pacha* / green; inbound after reaching far zone = *Uppu* / salt).

### Glossary
- **Crosser**: An attacking player moving freely in 2D to cross all lines, become *uppu*, and return home.
- **Guard**: A defender constrained to move along a single assigned transverse line $L_k$ ($k \in \{1, \dots, n_B - 1\}$).
- **Kili**: The primary defender who starts on the front line ($L_0$), claps to start the innings, and can traverse any segment of the field line graph.
- **Pacha ("Green")**: Status of a crosser traveling outbound from the Home zone toward the Far zone.
- **Uppu ("Salt")**: Status of a crosser who has successfully crossed line $L_{n_B}$ into the Far zone without being tagged.
- **Salt Delivery**: An *uppu* successfully returning across $L_0$ into the Home zone (+6 bonus points, completes 1 round trip).
- **Foul**: An *uppu* and a *pacha* occupying the exact same box at the end of a simulation tick. The offending crosser is declared **OUT**.
- **Out**: A crosser eliminated by tag, foul, or side-boundary exit for the remainder of the innings.

---

## 2. Field Geometry & Topology

Default coordinates are in metres.
- Lanes: $n_L = 2$, Lane width: $w = 3.0\text{ m}$, Total width: $W = n_L \cdot w = 6.0\text{ m}$.
- Box rows: $n_B = 5$, Row depth: $d = 3.0\text{ m}$, Field length: $L = n_B \cdot d = 15.0\text{ m}$.
- Safe zones: Home zone $y \in [-z, 0)$ where $z = 2.0\text{ m}$; Far zone $y \in (L, L + z]$.

```
       x=0             x=3             x=6
 y=17  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄   Far-zone boundary (safe)
                FAR ZONE (Safe) → UPPU
 y=15   J═══════════════J═══════════════J    L5  No fixed guard (Kili accessible)
        ║    box (0,4)  ║    box (1,4)  ║
 y=12   J───────────────J───────────────J    L4  Guard 4  ◄─►
        ║    box (0,3)  ║    box (1,3)  ║
 y=9    J───────────────J───────────────J    L3  Guard 3  ◄─►
        ║    box (0,2)  ║    box (1,2)  ║
 y=6    J───────────────J───────────────J    L2  Guard 2  ◄─►
        ║    box (0,1)  ║    box (1,1)  ║
 y=3    J───────────────J───────────────J    L1  Guard 1  ◄─►
        ║    box (0,0)  ║    box (1,0)  ║
 y=0    J═══════════════J═══════════════J    L0  Kili start (centre junction)
                HOME ZONE (Safe), PACHA
 y=-2  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄   Home-zone boundary (safe)

 J = Junction (graph node where Kili may change lines)
 ║ = Longitudinal line (x = 0, 3, 6)
 ─ / ═ = Transverse line (L0 … L5)
```

### Lines and Graph Definitions
- **Transverse Lines** $L_k$: segments at $y = k \cdot d$ for $k \in \{0, \dots, n_B\}$, spanning $x \in [0, W]$.
- **Longitudinal Lines**: segments at $x = j \cdot w$ for $j \in \{0, \dots, n_L\}$, spanning $y \in [0, L]$.
- **Junctions (Nodes)**: $(x, y) = (j \cdot w, k \cdot d)$ for $j \in \{0, \dots, n_L\}, k \in \{0, \dots, n_B\}$. Total nodes = $(n_L + 1)(n_B + 1)$ (18 by default).
- **Boxes $(j, k)$**: $j = \min(\lfloor x / w \rfloor, n_L - 1)$, $k = \lfloor y / d \rfloor$ for $0 \le y < L$. A crosser on a boundary line belongs to the higher-index box.

---

## 3. Classic Rules (R1 – R14)

| ID | Rule | Description |
|---|---|---|
| **R1** | **Field Layout** | The play area consists of $n_L \times n_B$ boxes with boundary lines and safe zones as specified in Section 2. |
| **R2** | **Safe Zones** | Crossers inside Home zone ($y \in [-z, 0)$) or Far zone ($y \in (L, L+z]$) are completely immune to tags. |
| **R3** | **Initial Setup** | Crossers start in the Home zone with non-overlapping positions. The Kili begins at the central junction of line $L_0$ ($(W/2, 0)$). Each guard $k$ starts on transverse line $L_k$ ($k \in \{1, \dots, n_B - 1\}$) at a uniformly random $x \in [0, W]$. |
| **R4** | **The Clap** | Play begins upon the Kili's clap. In Classic rules, clap occurs automatically at $t = 2.0\text{ s}$. Crossers are held behind line $L_0$ ($y < 0$) prior to the clap. Defenders may adjust positions before the clap. The innings clock starts immediately on the clap. |
| **R5** | **Guard Movement** | Guard $k$ is strictly constrained to transverse line $L_k$ with $x \in [0, W]$ and $y = k \cdot d$. |
| **R6** | **Kili Movement** | The Kili moves along the line network (both transverse and longitudinal lines) and may transition between intersecting lines only at junctions. Mid-tick turning is snapped within reach. |
| **R7** | **Crosser Movement & Side Exits** | Crossers move continuously in 2D. Stepping outside side boundaries ($x < 0$ or $x > W$) while in the field ($0 \le y \le L$) immediately results in the crosser being **OUT**. |
| **R8** | **Continuous Swept Tag** | A defender tags a crosser if their relative distance is $\le r_{\text{tag}}$ ($0.6\text{ m}$) at *any moment* during the tick step (continuous swept-circle collision). Safe-zone immunity applies. A tagged crosser is eliminated (**OUT**) for the innings. |
| **R9** | **Status Lifecycle** | Crossers enter as **Pacha**. Crossing $L_{n_B}$ into the Far zone transitions status to **Uppu**. Crossing $L_0$ into the Home zone delivers salt, awards bonus points, and resets status to **Pacha** for another trip. |
| **R10** | **The Uppu/Pacha Foul** | If an *uppu* and a *pacha* simultaneously occupy the same box at the end of a tick, the crosser that entered that box during that tick is **OUT**. If both entered in the same tick, the *uppu* is **OUT**. |
| **R11** | **Scoring & Monotonic Progress** | Each line crossed in sequence awards $+1$ point. Delivering salt awards $+6$ bonus points (total 18 points per full round trip). Re-crossing lines backwards or repeating lines without progression awards 0 points (anti-farming). |
| **R12** | **Innings End** | An innings terminates when all crossers are OUT or the live clock reaches $T = 120.0\text{ s}$. |
| **R13** | **Match Structure** | A full match consists of 4 innings (teams alternate attack and defence twice). Highest aggregate score wins. Ties trigger one pair of 60s extra innings; subsequent ties broken by total salt deliveries, then draw. |
| **R14** | **Resolution Order** | Within each simulation tick: Tags resolve **before** line crossings, ensuring defender precedence in simultaneous boundary events. |

---

## 4. Scoring Mechanism

Progress is tracked per crosser via a trip counter $\kappa \in \{0, \dots, 2(n_B + 1)\}$:
- **Outbound ($0 \to n_B + 1$):**
  - Cross $L_0$ ($y \ge 0$): $\kappa = 1$, $+1$ point.
  - Cross $L_1, \dots, L_{n_B - 1}$: $\kappa = 2 \dots n_B$, $+1$ point per line.
  - Cross $L_{n_B}$ into Far Zone: $\kappa = n_B + 1$ (6 by default), $+1$ point; status $\to$ **Uppu**.
- **Inbound ($n_B + 1 \to 2(n_B + 1)$):**
  - Re-enter field across $L_{n_B}$ ($y \le L$): $\kappa = n_B + 2$ (7 by default), $+1$ point.
  - Cross $L_{n_B - 1}, \dots, L_1$ in $-y$ direction: $\kappa = 8 \dots 11$, $+1$ point per line.
  - Cross $L_0$ into Home Zone: $\kappa = 2(n_B + 1)$ (12 by default), $+1$ point $+$ **$+6$ bonus** points.
  - Reset $\kappa \leftarrow 0$, trips $\leftarrow \text{trips} + 1$, status $\to$ **Pacha**.
- **Total Points per Trip**: $(n_B + 1) + (n_B + 1) + 6 = 6 + 6 + 6 = 18\text{ points}$.

---

## 5. Kilithattu+ Feature Toggles (P1 – P7)

In Classic mode, all toggles are OFF (`false`). In Kilithattu+ mode, P1–P4 are active by default.

| Toggle | Name | Mechanism | Parameters |
|---|---|---|---|
| **P1** | **Momentum & Stamina** | Acceleration limit $a_{\text{max}}$. Sprint/dash consumes stamina $e \in [0, 1]$; walking regenerates stamina. Stamina $< 0.10$ disables sprint. | $a_{\text{max}} = 20\text{ m/s}^2$, drain rate $0.35/\text{s}$, regen rate $0.20/\text{s}$, sprint thresh $0.10$. |
| **P2** | **Facing & Turning** | Defenders face toward Home or Far. Tags restricted to within $\pm 90^\circ$ of facing cone. Turning takes $t_{\text{turn}} = 0.30\text{ s}$, during which tagging is disabled. | Tag cone $\pm 90^\circ$, $t_{\text{turn}} = 0.30\text{ s}$. |
| **P3** | **Limited Vision (Fog)** | Players perceive others only within field-of-view cone and maximum radius. Unseen entities are masked in observations with last-seen memory. | Crosser: $200^\circ, 9\text{ m}$; Guard: $180^\circ, 9\text{ m}$; Kili: $360^\circ, 12\text{ m}$. |
| **P4** | **Strategic Clap** | Kili manually triggers clap within window $[t_{\text{min}}, t_{\text{max}}]$. Crossers may maneuver in Home zone. Crossing $L_0$ before clap = **False Start** $\to$ OUT. | Clap window $[1.0, 5.0]\text{ s}$. |
| **P5** | **Procedural Fields** | Variable field dimensions: $n_L \in \{2, 3\}$, $n_B \in \{4, \dots, 9\}$, crossers $n_a \in \{3, \dots, 6\}$. | Padded tensor representation with existence masks. |
| **P6** | **Team Signalling (Calls)** | Team members emit one of $K = 4$ discrete vocal/tactical calls every 10 ticks ($0.5\text{ s}$), received only by teammates. | $K = 4$, period 10 ticks. |
| **P7** | **Human-Fair Reflexes** | Artificial observation latency (delay ring buffer) and sticky-action probability applied to agents. | Delay $U\{0, \dots, 5\}$ ticks, sticky prob $U[0, 0.10]$. |

---

## 6. Default Physical and Simulation Parameters

| Parameter | Symbol | Classic Default | Plus Default | Units |
|---|---|---|---|---|
| Simulation frequency | $f$ | 20 | 20 | Hz ($\Delta t = 0.05\text{ s}$) |
| Lanes | $n_L$ | 2 | 2 | integer |
| Box rows | $n_B$ | 5 | 5 | integer |
| Lane width | $w$ | 3.0 | 3.0 | m |
| Row depth | $d$ | 3.0 | 3.0 | m |
| Safe zone depth | $z$ | 2.0 | 2.0 | m |
| Crossers | $n_a$ | 4 | 4 | integer |
| Guards | $n_g$ | 4 ($n_B - 1$) | 4 | integer |
| Tag reach radius | $r_{\text{tag}}$ | 0.6 | 0.6 | m |
| Crosser speed (walk / sprint) | $v_c$ | 5.0 | 3.5 / 6.0 | m/s |
| Guard speed (walk / dash) | $v_g$ | 4.2 | 3.5 / 5.0 | m/s |
| Kili speed (walk / dash) | $v_k$ | 4.8 | 4.0 / 5.5 | m/s |
| Innings live duration | $T$ | 120.0 | 120.0 | s (2400 ticks) |

---

## 7. Decision Log

- **D1 (Kili Role & Lines)**: Kili starts at the central junction of $L_0$ and roams all lines. $L_5$ has no fixed guard, creating an entry challenge for the far zone.
- **D2 (Team Sizes)**: Standard match has 4 crossers vs 5 defenders (1 Kili + 4 guards), balancing offensive split options against defensive line coverage.
- **D3 (Uppu Lifecycle)**: Uppu status is acquired upon clearing $L_5$ into the Far zone. Returning across $L_0$ delivers salt and resets status.
- **D4 (18-Point Scoring)**: 1 point per line crossed plus 6 bonus points on return aligns with traditional Atya-patya *lona* scoring, providing smooth reward gradient.
- **D5 (Foul Penalty)**: Offending crosser is OUT. Simple, strict, and forces crosser scheduling.
- **D6 (Safe Zones)**: Home and Far zones are immune to tags, preventing spawn-camping by Kili.
- **D7 (Defender Precedence)**: Tags resolve before line crossings within the same tick.
- **D8 (Side Boundary Exits)**: Exiting through side boundaries while between $L_0$ and $L_5$ eliminates the crosser.
- **D9 (No Inter-Defender Collisions)**: Kili can pass through or share line segments with guards.
- **D10 (Swept Continuous Detection)**: Linear swept-circle test eliminates tunneling at high speeds.
- **D11 (Padded Tensor Architecture)**: Max 6 crossers, 8 guards, 10 transverse lines, 4 longitudinal lines with boolean masks for uniform JIT compilation.
- **D12 (Innings Termination)**: All crossers out or clock expiry ends the innings immediately.
