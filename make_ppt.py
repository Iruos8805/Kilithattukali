from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

DARK_BG = RGBColor(10, 20, 16)
GOLD = RGBColor(229, 169, 60)
WHITE = RGBColor(241, 245, 249)
MUTED = RGBColor(148, 163, 184)
GREEN = RGBColor(52, 211, 153)

def set_bg(slide):
    bg = slide.shapes.add_shape(1, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK_BG
    bg.line.fill.background()

def add_header(slide, tag, title, time_str):
    set_bg(slide)
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = tag.upper()
    p0.font.size = Pt(10)
    p0.font.bold = True
    p0.font.color.rgb = GOLD
    p1 = tf.add_paragraph()
    p1.text = title
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    tb2 = slide.shapes.add_textbox(Inches(10.5), Inches(0.4), Inches(2.0), Inches(0.8))
    tf2 = tb2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = time_str
    p2.alignment = PP_ALIGN.RIGHT
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = GOLD

def add_card(slide, left, top, width, height, title, bullets):
    card = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = RGBColor(18, 32, 26)
    card.line.color.rgb = RGBColor(46, 125, 50)
    card.line.width = Pt(1)
    tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.15), Inches(width - 0.4), Inches(height - 0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN
    for b in bullets:
        p_b = tf.add_paragraph()
        p_b.text = "• " + b
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = WHITE
        p_b.space_before = Pt(4)

# Slide 1
s1 = prs.slides.add_slide(blank_layout)
add_header(s1, "Slide 1 of 6 • Title & Team Details", "Kilithattu-RL: Multi-Agent RL in Line-Constrained Traditional Games", "1:00 min")
add_card(s1, 0.8, 1.6, 5.7, 5.2, "Team & Academic Credentials", [
    "Souri S (AM.SC.U4AIE23151) — Environment & MAPPO Implementation",
    "Vishnu Tejas E (AM.SC.U4AIE23164) — Self-Play League & Game Client",
    "Dept. of AI, Amrita Vishwa Vidyapeetham, Amritapuri",
    "Domain: Multi-Agent RL, Two-Team Zero-Sum Markov Game (POSG)"
])
add_card(s1, 6.8, 1.6, 5.7, 5.2, "Game Structure & Decision Space", [
    "4 Crossers (Attack): Free 2D planar movement",
    "4 Line Guards (Defense): 1D lateral movement on assigned transverse lines",
    "1 Kili (Defense Captain): Roams perimeter & central spine graph",
    "Goal: Cross 6 transverse lines outbound to become Uppu, return safely without fouls",
    "Impact: Extreme spatial asymmetry with delayed rewards and zero human demonstration data"
])

# Slide 2
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "Slide 2 of 6 • Problem Background & Statement", "Cultural Context & Formal Game-Theoretic Problem Statement", "1:45 min")
add_card(s2, 0.8, 1.6, 5.7, 5.2, "Why Classical AI & Supervised Learning Fail", [
    "Zero Human Datasets: Oral tradition lacks recordings for behavioral cloning",
    "Non-Stationary Adversaries: Fixed heuristics & FSMs are rapidly exploited by adaptive feints",
    "Continuous 20 Hz Dynamics: 9 simultaneous agents make MCTS / tree search intractable",
    "Hard Team Foul: Uppu/Pacha rule demands coordinated scheduling, not just proximity avoidance"
])
add_card(s2, 6.8, 1.6, 5.7, 5.2, "Formal Problem Statement", [
    "Formulated as Zero-Sum Markov Game: G = <N, S, {Ai}, T, r_A = -r_D, gamma>",
    "Target: Team-Maxmin Equilibrium with Correlation (TMECor)",
    "Objective: max_pi_A min_pi_D E[ sum gamma^t * r_A(s_t, a_t) ]",
    "Goal: Maximize Points Per Innings (PPI) over 120s while minimizing foul rate & exploitability"
])

# Slide 3
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "Slide 3 of 6 • Literature Review & Novelty", "Benchmarking SOTA & Key Research Contributions", "1:30 min")
add_card(s3, 0.8, 1.6, 5.7, 5.2, "State-of-the-Art & Gaps", [
    "MAPPO (Yu et al., 2022): Tested only on cooperative benchmarks (SMAC, Football)",
    "Hide-and-Seek (Baker et al., 2020) & AlphaStar (2019): Homogeneous free-movement agents",
    "Fictitious Co-Play (Strouse et al., 2021): Evaluated only in 2-player cooperative cooking",
    "Gap 1: No benchmark features mixed 2D, 1D, and graph movement in one arena",
    "Gap 2: Prior MARL lacks hard team-internal spatial constraints (Uppu/Pacha foul)"
])
add_card(s3, 6.8, 1.6, 5.7, 5.2, "Proposed Novel Components", [
    "First Formal Model: Complete, verified JAX implementation of Kilithattu Kali",
    "Coupled Team Constraints: Mathematical formulation of intra-team spatial traffic fouls",
    "Exploitability Framework: Quantitative cross-table study uncovering cyclic vulnerability vs scripts"
])

# Slide 4
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "Slide 4 of 6 • MDP Design & Algorithm", "Tokenized Observation, Action Masks & Potential Shaping", "2:00 min")
add_card(s4, 0.8, 1.6, 5.7, 5.2, "MDP Formulation", [
    "Entity Tokens: 9 players x 27, 14 lines x 8, 1 global (~238 values; invariant to field size)",
    "State Features: Positions, velocities, kappa trip counter (0-12), patience timers",
    "Discrete Masked Actions: Crosser (9 dirs + sprint), Guard (3 actions), Kili (5 graph actions)",
    "Centralized Critic: Unrestricted global state during training (no fog/masks)"
])
add_card(s4, 6.8, 1.6, 5.7, 5.2, "Algorithm & Reward Formulation", [
    "Base Reward: r_A = (Delta Lines + 6 * Delta Delivery) / 18,  r_D = -r_A",
    "Potential Shaping: beta * (gamma * Phi(s') - Phi(s)),  Phi = mean crosser progress",
    "Policy Invariance: Phi(s_terminal) = 0 strictly preserves optimal Nash equilibria",
    "MAPPO Config: gamma=0.997, GAE lambda=0.95, Adam lr=3e-4, entropy decay 0.02 -> 0.003"
])

# Slide 5
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "Slide 5 of 6 • System Design & Pipeline", "End-to-End Implementation: Core, Train, Eval & Game", "2:00 min")
add_card(s5, 0.8, 1.6, 5.7, 5.2, "Engineering Pipeline", [
    "kilithattu-core: JAX 20 Hz engine, continuous swept-tag collision, Python reference verified",
    "kilithattu-train: 2,048 parallel GPU envs (jax.vmap), Transformer encoder + GRU memory",
    "kilithattu-eval: 37 automated property tests, exploitability best-response harness",
    "kilithattu-game: Pygame 60 fps client, <10 ms CPU inference per tick, .kreplay bit-for-bit logs"
])
add_card(s5, 6.8, 1.6, 5.7, 5.2, "Curriculum & Architecture", [
    "Curriculum Stages: S0 (solo navigation) -> S5 (full contact + fouls) -> S6 (Plus rules)",
    "Slot Architecture: Human, RL policy, or scripted bot interchangeable in any player slot",
    "Pure JAX GPU Execution: Eliminates CPU-GPU memory bottleneck during rollout and updates"
])

# Slide 6
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "Slide 6 of 6 • Current Progress & Findings", "50% Milestone Evidence, Cross-Table & Final Roadmap", "1:45 min")
add_card(s6, 0.8, 1.6, 5.7, 5.2, "Milestone Progress & Cross-Table Results", [
    "Engine Verified: 37/37 unit tests pass; differential-tested over 10,000 rollouts",
    "Throughput: 1,950 steps/s on local RTX 3050 GPU (vs 114 on CPU)",
    "Stage 0 Learning: Solo agent achieves 3 full round trips (64 pts) within 100 updates",
    "Learned Attack beats Scripted: 23.3 vs 20.3 PPI (78% win rate)",
    "Key Finding (Exploitability): Learned attack overfits to rushing; specialized defense crushes it (6.3 PPI) but collapses vs scripted attack (112.8 PPI)"
])
add_card(s6, 6.8, 1.6, 5.7, 5.2, "Roadmap for Remaining 50%", [
    "AlphaStar-Style League: PFSP matchmaking with main and league exploiters",
    "Multi-Seed Verification: 3 seeds with 95% confidence intervals and ablation studies",
    "Fictitious Co-Play (FCP): Diverse partner-pool training for robust human-AI teaming"
])

prs.save("Kilithattu_RL_MidReview.pptx")
print("Successfully generated Kilithattu_RL_MidReview.pptx")