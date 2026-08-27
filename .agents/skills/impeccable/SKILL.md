---
name: impeccable
description: >-
  Impeccable UI/UX design language and quality engineering principles.
  Use when designing, auditing, polishing, styling, typeset-refining, or animating web interfaces to prevent generic AI SaaS templates and achieve high visual craft, distinct typography, harmonious color tinting, clean rhythm, and responsive delight.
---

# Impeccable UI/UX Design Skill & Principles

Impeccable is a design language and quality engineering skill for AI-assisted frontend development created by Peter Akaus. It enforces high visual craft, distinct typography, harmonious spatial rhythm, and purposeful micro-interactions while deterministically avoiding generic AI SaaS design anti-patterns.

---

## 🎨 Core Design Principles

1. **Distinctive Typography over Generic Defaults**:
   - Avoid standard `Inter` / `Arial` fallback clutter for everything. Use characterful typography tailored to the product domain (e.g. `Outfit`, `Space Grotesk`, `Cabinet Grotesk`, `JetBrains Mono`, `Syne`, `DM Sans`).
   - Establish crisp hierarchy: scale heading weights (700-900) against crisp body readability (400-500).

2. **Harmonious Color Tinting & High Contrast**:
   - **No Pure Grayscale**: Never use `#000000` or neutral dead grays (`#888888`). Always tint grays with warm/cool primary accents (e.g., slate-900 with cyan/indigo undertones, zinc-950 with teal/emerald highlights).
   - **No Gray Text on Colored Backgrounds**: Ensure all text passes WCAG AA/AAA contrast standards against dark/light container surfaces.
   - **Vibrant Strategic Accents**: Avoid cliché purple-to-blue gradients. Use tailored, rich accent pairings (cyan/teal/emerald/gold).

3. **Structural Rhythm & Layout Clarity**:
   - **No Cards-Inside-Cards**: Stop wrapping every subsection inside nested container boxes. Use whitespace, subtle borders, grid alignment, typography scale, and background surface contrast to define section bounds.
   - **Grid Alignment & Proportions**: Maintain consistent 8px/12px grid baseline spacing for paddings, gaps, and section margins.

4. **Purposeful Motion & Micro-Interactions**:
   - Use smooth, natural easing (`cubic-bezier(0.16, 1, 0.3, 1)`) instead of dated bounce/elastic easing.
   - Enhance interactive buttons, tabs, chips, and modals with subtle hover elevation, active scale-down (`active:scale-[0.98]`), and smooth transitions.

5. **Harden Edge Cases & User Feedback**:
   - Design delightful empty states, responsive mobile adaptation, clean loading telemetry, text truncation (`truncate`/`line-clamp`), and clear status indicators.

---

## 🛠️ Impeccable Command Vocabulary

- **`/impeccable craft`**: Full shape-then-build visual flow.
- **`/impeccable audit`**: Technical quality check (contrast, responsiveness, performance).
- **`/impeccable polish`**: Visual refinement pass & shipping readiness.
- **`/impeccable bolder`**: Amplify visual hierarchy and contrast for under-designed elements.
- **`/impeccable quieter`**: Tone down busy layouts and reduce visual noise.
- **`/impeccable distill`**: Strip redundant UI clutter to its core essence.
- **`/impeccable typeset`**: Refine font pairings, leading, line lengths, and scale.
- **`/impeccable animate`**: Add smooth micro-animations and transition states.
- **`/impeccable colorize`**: Apply rich, purposeful color palettes.
- **`/impeccable layout`**: Fix alignment, grid spacing, and visual rhythm.
- **`/impeccable harden`**: Add error states, edge-case fallbacks, and truncation safety.

---

## 🚫 Deterministic Anti-Patterns (Avoid at all costs)

- ❌ Defaulting to `Inter` font everywhere without intentional selection.
- ❌ Cliché purple-to-blue background gradients.
- ❌ Nested card boxes inside parent cards.
- ❌ Gray text on dark colored buttons/pills.
- ❌ Generic rounded-square icon tiles sitting above every section heading.
- ❌ Abrupt transition state jumps without CSS timing functions.
