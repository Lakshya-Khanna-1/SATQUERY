# SatQuery AI — Impeccable Design Context (DESIGN.md)

> **Product Domain**: Earth Observation Vision-Language Assistant & Multi-Agent Intelligence System (SIH 2026 • ISRO / SAC).

---

## 🎨 Design Systems & Tokens

### 1. Typography
- **Primary Display & Headings**: `Outfit`, sans-serif (Weights: 600, 700, 800). Modern, geometric, precise.
- **Body & UI Controls**: `Outfit` or `Inter`, sans-serif (Weights: 400, 500). High legibility at small sizes.
- **Telemetry & Technical Data**: `JetBrains Mono` or `Fira Code`, monospace. Used for Lat/Lon coordinates, EPSG codes, timestamps, and model latency metrics.

### 2. Color Palette & Tinting
- **Surface Dark (Background)**: `#09090b` (Tinted obsidian zinc, not pure black).
- **Surface Elevation**: `#18181b` (Zinc-900 card surface) with border `#27272a` (Zinc-800).
- **Primary Accent**: `#06b6d4` (Cyan 500) & `#0891b2` (Cyan 600). Represents satellite laser/sensor telemetry.
- **Secondary Accent**: `#10b981` (Emerald 500). Represents vegetation and confidence indicators.
- **Action Highlight**: `#3b82f6` (Blue 500). Represents primary user interaction controls.
- **Warning/Alert**: `#f59e0b` (Amber 500). Used for resampling flags and rate-limit fallbacks.

### 3. Spatial Rhythm & Component Rules
- **No Nested Card Containers**: Section boundaries defined via clean typography spacing, line dividers, and subtle surface elevation rather than card-in-card boxes.
- **Tactile Interactive Controls**: Buttons and interactive chips feature `transition-all duration-200 ease-out`, hover elevation, and `active:scale-[0.98]` tactile feel.
- **Glassmorphism & Modals**: Modals use `backdrop-blur-md` with `bg-black/60` backdrops for smooth depth separation.
