# Specialist B Real Bi-Temporal Satellite Test Report (LEVIR-CC)

**Dataset:** LEVIR-CC Real Satellite Bi-Temporal Images (Texas, USA)
**Model:** Qwen2.5-VL-3B LoRA Adapter (`specialist_b`) + 4-Bit NF4 Quantization + Gemini 2.5 Flash Multimodal Synthesizer
**Total Test Cases:** 10 (5 Construction/Change Pairs, 5 No-Change Pairs)

## Summary of Results

| Test # | Type | Image Pair | Ground Truth Caption | Raw Model Answer | Gemini Synthesized Natural Language Answer |
|---|---|---|---|---|---|
| #01 | **Change** | `test_000004.png` | the vegetation has been replaced by a road and many villas around . | `The road is missing.` | The road is missing. |
| #02 | **Change** | `test_000017.png` | a road appears at the bottom and many houses are scattered replacing the trees . | `The trees have grown.` | The trees have grown. |
| #03 | **Change** | `test_000018.png` | the road has been rebuilt and the plants have been replaced by many villas . | `the person is no longer there` | the person is no longer there |
| #04 | **Change** | `test_000019.png` | the trees have been replaced by many buildings . | `The road is no longer visible.` | The road is no longer visible. |
| #05 | **Change** | `test_000020.png` | several houses appear on the left of the forest clearing . | `the white barn is gone` | the white barn is gone |
| #06 | **No Change** | `test_000001.png` | there is no difference . | `the trees have grown.` | the trees have grown. |
| #07 | **No Change** | `test_000002.png` | there is no difference . | `The trees have grown.` | The trees have grown. |
| #08 | **No Change** | `test_000003.png` | there is no difference . | `The trees have grown.` | The trees have grown. |
| #09 | **No Change** | `test_000005.png` | there is no difference . | `the trees have grown` | the trees have grown |
| #10 | **No Change** | `test_000006.png` | there is no difference . | `the road is missing` | the road is missing |


## Detailed Pair-by-Pair Analysis

### Pair #01 (`test_000004.png`) - [Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"the vegetation has been replaced by a road and many villas around ."*
  - *"trees are removed and an arc road is built with some buildings around ."*
  - *"a winding road is built and many houses are constructed beside it to replace the former vegetation ."*
  - *"forest is taken by the houses along the road ."*
  - *"the former forest has been replaced by a residential area with roads and houses ."*
- **Raw Model Answer:** `The road is missing.`
- **Gemini Natural Language Answer:** The road is missing.

### Pair #02 (`test_000017.png`) - [Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"a road appears at the bottom and many houses are scattered replacing the trees ."*
  - *"trees are removed and neatly arranged residential buildings with a circular road at the top of the area and a turning road through the buildings ."*
  - *"a road is built and many villas are constructed neatly on both sides of it ."*
  - *"a forest is built into residential areas and roads ."*
  - *"the desert has turned into a residential area with road which houses are built alongside ."*
- **Raw Model Answer:** `The trees have grown.`
- **Gemini Natural Language Answer:** The trees have grown.

### Pair #03 (`test_000018.png`) - [Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"the road has been rebuilt and the plants have been replaced by many villas ."*
  - *"trees and the road are replaced by a new road with neatly arranged residential buildings and another circular road at the top ."*
  - *"a road is built and many villas are constructed on both sides of it and the vegetation is also replaced by houses ."*
  - *"a large number of trees and space are replaced by a residential area and road ."*
  - *"the woods with a road going through have turned into the residential area with roads and houses ."*
- **Raw Model Answer:** `the person is no longer there`
- **Gemini Natural Language Answer:** the person is no longer there

### Pair #04 (`test_000019.png`) - [Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"the trees have been replaced by many buildings ."*
  - *"trees are replaced by residential buildings and the old arc dirt road is rebuilt ."*
  - *"a winding road is built and many houses are constructed around it to replace the former vegetation ."*
  - *"houses and roads take place of forest and spare ground ."*
  - *"the road is reconstructed with houses built alongside ."*
- **Raw Model Answer:** `The road is no longer visible.`
- **Gemini Natural Language Answer:** The road is no longer visible.

### Pair #05 (`test_000020.png`) - [Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"several houses appear on the left of the forest clearing ."*
  - *"some trees on the left disappear and some houses appear instead ."*
  - *"several houses are constructed on the bareland at the edge of the scene ."*
  - *"three houses are built on the ground ."*
  - *"three houses show up in the desert ."*
- **Raw Model Answer:** `the white barn is gone`
- **Gemini Natural Language Answer:** the white barn is gone

### Pair #06 (`test_000001.png`) - [No Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"there is no difference ."*
  - *"the two scenes seem identical ."*
  - *"the scene is the same as before ."*
  - *"no change has occurred ."*
  - *"almost nothing has changed ."*
- **Raw Model Answer:** `the trees have grown.`
- **Gemini Natural Language Answer:** the trees have grown.

### Pair #07 (`test_000002.png`) - [No Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"there is no difference ."*
  - *"the two scenes seem identical ."*
  - *"the scene is the same as before ."*
  - *"no change has occurred ."*
  - *"almost nothing has changed ."*
- **Raw Model Answer:** `The trees have grown.`
- **Gemini Natural Language Answer:** The trees have grown.

### Pair #08 (`test_000003.png`) - [No Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"there is no difference ."*
  - *"the two scenes seem identical ."*
  - *"the scene is the same as before ."*
  - *"no change has occurred ."*
  - *"almost nothing has changed ."*
- **Raw Model Answer:** `The trees have grown.`
- **Gemini Natural Language Answer:** The trees have grown.

### Pair #09 (`test_000005.png`) - [No Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"there is no difference ."*
  - *"the two scenes seem identical ."*
  - *"the scene is the same as before ."*
  - *"no change has occurred ."*
  - *"almost nothing has changed ."*
- **Raw Model Answer:** `the trees have grown`
- **Gemini Natural Language Answer:** the trees have grown

### Pair #10 (`test_000006.png`) - [No Change]
- **Question:** What changed between the two images?
- **Ground Truth Captions:**
  - *"there is no difference ."*
  - *"the two scenes seem identical ."*
  - *"the scene is the same as before ."*
  - *"no change has occurred ."*
  - *"almost nothing has changed ."*
- **Raw Model Answer:** `the road is missing`
- **Gemini Natural Language Answer:** the road is missing

