# SatQuery AI - Complete Project Report

**Problem Statement ID:** 26167  
**Programme:** Smart India Hackathon 2026  
**Problem Owner:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)  
**Category:** Software  
**Theme:** Space Technology  
**Repository:** [Lakshya-Khanna-1/SATQUERY](https://github.com/Lakshya-Khanna-1/SATQUERY)

---

## 1. Executive Summary

SatQuery AI is an interactive assistant for understanding satellite and remote-sensing imagery through ordinary language. A user can upload one satellite image or a related pair of images and ask questions such as:

- What land cover and major objects are visible here?
- Highlight the water body in this image.
- What changed between these two dates?
- Has the built-up area increased or decreased?
- Use an optical image and a radar image together to study the region.

Instead of expecting the user to understand Geographic Information Systems (GIS), satellite sensors, image bands, coordinate systems, model selection, or machine-learning parameters, SatQuery AI is designed to make those decisions automatically.

The central idea is that no single artificial-intelligence model is ideal for every remote-sensing task. SatQuery therefore uses a **cooperating specialist architecture**. A controller examines the query and available images, identifies the requested task, selects the relevant specialist model, collects its output, and returns a structured result containing:

- a natural-language answer;
- the identified task and input type;
- the model or specialist used;
- visual evidence such as a highlighted region or change overlay;
- a confidence indicator;
- processing metadata and latency; and
- an auditable execution trace.

The current repository contains a functioning research and demonstration prototype. It has a web interface, three specialist services, an agentic LangGraph controller, input and GeoTIFF utilities, model-training code, locally available trained adapters and checkpoints, session history, coordinate-based imagery retrieval, an n8n-compatible front door, and PDF report generation.

The project should presently be described as a **working multi-specialist prototype with an end-to-end application architecture**. Some important functions are fully represented in code but still need stronger integration, calibration, benchmark evidence, and production hardening. These current limitations are documented transparently in this report.

---

## 2. The Problem SatQuery AI Addresses

### 2.1 Why satellite imagery is difficult to use

Satellite images contain valuable information for agriculture, disaster management, urban planning, forestry, water-resource assessment, infrastructure monitoring, defence, logistics, and environmental analysis. However, extracting that information normally requires several kinds of expertise:

- understanding the characteristics of different satellite sensors;
- selecting useful spectral bands;
- reading geospatial metadata;
- working with coordinate reference systems (CRS);
- aligning images captured at different times or by different sensors;
- selecting the correct machine-learning model;
- interpreting segmentation masks, classification outputs, or change maps; and
- preparing the result in a form that decision-makers can use.

This creates a gap between the people who need satellite intelligence and the specialised tools required to obtain it.

### 2.2 Why one image or one model is often insufficient

Different observations reveal different kinds of information:

- **Optical and multispectral imagery** provides colour, spectral, texture, and contextual information.
- **Synthetic Aperture Radar (SAR)** provides structural and surface-response information and can operate at night and through cloud cover.
- **Bi-temporal images** show the same place at different dates and are necessary for studying change.

A generic vision-language model may describe a normal photograph well, but remote-sensing imagery has a different viewpoint, scale, spectral composition, terminology, and class distribution. It may also require pixel-level localisation or comparison between multiple observations. The problem statement therefore requires remote-sensing adaptation and agentic selection of suitable specialist models rather than dependence on a single unadapted model.

### 2.3 Problems with many existing approaches

Existing solutions are commonly limited in one or more of the following ways:

1. **Single-task design:** Separate tools are used for classification, detection, visual question answering, change detection, and SAR analysis.
2. **Expert-oriented workflow:** Users must know which model, sensor, band, projection, and parameters to select.
3. **Weak multimodal reasoning:** An optical image, SAR image, and historical image pair are often processed independently.
4. **Limited explainability:** A prediction may be returned without spatial evidence or a record of how it was produced.
5. **Generic-model dependence:** Stock VLMs may not understand small objects, land-cover patterns, or domain-specific remote-sensing language reliably.
6. **Fragmented output:** Text, masks, metadata, confidence, and reporting may live in separate tools.
7. **High compute requirements:** Large monolithic models are difficult to run on consumer or workstation GPUs.

SatQuery AI is intended to bring these operations together in one guided, query-driven application.

---

## 3. Proposed Solution

SatQuery AI works like a conversational front end to a set of remote-sensing tools.

The user provides imagery and asks a question. The system then performs the following high-level process:

1. **Understand the request** - determine whether the user wants a description, answer, highlighted region, change analysis, or optical-SAR analysis.
2. **Inspect the inputs** - check how many images were supplied and whether they can be decoded; read GeoTIFF metadata when available.
3. **Select a specialist** - route the request to the model intended for that type of task.
4. **Run the analysis** - generate a text answer, embedding, highlighted region, or change map.
5. **Prepare the response** - convert model output into readable language and collect available visual evidence.
6. **Record what happened** - report the task, input modality, models, latency, confidence, CRS, and timestamp.
7. **Present and preserve the result** - show it in the interface, store the conversation locally, and optionally generate a PDF report.

This approach reduces the need for the user to manually build a GIS or machine-learning workflow.

---

## 4. Product Philosophy and Ideology

SatQuery AI is built around the following principles.

### 4.1 Natural language should be the control layer

Users should be able to state the information they need without knowing which technical pipeline is required. The query is treated as an instruction to the system, not merely as text to be answered by a chatbot.

### 4.2 Specialists are better than one universal model

A model trained for single-image question answering has different strengths from a model trained to compare two dates or align optical and SAR information. SatQuery separates these responsibilities so that each component can be trained, evaluated, replaced, or improved independently.

### 4.3 Agentic AI should take observable actions

The project uses the word "agentic" to mean that the controller does more than generate prose. It classifies the task, validates inputs, chooses a tool, calls the tool, collects evidence, and records the execution. The visible tool-use trace is more important than hidden reasoning text.

### 4.4 Evidence should accompany conclusions

Where possible, a textual answer should be supported by a highlighted region, segmentation overlay, image pair, change mask, class breakdown, or geospatial metadata.

### 4.5 Domain adaptation is essential

Remote-sensing imagery should not be treated as an ordinary photograph. The architecture includes LoRA adapters and a custom multimodal/change model so that specialist components can learn remote-sensing tasks and terminology.

### 4.6 The system should remain feasible on limited hardware

The selected vision-language base model is approximately three billion parameters, and the adaptation strategy uses QLoRA rather than a full fine-tune. The remaining specialist uses a comparatively lightweight ResNet-18 and U-Net architecture. This modular approach is intended for a single consumer or workstation GPU rather than a large data-centre deployment.

### 4.7 The interface should hide unnecessary complexity

The main interaction is intentionally simple: attach imagery, ask a question, inspect the answer and evidence, and expand the technical trace only when required.

---

## 5. Supported Use Cases

### 5.1 Single-image visual question answering

The user uploads one image and asks a factual question about visible land cover, objects, infrastructure, water, vegetation, roads, or other scene content.

Example:

> Are there built-up structures in this image?

### 5.2 Scene description and captioning

The user requests a general description of the satellite scene.

Example:

> Describe the land cover and important objects visible here.

### 5.3 Text-guided grounding and highlighting

The user names a feature and asks the application to locate or highlight it. The current implementation combines vision-language grounding with Segment Anything Model 2 (SAM 2), with limited spectral fallbacks for some land-cover types.

Example:

> Highlight the water body in this image.

### 5.4 Bi-temporal change question answering

The user supplies an earlier image and a later image and asks what changed. A dedicated vision-language specialist processes both images in a multi-image prompt.

Example:

> Has the built-up area increased between Time 1 and Time 2?

### 5.5 Spatial change-map generation

The custom segmentation component can generate a transparent overlay representing predicted change classes such as water, forest, vegetation, buildings, and bare land.

### 5.6 Optical-SAR paired analysis

The custom fusion component creates aligned feature representations for optical and SAR observations. Its purpose is to use complementary spectral and structural information rather than relying on either modality alone.

### 5.7 Coordinate-based observation retrieval

The interface includes a coordinate selector. The backend can obtain current and historical map imagery and query the Element84 Earth Search STAC service for Sentinel-2 pass metadata. This supports a convenient demonstration in which the user selects a location and period instead of manually preparing both images.

### 5.8 Auditable reporting

Results can be displayed with their execution trace and exported into a PDF intelligence report containing the query, imagery, analysis, confidence, and processing summary.

---

## 6. Inputs and Outputs

### 6.1 Intended input scope

The problem statement defines three input configurations:

| Input configuration | Meaning | Typical tasks |
|---|---|---|
| Single image | One optical, multispectral, or SAR observation | VQA, captioning, grounding |
| Bi-temporal pair | The same place at two dates | Change-VQA, change description, change mapping |
| Cross-modal pair | Co-registered optical/multispectral and SAR observations | Fusion and joint analysis |

GeoTIFF and TIFF are the primary geospatial formats. PNG and JPEG are intended mainly for prescribed public benchmark data and demonstrations.

### 6.2 Current API representation

The web application reads uploaded files as Base64 data URLs and sends them to the backend in JSON. The backend decodes the images and attempts to inspect GeoTIFF metadata when the byte header indicates a TIFF.

### 6.3 Returned result

A normal result can contain:

- the original query;
- a readable answer;
- a task type such as `vqa`, `caption`, `grounding`, `change_vqa`, or `fusion`;
- the input modality;
- a confidence value;
- one or more input images;
- an evidence overlay;
- a list of specialists and model names;
- validation status;
- CRS or GeoTIFF metadata;
- execution latency; and
- a timestamp.

---

## 7. End-to-End User Journey

### 7.1 Starting an analysis

The user opens the local web application, starts a new session, and either uploads images or selects a location through the coordinate interface.

### 7.2 Preparing imagery

One uploaded image is treated as a single observation. When two images are attached, the interface allows the user to assign them as Time 1 and Time 2. Images are ordered before submission so that the comparison direction remains clear.

### 7.3 Asking a question

The user enters a natural-language query and chooses Quick or Deep analysis:

- **Quick mode** requests a concise answer.
- **Deep mode** requests a longer, structured geospatial explanation.

### 7.4 Processing

The request normally goes through the n8n-compatible webhook proxy to the LangGraph orchestrator. The interface displays processing messages such as task classification, spatial validation, and specialist execution.

### 7.5 Viewing the result

The result card displays:

- confidence or failure status;
- the source imagery;
- an optional evidence overlay;
- the synthesised answer; and
- a collapsible execution trace.

### 7.6 Saving and exporting

Messages are stored in a local SQLite database. Sessions can be reopened, pinned, or deleted. A user can also request a downloadable PDF report.

---

## 8. System Architecture

### 8.1 High-level architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                        SatQuery Web UI                       │
│ Upload / Coordinates / Query / Evidence / Trace / PDF       │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTP + JSON/Base64
                               ▼
┌──────────────────────────────────────────────────────────────┐
│               n8n Workflow or Local Proxy :5678             │
│                  Webhook and retry front door                │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI + LangGraph Orchestrator :8000         │
│ Classify → Validate → Select → Execute → Combine → Log      │
└───────────────┬──────────────────┬──────────────────┬─────────┘
                │                  │                  │
                ▼                  ▼                  ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│ Specialist A      │  │ Specialist B      │  │ Specialist C      │
│ VQA + Grounding   │  │ Change-VQA        │  │ Fusion + Mask     │
│ Port 8001         │  │ Port 8002         │  │ Port 8003         │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          └──────────────────────┴──────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────┐
│ Answer + Evidence + Confidence + Execution Trace            │
│ SQLite History + ReportLab PDF                              │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 Architectural layers

#### Presentation layer

The React application handles upload, image ordering, coordinate selection, query entry, Quick/Deep selection, results, overlays, trace display, history, theme selection, and report download.

#### Entry and reliability layer

An exported n8n workflow defines a webhook, an HTTP call to the orchestrator, and a response node. Its HTTP request is configured to retry selected server errors. For the local demonstration, a FastAPI proxy implements the same basic forwarding behaviour without requiring a full n8n installation.

#### Orchestration layer

LangGraph holds the request state and executes five nodes in sequence:

1. `classify_task`
2. `validate_input`
3. `select_and_call_specialist`
4. `combine_outputs`
5. `log_execution`

The current classifier uses Gemini 2.5 Flash when API keys are available and deterministic keyword rules as a fallback. The same Gemini service is used to rewrite the specialist output into Quick or Deep natural language. When Gemini is unavailable, the raw specialist output is returned with an explicit offline notice.

#### Specialist layer

Each major task is exposed through an independent FastAPI service. This isolates model loading and allows services to be improved or deployed separately.

#### Data and geospatial layer

Python modules validate image files, extract GeoTIFF metadata, check CRS and grid compatibility, and reproject mismatched raster pairs using Rasterio.

#### Persistence and reporting layer

SQLite stores sessions and messages. ReportLab generates printable PDF intelligence reports from application results.

---

## 9. Specialist Models

### 9.1 Specialist A - Single-image VQA, captioning, and grounding

**Purpose:** Understand one optical or multispectral image and answer questions about it.

**Model approach:**

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Adaptation: LoRA/QLoRA adapter
- Training data: VRSBench and RSVQA-LR
- Service: FastAPI on port 8001

When the local model and adapter are available, the service loads the Hugging Face Qwen vision-language model directly and attaches the PEFT LoRA adapter. Model loading occurs in a background thread.

For grounding, the current working-tree version adds a two-stage process:

1. Gemini identifies candidate bounding boxes for the requested objects or features.
2. SAM 2 converts the boxes into segmentation masks.

If online grounding or SAM is unavailable, the service can fall back to bounding boxes or limited colour/spectral rules for water and vegetation. It then draws a cyan overlay and returns coverage and object-count information.

The service also contains a lightweight RGB-composition fallback for cases where the fine-tuned VLM cannot be loaded. This keeps the demonstration responsive, but such heuristic results must not be presented as equivalent to trained-model inference.

### 9.2 Specialist B - Bi-temporal Change-VQA

**Purpose:** Compare two images and answer natural-language questions about change.

**Model approach:**

- Base model: a separate copy of `Qwen/Qwen2.5-VL-3B-Instruct`
- Adaptation: task-specific LoRA/QLoRA adapter
- Training data: CDVQA questions and SECOND image pairs
- Prompt format: two images labelled as Before/Time 1 and After/Time 2
- Service: FastAPI on port 8002

When the adapted VLM is available, both images are included in the Qwen conversation and the model produces a change description. If the model is unavailable, the service computes a simple pixel-difference ratio and returns a generic change statement. That fallback provides operational continuity but does not offer semantic change understanding equivalent to Specialist B.

### 9.3 Specialist C - Optical-SAR fusion and change segmentation

**Purpose:** Represent complementary optical and radar information and produce pixel-level change evidence.

Specialist C contains two neural subsystems.

#### Dual-branch fusion encoder

- Optical branch: ImageNet-initialised ResNet-18
- SAR branch: ResNet-18 modified for two input channels
- Projection: both branches produce 256-dimensional embeddings
- Training objective: cross-modal similarity/contrastive alignment
- Output: optical embedding, SAR embedding, joint embedding, cosine similarity, and alignment score

#### Bi-temporal U-Net

- Input: two RGB observations concatenated into six channels
- Output: a six-class per-pixel mask
- Change classes: unchanged/background, water, forest/tree, low vegetation/agriculture, building/urban, and playground/bare land
- Visual output: transparent RGBA overlay with a distinct colour for each class

Specialist C is served directly as a PyTorch model through FastAPI on port 8003. A local checkpoint is present in the current working environment.

---

## 10. Agentic Orchestration

### 10.1 Request state

The LangGraph state stores the query, images, metadata, analysis mode, classified task, input modality, validation result, specialist responses, final answer, confidence, evidence image, execution trace, and start time.

### 10.2 Task classification

The controller recognises three main routes:

- single-image VQA/captioning/grounding;
- bi-temporal Change-VQA; and
- optical-SAR fusion.

Gemini is used for semantic classification when available. Keywords and image count provide deterministic fallback behaviour.

### 10.3 Automatic historical pairing

For historical-change queries, the controller can extract coordinates and a year interval from request metadata or query text. It then retrieves a historical baseline and pairs it with the supplied present image. If coordinates are absent, the current code uses a default Ahmedabad location.

### 10.4 Input validation

The runtime controller currently checks:

- whether an image is present;
- whether pair-based tasks have at least two images;
- whether each Base64 value can be decoded into a readable image; and
- whether a TIFF contains readable GeoTIFF metadata.

The repository also has a more complete pairing module capable of comparing CRS, bounds, resolution, transform, dimensions, and band count and reprojecting the second raster onto the first raster's grid. However, this full pairing function is not currently called by the main request graph.

### 10.5 Specialist selection

Current routing is:

| Task | Specialist called |
|---|---|
| VQA, captioning, grounding | Specialist A |
| Change-VQA | Specialist B |
| Optical-SAR fusion | Specialist C |

Although the architecture permits multiple specialists, the current request path selects one specialist for each task. Change-VQA does not presently call Specialist C automatically for an additional change map.

### 10.6 Response synthesis

The controller collects successful specialist answers, averages their returned confidence values, selects the latest evidence image, and asks Gemini to format the response. If every specialist fails, the failure is returned instead of being hidden behind a generated answer.

### 10.7 Execution trace

The final trace records:

- task type;
- Quick or Deep analysis mode;
- input modality;
- number of images;
- validation status;
- resampling flag;
- CRS;
- specialists invoked;
- model names;
- total latency;
- confidence; and
- UTC timestamp.

This trace is central to the explainability strategy.

---

## 11. Data and Preprocessing

### 11.1 Common dataset format

The repository defines a common sample structure across dataset loaders. A sample normally contains:

- a list of one or more images;
- a query;
- a reference answer;
- a task type; and
- dataset-specific metadata.

This makes it easier for training and evaluation code to work with different public datasets.

### 11.2 VRSBench

VRSBench is intended for remote-sensing captioning, grounding, and VQA. The loader can stream data through Hugging Face and convert its fields into the common SatQuery format.

### 11.3 RSVQA-LR

RSVQA-LR provides low-resolution satellite VQA questions and answers. It is used with VRSBench to adapt and evaluate Specialist A.

### 11.4 CDVQA and SECOND

CDVQA supplies change-related questions and answers, while SECOND provides corresponding before/after imagery. The loader joins the question records to paired images for Specialist B.

The current loader substitutes another available SECOND image when an exact filename match is missing. That behaviour is convenient for keeping a small demonstration dataset running, but it can break the relationship between a question and its correct scene. Proper benchmark evaluation should instead reject or explicitly report unmatched records.

### 11.5 BigEarthNet-MM

BigEarthNet-MM provides paired Sentinel-2 optical/multispectral and Sentinel-1 SAR observations. The loader uses the spatial-pairing module and prepares the pair for Specialist C.

### 11.6 GeoTIFF validation and pairing

The repository uses Rasterio to inspect:

- driver and file type;
- band count;
- width and height;
- data type;
- CRS;
- spatial transform;
- bounds; and
- pixel resolution.

The pairing module can use `rasterio.warp.reproject` to resample a mismatched image onto a reference grid. Bilinear resampling is available for imagery, while nearest-neighbour behaviour can be selected for categorical masks.

---

## 12. Training Methodology

### 12.1 Why LoRA and QLoRA are used

Training every parameter of a multi-billion-parameter VLM requires considerable GPU memory and time. LoRA learns a small set of low-rank adapter matrices while keeping most base-model parameters frozen. QLoRA further reduces memory use by loading the base model in four-bit form during training.

This provides three advantages:

1. Specialist A and Specialist B can share the same base architecture while retaining different learned behaviours.
2. Adapter files are much smaller than full model checkpoints.
3. Training becomes more feasible on a single consumer or workstation GPU.

### 12.2 Specialist A training

Specialist A uses a multimodal collator that formats the image, question, and answer using the Qwen chat template. The LoRA configuration targets attention and feed-forward projection layers with rank 16, alpha 32, and dropout 0.05.

### 12.3 Specialist B training

Specialist B uses a similar LoRA configuration, but its prompt contains two images and explicitly describes them as Time 1 and Time 2. It is trained separately so that change-language patterns do not compete with the single-image task.

### 12.4 Specialist C training

The fusion encoder is trained to make matching optical and SAR embeddings more similar. The segmentation head is intended to learn change classes from paired images and masks.

In the current script, the segmentation targets are generated from image-difference thresholds rather than reading true six-class SECOND reference masks. This means the current checkpoint demonstrates the architecture and inference path, but it should not yet be described as a validated semantic six-class change model.

### 12.5 Serving approach

The project documentation describes a possible production path in which LoRA adapters are merged with the base models, converted to GGUF, registered in Ollama, and served locally. That is a valid deployment concept, but it is not the active Specialist A/B implementation in the present code. The current services load Hugging Face models and PEFT adapters directly.

---

## 13. Technology Stack

### 13.1 Frontend

| Technology | Current purpose |
|---|---|
| React 19 | Component-based web interface |
| Vite 6 | Development server and production build tool |
| Tailwind CSS 4 | Interface styling and responsive layout |
| Lucide React | Interface icons |
| React Markdown | Rendering structured model answers |
| Browser FileReader and Canvas APIs | Image upload, preview, and coordinate-map handling |

Despite some README references to Next.js, the actual frontend is a React single-page application built with Vite.

### 13.2 Backend and orchestration

| Technology | Current purpose |
|---|---|
| Python 3.11/3.12 | Main backend, data, training, and model language |
| FastAPI | Orchestrator and specialist REST APIs |
| Uvicorn | Local ASGI service runner |
| Pydantic | Request and response validation |
| LangGraph | Stateful orchestration workflow |
| Requests / urllib | Service-to-service and external API communication |
| n8n workflow JSON | Webhook and retry front door |

### 13.3 Artificial intelligence and computer vision

| Technology | Current purpose |
|---|---|
| PyTorch | Training and inference framework |
| Torchvision | ResNet backbones and image transforms |
| Hugging Face Transformers | Qwen2.5-VL model loading and generation |
| PEFT | LoRA adapter training and loading |
| BitsAndBytes | Four-bit QLoRA quantisation |
| Accelerate | Device placement and efficient model execution |
| Qwen VL utilities | Preparing multimodal Qwen inputs |
| SAM 2 through Ultralytics | Prompted object/region segmentation |
| OpenCV | Masks, morphology, contours, and overlay generation |
| NumPy / SciPy | Pixel operations and lightweight visual analysis |
| Gemini 2.5 Flash API | Current intent classification, answer synthesis, and grounding boxes |

### 13.4 Geospatial and imagery technologies

| Technology | Current purpose |
|---|---|
| Rasterio/GDAL ecosystem | GeoTIFF reading, CRS, transforms, bounds, and reprojection |
| PyProj | Coordinate reference support |
| Affine | Raster transform handling |
| TIFFFile | TIFF data support |
| Pillow | Image decoding, conversion, resizing, and export |
| PySTAC Client | Querying the Element84 Earth Search STAC catalogue |

### 13.5 Data and storage

| Technology | Current purpose |
|---|---|
| Hugging Face Datasets | Streaming remote-sensing datasets |
| PyTorch Dataset classes | Unified model-ready samples |
| SQLite | Local sessions and message history |
| JSON/Base64 | API exchange format for metadata and images |

### 13.6 Reporting

| Technology | Current purpose |
|---|---|
| ReportLab | PDF intelligence-report generation |
| Matplotlib/Pillow-style overlays | Visual evidence production where applicable |

---

## 14. Frontend and User Experience

The interface follows a conversational pattern similar to modern AI assistants while exposing remote-sensing-specific functions.

Current features include:

- one or multiple image attachment;
- Time 1 and Time 2 ordering;
- coordinate and radius selector;
- current-versus-historical image retrieval;
- Quick and Deep response modes;
- switch between the n8n and direct-orchestrator paths;
- processing telemetry messages;
- input-image presentation;
- evidence-overlay presentation;
- confidence badge;
- Markdown-formatted answers;
- collapsible execution trace;
- PDF download;
- SQLite-backed analysis history;
- pinning and deletion of sessions; and
- light and dark themes.

The sidebar is collapsed by default, allowing the primary upload/query/result workflow to remain visually dominant.

---

## 15. Reporting and Persistence

### 15.1 SQLite memory

The application stores sessions and individual messages in `data/satquery_memory.db`. Message records can contain text, images, confidence, task type, analysis mode, evidence images, execution trace, and timestamp.

This is local application memory, not long-term model memory. It improves usability by allowing users to reopen earlier analyses.

### 15.2 PDF intelligence reports

The report generator can include:

- report date and confidence;
- the original question;
- the analytical answer;
- input imagery;
- a visual overlay;
- task and modality;
- specialists and models;
- latency; and
- spatial-alignment information.

The existing sample PDF proves that the reporting pipeline and layout exist. Its example imagery is synthetic and therefore does not constitute benchmark evidence.

---

## 16. How SatQuery AI Is Better Than a Conventional Approach

| Conventional approach | SatQuery AI approach |
|---|---|
| User selects separate tools manually | Query is automatically classified and routed |
| One generic model handles every task | Separate specialists handle understanding, comparison, fusion, and segmentation |
| Primarily text-only output | Text can be paired with highlighted regions or overlays |
| Limited visibility into processing | Observable execution trace records task, model, latency, and metadata |
| Separate workflows for one and two images | One interface accepts single, bi-temporal, and cross-modal inputs |
| Large-model fine-tuning is expensive | LoRA/QLoRA reduces adaptation cost |
| GIS expertise is required | Natural language becomes the main control surface |
| Results must be manually documented | PDF intelligence report can be generated automatically |
| Failure may be hidden behind a generated answer | The UI and controller expose explicit execution failures |

The strongest differentiation is not any one neural model. It is the integration of natural-language task selection, specialist execution, geospatial context, evidence, auditability, and reporting into one workflow.

---

## 17. Feasibility

### 17.1 Technical feasibility

The project is technically feasible because its major components already exist and communicate locally:

- the React frontend runs on port 3010;
- the n8n-compatible proxy runs on port 5678;
- the orchestrator runs on port 8000;
- Specialists A, B, and C run on ports 8001, 8002, and 8003;
- all five backend health endpoints respond successfully in the present environment;
- locally downloaded Qwen base-model files, LoRA adapter weights, the Specialist C checkpoint, and SAM 2 weights are present; and
- the source tree passes Python syntax parsing.

The modular service architecture also limits the effect of a single component failure and permits independent development.

### 17.2 Hardware feasibility

The design is aimed at one CUDA-capable consumer or workstation GPU:

- four-bit loading reduces the memory required by Qwen2.5-VL;
- LoRA adapters avoid separate full-model fine-tunes;
- models are split into services;
- images are resized before VLM inference; and
- Specialist C uses ResNet-18 and a lightweight U-Net rather than a very large vision transformer.

Exact VRAM, throughput, and latency claims should be measured under a controlled demonstration configuration before publication.

### 17.3 Data feasibility

All selected datasets are publicly accessible, and the architecture supports streamed or subset-based training rather than requiring the entire BigEarthNet archive. However, raw datasets are intentionally excluded from Git, so a reproducible setup process is required for a new machine.

### 17.4 Operational feasibility

The one-command launcher starts the three specialist services, orchestrator, proxy, and frontend. This is practical for a local SIH demonstration. Production deployment would require container orchestration, authentication, monitoring, request limits, secrets management, and a shared object store.

---

## 18. Viability and Potential Adoption

### 18.1 Primary beneficiaries

- government remote-sensing and planning departments;
- disaster-management authorities;
- agricultural and forestry agencies;
- water-resource and environmental teams;
- infrastructure and urban-development bodies;
- academic researchers and students; and
- organisations that need satellite insight but lack a dedicated GIS team.

### 18.2 Practical value

SatQuery can reduce the time required to move from an observation to an understandable result. It can also standardise the response format by attaching visual evidence and processing metadata to every analysis.

### 18.3 Deployment models

Possible deployment models include:

- an offline workstation for sensitive imagery;
- an internal departmental web service;
- an API integrated into Bhuvan, Bhoonidhi, or another imagery portal;
- a disaster-response command-centre tool;
- an educational or research platform; and
- a managed service for non-sensitive commercial imagery.

### 18.4 Cost considerations

The modular architecture can reduce training and inference costs because only the required specialist needs to run. LoRA adapters are comparatively small, and Specialist C is lightweight relative to a large general-purpose VLM.

The current use of Gemini introduces network dependence, API limits, and usage costs. A fully local deployment could replace it with the documented Ollama-based controller or another local language model after equivalent routing and synthesis quality is validated.

### 18.5 Sustainability

The specialist design supports incremental improvement. A new flood model, crop classifier, object detector, or higher-quality SAR fusion model can be registered without replacing the complete application.

---

## 19. Current Repository Status

The main `SATQUERY` directory currently contains:

- data validation, GeoTIFF parsing, pairing, retrieval, and dataset-loader modules;
- QLoRA training and evaluation scripts for Specialists A and B;
- training, evaluation, and model definitions for Specialist C;
- locally available base weights, adapters, SAM weights, and Specialist C checkpoint, excluded from Git because of size;
- three FastAPI specialist services;
- a LangGraph orchestrator and API;
- an n8n workflow and local proxy;
- a React/Vite frontend and existing production build;
- SQLite session storage;
- PDF report generators and a sample report;
- local launcher scripts; and
- SIH requirement and judge-oriented documentation.

The repository's `main` branch currently matches the remote `origin/main` commit `ea189fe`. The working tree also contains ongoing local changes to the Gemini synthesiser and Specialist A grounding implementation, as well as a changing SQLite database generated by application use.

The adjacent `unncc` folder is treated as supplementary material. It contains removed experiments, test assets, scratch files, and other non-core artifacts and is not part of the main application architecture described here.

---

## 20. Current Limitations and Gaps

This section distinguishes the intended architecture from the exact current implementation.

### 20.1 Full geospatial pairing is not connected to the main graph

The pairing module can detect and correct CRS, bounds, grid, and resolution mismatches. The orchestrator imports those functions but its active validation node does not call them. Consequently, `was_resampled` remains false in the normal request path even when a pair might require alignment.

**Required improvement:** Decode uploaded GeoTIFF pairs into temporary rasters or memory datasets, run the alignment check, return corrected imagery or a validation failure, and record the actual resampling parameters.

### 20.2 Modality is inferred mainly from query wording

The current classifier treats words such as "SAR," "radar," and "fusion" as evidence of a cross-modal pair. It does not robustly identify optical versus SAR data from band metadata or user-assigned modality.

**Required improvement:** Add explicit image roles and band-level modality validation.

### 20.3 Change-VQA and change-map generation are separate routes

The Change-VQA route calls only Specialist B. Specialist C is not automatically called to add a spatial change overlay, despite some older documentation describing combined execution.

**Required improvement:** Define a compound route that calls B for language and C for spatial evidence, then fuses both outputs.

### 20.4 Specialist C fusion path simplifies SAR input

In the generic `/infer` fusion endpoint, the second uploaded image is converted to grayscale and duplicated into two synthetic channels. This does not preserve genuine VV/VH SAR data.

**Required improvement:** Accept two-band SAR arrays or GeoTIFF bytes directly, apply SAR-specific normalisation, and validate the modality before inference.

### 20.5 Specialist C training uses pseudo change labels

The current segmentation training script converts pixel differences into pseudo classes instead of reading true semantic reference masks.

**Required improvement:** Map SECOND labels into a documented class taxonomy, train on genuine masks, and report IoU/F1 per class.

### 20.6 Confidence is not calibrated

Several services return fixed confidence values such as 0.92, 0.94, 0.95, or 0.96. Cross-modal similarity is converted linearly into an alignment score. These values are presentation indicators, not validated probability estimates.

**Required improvement:** Calibrate confidence on held-out validation data using task-appropriate methods and expose the calculation in the trace.

### 20.7 Some fallback answers are generic

When a large model is unavailable, Specialists A and B use pixel-colour or difference heuristics. This keeps the application usable but can produce broad statements unsupported by semantic recognition.

**Required improvement:** Clearly label fallback mode in the API and UI, lower its confidence, and restrict its claims to measurements it can actually compute.

### 20.8 Specialist A has duplicated inference code

`run_native_vqa` is currently defined twice in the service file. The later definition overrides the earlier one, increasing maintenance risk.

The current grounding integration can also interpret a general description request with no segmented object as "No scene overview detected" because a zero object count is treated as a negative detection.

**Required improvement:** Remove the duplicate, separate VQA from grounding response logic, and add regression tests for caption and overview requests.

### 20.9 Dataset matching can silently substitute an image

The CDVQA loader selects another available image if an exact referenced filename is missing. This risks pairing a valid question with an unrelated scene.

**Required improvement:** Maintain a validated mapping manifest and fail or skip explicitly when an item cannot be matched.

### 20.10 Documentation has drifted from the implementation

Examples include:

- documentation calls the frontend Next.js, while it is React/Vite;
- documentation describes an Ollama controller, while the active system uses Gemini;
- documentation lists export, Modelfile, and test files no longer present in the cleaned main repository; and
- some demonstration documentation describes multi-specialist change routing that is not present in the active graph.

**Required improvement:** Make this report and the code-generated API schema the source of truth, then update older README files.

### 20.11 Reproducibility is incomplete

Raw data and model weights are correctly excluded from Git, but the repository does not yet provide one complete automated bootstrap that downloads every required artifact and verifies hashes and versions. Some dependencies required at runtime are documented separately rather than declared together in one environment file.

**Required improvement:** Add a version-locked environment, model/data manifest, download instructions, checksum verification, and a smoke-test suite.

### 20.12 Production security is not implemented

Services currently allow broad CORS access and have no authentication, user isolation, rate limits, or encrypted remote storage. This is acceptable for a local prototype but not a public production deployment.

**Required improvement:** Add authentication, scoped CORS, secret management, request-size limits, audit retention, and deployment hardening.

### 20.13 Benchmark evidence needs strengthening

The repository contains small evaluation claims and synthetic demonstration reports, but SIH judging will use public test subsets and an undisclosed ISRO/SAC dataset.

**Required improvement:** Produce reproducible test-split results with sample counts, metric definitions, baselines, confidence intervals where practical, failure examples, and separate results for VQA, grounding, Change-VQA, fusion, and segmentation.

---

## 21. Risk Analysis and Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Domain shift from public data to Cartosat/RISAT | Reduced accuracy on hidden evaluation data | Sensor-agnostic preprocessing, metadata validation, augmentation, and cross-dataset testing |
| Optical-SAR misalignment | Incorrect fusion or change conclusions | Connect the existing Rasterio alignment engine to runtime processing |
| Cloud/API unavailability | Intent, synthesis, or grounding failure | Retain deterministic routing and move synthesis/grounding to validated local models |
| GPU memory pressure | Slow start or out-of-memory errors | Four-bit loading, smaller images, one active specialist at a time, model lifecycle management |
| Uncalibrated confidence | Misleading user trust | Validation-set calibration and traceable confidence components |
| Hallucinated natural-language conclusions | Operationally unsafe output | Evidence constraints, fallback labels, structured outputs, and refusal when evidence is insufficient |
| Missing data/model artifacts | Poor reproducibility | Versioned manifest, checksums, bootstrap scripts, and container images |
| Sensitive imagery exposure | Security and policy risk | Local/offline deployment, access control, encryption, and retention policies |

---

## 22. Evaluation Strategy

A complete evaluation should measure each capability separately before reporting an overall score.

### 22.1 Single-image VQA

- Exact-match and relaxed answer accuracy
- Accuracy by question category
- Performance on count, presence, comparison, and land-cover questions

### 22.2 Captioning

- BLEU, METEOR, ROUGE-L, CIDEr, or semantic similarity
- Human assessment of factuality and coverage

### 22.3 Grounding and segmentation

- Bounding-box Intersection over Union (IoU)
- Recall at IoU thresholds
- Mask IoU, Dice/F1, and false-positive rate

### 22.4 Change-VQA

- Overall and per-question-type accuracy
- Direction-of-change accuracy
- Consistency when Time 1 and Time 2 are swapped

### 22.5 Change mapping

- Binary and per-class IoU
- Precision, recall, F1, and boundary quality
- Performance by change size and land-cover category

### 22.6 Optical-SAR fusion

- Cross-modal retrieval accuracy
- Paired-versus-unpaired embedding separation
- Improvement over optical-only and SAR-only baselines
- Downstream land-cover or change performance

### 22.7 Orchestration

- Task-routing accuracy
- Invalid-input rejection accuracy
- Tool-call success rate
- End-to-end latency
- Trace completeness
- Graceful degradation during service failure

### 22.8 Product quality

- Time from upload to usable answer
- Report completeness
- Usability with non-technical participants
- Accessibility and responsiveness

---

## 23. Recommended Development Priorities

The highest-value next steps are:

1. Connect real GeoTIFF pairing and reprojection to the LangGraph validation node.
2. Fix Specialist A's general-query response path and remove duplicate inference code.
3. Add a combined Change-VQA plus change-map route using Specialists B and C.
4. Preserve and validate genuine two-band SAR input through the complete fusion path.
5. Train Specialist C with true masks and report proper segmentation metrics.
6. Replace fixed confidence constants with calibrated scores.
7. Build a reproducible evaluation harness using exact public test splits.
8. Synchronise README and judge documentation with the active implementation.
9. Consolidate and lock all runtime dependencies.
10. Add containerised startup, secrets management, and deployment health monitoring.

---

## 24. Final Assessment

SatQuery AI has a strong and relevant central concept: use natural language to coordinate specialised remote-sensing models and produce evidence-backed, auditable results. Its principal innovation is the workflow that connects task interpretation, specialist selection, multimodal imagery, spatial evidence, traceability, and reporting.

The current codebase goes beyond a static concept. It contains a live web application, independent AI services, a real LangGraph workflow, trained local model artifacts, geospatial utilities, persistence, external imagery integration, and PDF reporting. This makes it a credible end-to-end prototype.

At the same time, the project is not yet a fully validated operational remote-sensing platform. Several advanced claims in older documentation are ahead of the active runtime: complete pair alignment is not wired into orchestration, fusion simplifies SAR input, Specialist C uses pseudo-labels, confidence is largely fixed, and benchmark evidence is limited. These are solvable engineering and evaluation gaps rather than reasons to discard the architecture.

The most accurate description of the project today is:

> **SatQuery AI is a working agentic remote-sensing research prototype that allows non-technical users to query single, paired, and multimodal satellite observations through one conversational interface, while exposing the specialist workflow, visual evidence, and execution metadata behind each result.**

With stronger geospatial integration, genuine SAR handling, calibrated uncertainty, reference-mask training, and reproducible benchmark evaluation, the existing architecture can mature into a viable decision-support platform for ISRO/SAC and other Earth-observation users.

---

## 25. Important Source Files

| Area | File |
|---|---|
| Problem and requirements | [`specs/PRD.md`](specs/PRD.md) |
| Architecture overview | [`README.md`](README.md) |
| Project constraints and model stack | [`AGENTS.md`](AGENTS.md) |
| Current task status | [`TASKLIST.md`](TASKLIST.md) |
| LangGraph controller | [`orchestrator/graph.py`](orchestrator/graph.py) |
| Orchestrator API | [`orchestrator/main.py`](orchestrator/main.py) |
| Language synthesis and intent | [`orchestrator/gemini_synthesizer.py`](orchestrator/gemini_synthesizer.py) |
| Specialist A service | [`services/vqa_service/main.py`](services/vqa_service/main.py) |
| Grounded SAM 2 integration | [`services/vqa_service/sam_segmenter.py`](services/vqa_service/sam_segmenter.py) |
| Specialist B service | [`services/changevqa_service/main.py`](services/changevqa_service/main.py) |
| Specialist C service | [`models/serve_c.py`](models/serve_c.py) |
| Specialist C model | [`finetune/specialist_c/model_c.py`](finetune/specialist_c/model_c.py) |
| Specialist C training | [`finetune/specialist_c/train_specialist_c.py`](finetune/specialist_c/train_specialist_c.py) |
| GeoTIFF validation | [`data/validator.py`](data/validator.py) |
| Spatial pairing and reprojection | [`data/pairing.py`](data/pairing.py) |
| GeoTIFF metadata parser | [`data/geotiff_parser.py`](data/geotiff_parser.py) |
| Coordinate imagery retrieval | [`data/satellite_fetcher.py`](data/satellite_fetcher.py) |
| SQLite persistence | [`data/database.py`](data/database.py) |
| Frontend application | [`frontend/src/App.jsx`](frontend/src/App.jsx) |
| Frontend dependencies | [`frontend/package.json`](frontend/package.json) |
| n8n workflow | [`n8n/satquery_workflow.json`](n8n/satquery_workflow.json) |
| Local n8n-compatible proxy | [`n8n/mock_n8n_server.py`](n8n/mock_n8n_server.py) |
| Unified launcher | [`start_satquery.py`](start_satquery.py) |
| PDF generator | [`reports/generate.py`](reports/generate.py) |

---

*This report describes the main `SATQUERY` folder as it exists in the audited local working tree. The adjacent `unncc` directory is supplementary and is not treated as part of the main application.*
