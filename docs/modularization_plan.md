# Nurvek-Vehicle Modularization Plan

**Date:** June 3, 2025
**Objective:** Refactor the existing `nurvek-v.py` script into a modular structure with a strict typesafe workflow, no `__init__.py` files (relying on namespace packages), and a preference for `pathlib` over the `os` module. The `subprocess` module will be integrated if/when a need to run external processes arises.

## Phase 1: Functional Modularization

This phase focuses on separating existing functions into logical modules.

**Current State (as of plan creation):**
*   `src/config/settings.py`: Created. Contains all global configuration constants. `nurvek-v.py` imports from it.
*   `src/database/db_utils.py`: Created. Contains `init_db_connection` and `flush_db_batch`. `nurvek-v.py` imports from it.
*   `src/utils/drawing.py`: Created. Contains `draw_text_with_background`. Integration into `nurvek-v.py` is pending/problematic.
*   `nurvek-v.py`: Still contains most of the application logic.

**Planned Steps for Phase 1:**

```mermaid
graph TD
    A[Start Phase 1] --> B{Current nurvek-v.py};
    
    subgraph "Step 1: Finalize Drawing Utilities"
        direction LR
        B1[nurvek-v.py] --> S1_A{Action: Read nurvek-v.py};
        S1_A --> S1_B["Action: In nurvek-v.py (via write_to_file if apply_diff is problematic):<br/> - Remove local draw_text_with_background() definition.<br/> - Add 'from src.utils.drawing import draw_text_with_background'."];
    end

    subgraph "Step 2: OCR & Preprocessing Utilities"
        direction LR
        B2[nurvek-v.py] --> S2_A[Create src/core/ocr_utils.py];
        S2_A --> S2_B[Move preprocess_lp_for_ocr to ocr_utils.py];
        S2_B --> S2_C[Move extract_license_plate_info_ocr to ocr_utils.py];
        S2_C --> S2_D[Add imports & type hints to ocr_utils.py];
        S2_D --> S2_E[Update nurvek-v.py: remove functions, add import from src.core.ocr_utils];
    end

    subgraph "Step 3: Core Pipeline Module"
        direction LR
        B3[nurvek-v.py] --> S3_A[Create src/core/pipeline.py];
        S3_A --> S3_B[Move ocr_worker_function to pipeline.py];
        S3_B --> S3_C[Move run_pipeline_final to pipeline.py (rename to run_main_pipeline)];
        S3_C --> S3_D["Handle queues (ocr_task_queue, display_results_queue)<br/>and worker_stats within pipeline.py (e.g., module-level)"];
        S3_D --> S3_E["Handle ENABLE_LP_PREPROCESSING toggle:<br/>run_main_pipeline modifies a module-level global ENABLE_LP_PREPROCESSING in pipeline.py.<br/>ocr_utils.extract_license_plate_info_ocr takes it as a parameter."];
        S3_E --> S3_F[Add imports & type hints to pipeline.py];
        S3_F --> S3_G[Update nurvek-v.py to import main func from src.core.pipeline];
    end

    subgraph "Step 4: Main Entry Point"
        direction LR
        B4[nurvek-v.py] --> S4_A[Create main.py (by writing content of refactored nurvek-v.py)];
        S4_A --> S4_B[Ensure main.py's 'if __name__ == \"__main__\":' block is clean];
        S4_B --> S4_C["main.py handles: <br/>- Necessary imports (settings, pipeline func, db_utils, pathlib, etc.)<br/>- Path setup (pathlib)<br/>- Model & EasyOCR initialization<br/>- DB connection setup<br/>- Calling main pipeline function<br/>- Global try/except/finally for cleanup"];
    end
    
    S4_C --> S5[Step 5: Consolidate & Refine Imports/Type Hints Across All Modules];
    S5 --> S6[Step 6: Thoroughly Test Functionality of main.py];
    S6 --> S7[Step 7: Manually Delete original nurvek-v.py];

    A --> S1_A;
    S1_B --> S2_A;
    S2_E --> S3_A;
    S3_G --> S4_A;
```

**Detailed Actions for Phase 1 (Recap - some already done):**

1.  **Finalize Drawing Utilities Integration (`src/utils/drawing.py` & `nurvek-v.py`):**
    *   **COMPLETED (pending successful test of `main.py`):** `nurvek-v.py` modified via `write_to_file` to remove local `draw_text_with_background` and import from `src.utils.drawing`.
2.  **Create OCR & Preprocessing Utilities (`src/core/ocr_utils.py`):**
    *   **COMPLETED (pending successful test of `main.py`):** `src/core/ocr_utils.py` created. `preprocess_lp_for_ocr` and `extract_license_plate_info_ocr` moved. `nurvek-v.py` updated. `extract_license_plate_info_ocr` now takes `enable_preprocessing` as a parameter.
3.  **Create Core Pipeline Module (`src/core/pipeline.py`):**
    *   **COMPLETED (pending successful test of `main.py`):** `src/core/pipeline.py` created. `ocr_worker_function` and `run_pipeline_final` (renamed `run_main_pipeline`) moved. Queues and `ENABLE_LP_PREPROCESSING` toggle state managed as module-level globals in `pipeline.py`. `ocr_worker_function` passes the `pipeline.ENABLE_LP_PREPROCESSING` to `extract_license_plate_info_ocr`.
4.  **Create Main Entry Point (`main.py`):**
    *   **COMPLETED (pending successful test of `main.py`):** `nurvek-v.py`'s content (after previous steps) written to `main.py`. `main.py` now imports `run_main_pipeline` from `src.core.pipeline` and calls it.
5.  **Consolidate & Refine:**
    *   Ongoing with each step.
6.  **Test Thoroughly:**
    *   **NEXT ACTION:** Run `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`.
7.  **Cleanup:**
    *   Manually delete `nurvek-v.py` after successful testing.

## Phase 2: Class-Based Refactoring

**Objective:** Convert the functionally modularized codebase into a more object-oriented structure.

**General Approach:** Iteratively refactor modules into classes, starting with the core pipeline.

**Step 1: Refactor `src/core/pipeline.py` into a `NurvekPipeline` Class**
*   **Class Definition:** Create `NurvekPipeline` in `src/core/pipeline.py`.
*   **`__init__` Method:**
    *   Accepts `video_source_path`, `vehicle_model`, `lp_model_instance`, `ocr_reader_instance`, `db_conn`, `db_cursor`, `worker_stats_ref`.
    *   Initializes instance attributes for these, plus:
        *   `self.ocr_task_queue = queue.Queue(maxsize=settings.MAX_OCR_TASK_QUEUE_SIZE)`
        *   `self.display_results_queue = queue.Queue(maxsize=settings.MAX_DISPLAY_RESULT_QUEUE_SIZE)`
        *   `self.enable_lp_preprocessing: bool = settings.ENABLE_LP_PREPROCESSING`
        *   `self.display_ocr_on_gui: bool = True`
        *   `self.active_filters: Dict[str, bool] = {}` (to be populated)
        *   `self.filter_keys_map: Dict[int, str] = {}` (to be populated)
        *   `self.tracked_vehicle_data: Dict[int, Dict[str, Any]] = {}`
        *   `self.gui_works: bool = True`
        *   Other necessary state variables (e.g., `window_name`).
*   **Methods:**
    *   `_ocr_worker_function(self)`: Adapts existing worker logic. Uses `self.ocr_task_queue`, `self.display_results_queue`, `self.enable_lp_preprocessing`.
    *   `run(self)`: Main execution method, containing the primary video processing loop. Starts the OCR worker thread.
    *   Helper methods (e.g., `_process_frame`, `_handle_gui_input`, `_update_ocr_results`, `_draw_overlays`, `_initialize_filters_and_controls`, `_shutdown`).
*   **`main.py` Update:**
    *   Import `NurvekPipeline`.
    *   Instantiate `NurvekPipeline` with all dependencies.
    *   Call `pipeline_instance.run()`.

**Step 2: Introduce `TrackedObject` (or `VehicleData`) Class (e.g., in `src/core/tracked_object.py`)**
*   **Attributes:** `track_id`, `lp_text`, `lp_confidence`, `lp_bbox_local`, `lp_confirmed`, `lp_ocr_attempts`, `lp_last_ocr_frame`, `lp_display_text`.
*   **Methods:** `add_ocr_attempt()`, `try_confirm_lp()`, `get_display_info()`.
*   **Integration:** `NurvekPipeline.tracked_vehicle_data` becomes `Dict[int, TrackedObject]`.

**Step 3: Introduce `ModelManager` Class (e.g., in `src/utils/model_loader.py`) - Optional**
*   Encapsulates YOLO model loading and device placement.
*   `main.py` uses it, passes loaded models to `NurvekPipeline`.

**Step 4: Introduce `DataLogger` Class (e.g., in `src/database/db_utils.py` or new file) - Optional**
*   Abstracts database logging for detection records.
*   `NurvekPipeline` (OCR worker part) uses an instance of `DataLogger`.

**Step 5: Integrate Python's `logging` Module**
*   Replace `print()` statements used for logging/status updates with the `logging` module throughout the codebase.
*   Configure basic logging in `main.py` (e.g., level, format, output to console/file).

```mermaid
graph TD
    M[main.py] --> INIT[Instantiates NurvekPipeline];
    
    subgraph NurvekPipeline Class in src.core.pipeline
        direction LR
        CONST[__init__(models, db, etc.)] --> ATTR[Instance Attributes:<br/>- queues<br/>- enable_lp_preprocessing<br/>- tracked_vehicle_data<br/>- etc.];
        ATTR --> RUN[run() method];
        RUN --> LOOP[Main Processing Loop];
        LOOP --> PF[_process_frame() method];
        PF --> VDT[Vehicle Detection];
        PF --> OCRQ[OCR Task Queuing Logic];
        PF --> GUI_U[_update_ocr_results() method];
        GUI_U --> TD_Update[Updates self.tracked_vehicle_data (now potentially Dict[int, TrackedObject])];
        PF --> DRAW[_draw_overlays() method];
        LOOP --> K_IN[_handle_gui_input() method];
        K_IN --> TOGGLE_STATE[Modifies self.enable_lp_preprocessing, etc.];
        
        RUN --> START_WORKER[Starts _ocr_worker_function thread];
        WORKER[_ocr_worker_function() method] -.- RUN;
        WORKER --> OCR_PROC[Uses self.ocr_task_queue, self.display_results_queue];
        WORKER --> OCR_UTIL[Calls ocr_utils.extract_license_plate_info_ocr with self.enable_lp_preprocessing];
        
        RUN --> SHUT[_shutdown() method];
    end

    INIT --> RUN;
```

This updated plan provides a clear path for both completing the initial functional separation and then moving into a more robust, object-oriented design.