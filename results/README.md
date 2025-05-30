# DiTOX / Results

This folder contains all the raw results from our experiments conducted with DiTOX.  
In total, we ran 130 models spanning opsets 7 to 12, covering categories such as classification, object detection, question answering, sentiment analysis, summarization, and generic text generation.

The results are organized into subfolders within the repository. Files with the keyword **"each"** in their names indicate per-pass individual runs. Additionally, the results for each rank are chunked to facilitate better management and execution of the experiments.

- `classification`: Results for classification models.  
- `detection_all`: Brief information for all object detection models.  
- `detection_full`: Detailed information for all object detection models.  
- `detection_diff`: Brief information for object detection models where differences were found.  
- `detection_all_info`: Extended information for object detection models where differences were found.  

- `text`: Results for text models. For GPT-2 specifically, runs are performed selecting the top-1, top-2, and top-3 tokens in separate runs, organized into their respective folders.
