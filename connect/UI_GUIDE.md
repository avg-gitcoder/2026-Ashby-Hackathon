# Ashby pathology workspace

This UI wraps the same `ReportSession` used by the latest
`Run_Pathology_Report.ipynb`, rather than executing or modifying notebook cells.
Image and existing-caption defaults are read safely from literal assignments in
that notebook. Model/Python settings default to its `Settings` class and can be
changed in the sidebar. Preserve the same working paths used on Delta.

## Run on Delta

In the repository's `connect` directory, activate your working notebook environment:

```bash
python -m pip install -r requirements-ui.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Use VS Code Remote SSH's **Ports** panel to forward port **8501**, then open the
forwarded address in your local browser. Keep the forwarded port private. If the
UI is running in an allocated CPU compute node, tunnel to that node using Delta's
documented compute-node forwarding procedure instead of forwarding the login
node's port. The process must be able to call `sbatch`, `sacct`, `squeue`, `scancel`
and read the existing project paths. GPU models still execute only inside the
existing Slurm GPU jobs; this UI never loads them itself.

Do not disable Streamlit's XSRF/CORS protection. The server and SSH session must
remain running; this is a private Delta UI, not a public hosted website.

## Use

1. Select a Delta image filename, or upload TIFF/PNG/JPEG.
2. Keep the notebook's existing caption only for its matching image; otherwise
   uncheck **Use existing PathGen caption**. The runner verifies image content and paths.
3. Click **Run / resume analysis**. Existing valid stages are reused; missing stages
   are submitted to Slurm. Queue time is included in the wait.
4. Review the original PathGen text, MedGemma explanation, five MUSK classifications
   and selected-class probabilities. Download TXT or JSON.
5. After a failure, fix the dependency/model path and click again, or select an
   individual stage. Preceding successful stages are restored from disk.

All interface labels are in English; model explanation text is shown unchanged.
Uploaded files remain under `reports/ui-inputs/`. Results share the notebook's
`reports/staged/` cache. Uploaded copies have different paths from original dataset
images and therefore do not share their cached captions. Use Delta-file mode to
reuse the notebook's results. Do not run the notebook and UI simultaneously or
start multiple browser sessions: the existing stage submission lock prevents
duplicate submissions but does not implement a multi-user queue.

Refreshing or closing the browser does not mean the GPU job is cancelled. Reopen
with the same inputs and click run to reconnect through the runner's saved job
state. If a process died with a `submit.lock`, follow `README.md` recovery guidance
and check the recorded job before clearing the lock. Streamlit's stop button is
not a GPU cancellation control; use `scancel JOB_ID` when cancellation is needed.

UI uploads and reports are excluded from Git. Existing notebook outputs and models
are untouched. Delta GPU execution must be checked in your working environment.
