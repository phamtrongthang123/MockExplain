# MockExplain user guide

MockExplain inspects a model's structure and suggests interpretability methods
and papers. It runs locally and does not need PyTorch, a GPU, or an API key.

## Features

- Inspect a PyTorch `.pt` state dictionary, a Python `.py` architecture, or both.
  An optional JSON config can add context.
- View tensor metadata, sampled weight statistics, source declarations, and
  structural clues when available.
- Get short method suggestions with links to papers or articles. Open **Details**
  for the reason, requirements, and limitations.
- Upload multiple models, switch between them, and compare their profiles.
- Save a JSON or Markdown report; reopen a saved JSON report later.
- Inspect files locally without executing the uploaded Python or running the model.

## Tutorial

### 1. Start the app

Install Python 3.10 or newer, then clone the repository and start MockExplain:

```bash
git clone https://github.com/phamtrongthang123/MockExplain.git
cd MockExplain
python3 -m mockexplain
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Keep the terminal open while
using the app; press **Ctrl+C** there to stop it.

### 2. Try a model

1. Click **Upload model** and select
   [`examples/tiny_cnn.pt`](../examples/tiny_cnn.pt) from the cloned repository.
2. Under **Model overview**, see the CNN candidate and its tensor summary. Expand
   **Files**, **Tensors**, or **Weight samples** for more information.
3. Under **Suggested methods**, find filter inspection and Grad-CAM. Follow the
   **Article** or **Paper** link, or open **Details** to see what each method needs.

The app suggests methods; it does not run them or measure model behavior.

### 3. Compare another model

1. Click **Upload model** again and select
   [`examples/transformer.py`](../examples/transformer.py).
2. Read its overview and suggestions, including J-lens and sparse autoencoders.
3. Select **Compare profiles** to see both models side by side. Use **Model
   profile** in the header to switch between them.

### 4. Save your work

Click **Save JSON report** to download a report you can reopen with **Open saved
report**. Use **Save Markdown** for a readable report. Profiles stay in the
current browser tab, so save a report before refreshing or closing it.

## Use your own model

Choose **Upload model** to select a `.pt` file, a `.py` file, or both, with an
optional JSON config. **Upload folder** collects these formats from a folder; it
must contain at most one of each. Combined uploads are limited to 128 MiB.

If you upload weights first, **Attach .py / config** adds source or configuration
in a new profile. The original profile remains available for comparison. The app
does not verify that attached files correspond to the checkpoint.

The `.pt` reader supports ordinary uncompressed PyTorch tensor state dictionaries,
not full serialized models, sharded checkpoints, or safetensors. For supported
formats and detailed limits, see the [README](../README.md#supported-inspection-and-safety-bounds).
If a structure is unfamiliar, the app may show **Unknown / not covered** without
suggesting a method. Model families are inferred from structural clues, and weight
statistics cover only the displayed samples.

If port 8000 is already in use, run `python3 -m mockexplain --port 8001` and open
`http://127.0.0.1:8001` instead. See the [demo video](../MockExplain_Demo.mp4) for
a short walkthrough.
