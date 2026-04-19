These scripts will produce docx files with tables of figures. Output files can be used with Google Docs (Open->Upload) for faster results processing.

Within the conda environment, `pip install python-docx`

At the top of each file are settings. The DIRNAME_SPEC can be copied from the hydra config for the run. ROOT needs to be set to the working directory (which is included in the DIRNAME_SPEC, just so the whole string from the YAML can be copied over). PRIORITY settings determine order and positions of parameters of sweeps; earlier in the list is more major (and more spread out in the table). FIG_SUBDIR, FIG_FILES are set based on preferences for the output. IMAGE_SPECS provides names for the figures and is needed to set which figures are used.

After running the script, the docx file will be saved in the working directory, alongside the folders of each run. Within Google Docs, open (upload) the file. Copy and paste the contents into the collaboration's Google Doc with other work; when in pageless mode, column widths constrain figure sizes.

Note that the file won't look great if loaded in other ways due to the image sizes. These sizes can be changed within the scripts (d_overfit: line 202, d_train: line 136).