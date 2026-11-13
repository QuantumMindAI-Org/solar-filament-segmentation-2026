"""Solar filament segmentation, QMAI entry: model, inference, ensemble, post-processing, CSV writer.

The pipeline, in the order reproduce.py runs it:

    H-alpha JPEG -> preprocess.load_halpha -> inference.probability_map (tiled U-Net)
      -> inference.quantise (8-bit) -> ensemble.ensemble_maps (integer mean, half up)
      -> postprocess.instance_rows (hysteresis + minimum area) -> submission.write_csv
"""
