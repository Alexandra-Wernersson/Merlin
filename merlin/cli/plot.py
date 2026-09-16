import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Plot corner/coverage/loss diagnostics for a merlin run."
    )
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("--mode", required=True, choices=["corner", "coverage", "loss"])
    parser.add_argument("--train-id", required=True, type=int,
                         help="Which run_dir/train_<N> subfolder to plot (from merlin-train's output)")
    parser.add_argument("--smooth", type=float, default=None,
                         help="Corner-mode only: swyft.plot_corner/get_pdf smoothing "
                              "(overrides PLOTTING.smooth_swyft; defaults to 1.0 if neither is set)")
    parser.add_argument("--bins", type=int, default=None,
                         help="Corner-mode only: swyft.plot_corner/get_pdf bin count "
                              "(overrides PLOTTING.nbins_swyft; defaults to 100 if neither is set)")
    parser.add_argument("--fiducial-override", type=str, default=None,
                         help="Corner-mode only: path to a YAML file of {param_name: value} "
                              "entries to overlay on this config's FIDUCIAL, evaluating the "
                              "checkpoint at a different fiducial cosmology instead of the "
                              "train_<N>'s own saved mock observation (overrides "
                              "PLOTTING.eval_fiducial if both are given). Saved as "
                              "corner_eval_fiducial.pdf, not corner.pdf.")
    parser.add_argument("--cols", type=int, default=None,
                         help="Coverage-mode only: panel-grid column count "
                              "(default: up to 5 -- see coverage.run_coverage_test)")
    args = parser.parse_args()

    from ..config import load_config, populate_train_dir
    from ..plotting import plot_corner_mode, plot_coverage_mode, plot_loss_mode

    config = load_config(args.config)
    populate_train_dir(config, args.config, train_id=args.train_id)

    fiducial_override = None
    if args.fiducial_override:
        import yaml
        with open(args.fiducial_override) as f:
            fiducial_override = yaml.safe_load(f)

    plot_name = args.mode
    if args.mode == "corner" and (fiducial_override or config.get("PLOTTING", {}).get("eval_fiducial")):
        plot_name = "corner_eval_fiducial"
    output_path = os.path.join(config["STORES"]["plots_dir"], f"{plot_name}.pdf")

    if args.mode == "corner":
        plot_corner_mode(config, output_path, smooth=args.smooth, bins=args.bins,
                          fiducial_override=fiducial_override)
    elif args.mode == "coverage":
        plot_coverage_mode(config, output_path, n_cols=args.cols)
    else:
        plot_loss_mode(config, output_path)

    print(f"Saved {output_path}")
