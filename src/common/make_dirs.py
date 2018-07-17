from pathlib import Path

def create_save_dirs(out):
    out_dir = Path(out)
    log_dir = out_dir / 'logs'
    plot_dir = out_dir / 'plots'
    other_dir = out_dir / 'others'
    result_dir = out_dir / 'models'
    snapshot_dir = result_dir / 'snapshot'
    snapshot_trainer_dir = snapshot_dir / 'trainer'
    snapshot_model_dir = snapshot_dir / 'models'
    snapshot_opt_dir = snapshot_dir / 'optimizers'
    final_result = result_dir / 'final_result'

    if not out_dir.exists():
        out_dir.mkdir()
    if not log_dir.exists():
        log_dir.mkdir()
    if not plot_dir.exists():
        plot_dir.mkdir()
    if not other_dir.exists():
        other_dir.mkdir()
    if not result_dir.exists():
        result_dir.mkdir()
    if not snapshot_dir.exists():
        snapshot_dir.mkdir()
    if not snapshot_trainer_dir.exists():
        snapshot_trainer_dir.mkdir()
    if not snapshot_model_dir.exists():
        snapshot_model_dir.mkdir()
    if not snapshot_opt_dir.exists():
        snapshot_opt_dir.mkdir()
    if not final_result.exists():
        final_result.mkdir()

    result = {'log_dir': log_dir,
              'plot_dir': plot_dir,
              'snapshot_dir': snapshot_dir,
              'snapshot_trainer_dir': snapshot_trainer_dir,
              'snapshot_model_dir': snapshot_model_dir,
              'snapshot_opt_dir': snapshot_opt_dir,
              'final_result': final_result}

    return result
