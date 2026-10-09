r"""
finetune_mito.py - fine-tune a registered empanada model headlessly (same code path as the
"Finetune a model" widget) on a dataset made by make_finetune_dataset.py, then register the result.

  python finetune_mito.py --train-dir finetune_data\ds4\train --eval-dir finetune_data\ds4\val ^
                          --base-model MitoNet_v1 --name mito_totoro_v2 --iterations 2000 --finetune-layer all

Outputs <model-dir>/<name>.pth + .yaml (default model-dir: %USERPROFILE%\.empanada\finetuned) and registers
<name> so it shows up in the napari model drop-down. Use --dry-run to only print the resolved config.
"""
import argparse
import os
import sys
from glob import glob

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-dir', required=True)
    p.add_argument('--eval-dir', default=None)
    p.add_argument('--base-model', default='MitoNet_v1', help='registered model to start from (default MitoNet_v1)')
    p.add_argument('--name', required=True, help='name of the new model (no spaces)')
    p.add_argument('--model-dir', default=os.path.join(os.path.expanduser('~'), '.empanada', 'finetuned'))
    p.add_argument('--finetune-layer', default='none', choices=['none', 'stage4', 'stage3', 'stage2', 'stage1', 'all'],
                   help='encoder layers to unfreeze (decoder/heads always train). default none')
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--patch-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4, help='default 4 (fits an 8 GB GPU at 512 px)')
    p.add_argument('--max-lr', type=float, default=None, help='OneCycleLR max_lr (default from config: 0.003)')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--no-amp', action='store_true', help='disable mixed precision')
    p.add_argument('--custom-config', default=None, help='alternative empanada training config yaml')
    p.add_argument('--force', action='store_true', help='overwrite an existing model registration with --name')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def build_config(args):
    import empanada_napari
    from empanada_napari.utils import get_configs, abspath
    from empanada.config_loaders import load_config

    main_config = args.custom_config or abspath(empanada_napari.__file__, 'training/finetune_config.yaml')
    config = load_config(main_config)
    model_configs = get_configs()
    if args.base_model not in model_configs:
        sys.exit(f'unknown base model {args.base_model}; available: {sorted(model_configs)}')
    model_config = load_config(model_configs[args.base_model])
    config['MODEL'] = {}
    for k, v in model_config.items():
        if k != 'FINETUNE':
            config['MODEL'][k] = v
        else:
            config[k] = v

    train_dir = os.path.abspath(args.train_dir)
    n_imgs = len(glob(os.path.join(train_dir, '**/images/*')))
    bsz = args.batch_size
    if n_imgs < bsz:
        sys.exit(f'need at least {bsz} training images, found {n_imgs} in {train_dir}')
    epochs = max(1, int(args.iterations // (n_imgs // bsz)))

    config['model_name'] = args.name
    config['TRAIN']['train_dir'] = train_dir
    config['TRAIN']['model_dir'] = os.path.abspath(args.model_dir)
    config['TRAIN']['finetune_layer'] = args.finetune_layer
    config['TRAIN']['batch_size'] = bsz
    config['TRAIN']['workers'] = args.workers
    config['TRAIN']['amp'] = not args.no_amp
    if args.max_lr is not None:
        config['TRAIN']['schedule_params']['max_lr'] = args.max_lr
    for aug in config['TRAIN']['augmentations']:
        for k in list(aug.keys()):
            if ('height' in k or 'width' in k) and aug.get(k) is None:
                aug[k] = args.patch_size
    config['TRAIN']['save_freq'] = max(1, epochs // 5)
    config['EVAL']['eval_dir'] = os.path.abspath(args.eval_dir) if args.eval_dir else None
    config['EVAL']['epochs_per_eval'] = max(1, epochs // 5)
    if 'epochs' in config['TRAIN']['schedule_params']:
        config['TRAIN']['schedule_params']['epochs'] = epochs
    else:
        config['TRAIN']['epochs'] = epochs
    for section in ('TRAIN', 'EVAL'):
        for metric in config[section]['metrics']:
            if metric['metric'] in ('IoU', 'PQ'):
                metric['labels'] = config['MODEL']['labels']
            elif metric['metric'] == 'F1':
                metric['labels'] = config['MODEL']['thing_list']
    print(f'{n_imgs} training patches, batch {bsz} -> {epochs} epochs (~{epochs * (n_imgs // bsz)} iterations)')
    return config


def main():
    args = parse_args()
    config = build_config(args)
    if args.dry_run:
        print(yaml.dump(config, sort_keys=False))
        return

    os.makedirs(config['TRAIN']['model_dir'], exist_ok=True)
    from empanada_napari import finetune
    from empanada_napari.utils import add_new_model

    finetune.main(config)

    out_yaml = os.path.join(config['TRAIN']['model_dir'], args.name + '.yaml')
    reg_yaml = os.path.join(os.path.expanduser('~'), '.empanada', 'configs', args.name + '.yaml')
    if os.path.exists(reg_yaml) and args.force:
        os.remove(reg_yaml)
    add_new_model(args.name, out_yaml)
    print(f'finished. model: {out_yaml.replace(".yaml", ".pth")}  registered as "{args.name}"')


if __name__ == '__main__':
    main()
