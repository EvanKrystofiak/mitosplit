r"""
register_finetuned_model.py - Register a fine-tuned empanada model so it appears in the
"Model" drop-down of every empanada-napari widget (2D Inference, 3D Inference, Finetune, ...).

What it does (same thing as the "Register a model" widget, but scriptable and idempotent):
  1. copies the .pth TorchScript file to  %USERPROFILE%\.empanada\<name>.pth   (unless --no-copy)
  2. writes                                %USERPROFILE%\.empanada\configs\<name>.yaml
     with the "model:" entry pointing at the copied file
  3. test-loads the model with torch.jit.load to make sure it is usable

Usage (inside the empanada-gpu environment):
  python register_finetuned_model.py --pth "..\finetuned model\mito_em_totoro.pth" --yaml "..\finetuned model\mito_em_totoro.yaml"
  python register_finetuned_model.py --pth model.pth --yaml model.yaml --name my_mito_model --no-copy
"""
import argparse
import os
import shutil
import sys

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pth', required=True, help='TorchScript .pth model file')
    p.add_argument('--yaml', required=True, help='empanada model config .yaml')
    p.add_argument('--name', default=None, help='model name shown in napari (default: yaml file name)')
    p.add_argument('--no-copy', action='store_true', help='reference the .pth where it is instead of copying it to ~/.empanada')
    p.add_argument('--force', action='store_true', help='overwrite an existing registration with the same name')
    args = p.parse_args()

    pth = os.path.abspath(args.pth)
    cfg_path = os.path.abspath(args.yaml)
    if not os.path.isfile(pth):
        sys.exit(f'model file not found: {pth}')
    if not os.path.isfile(cfg_path):
        sys.exit(f'config file not found: {cfg_path}')

    name = args.name or os.path.splitext(os.path.basename(cfg_path))[0]
    empanada_dir = os.path.join(os.path.expanduser('~'), '.empanada')
    config_dir = os.path.join(empanada_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    out_yaml = os.path.join(config_dir, f'{name}.yaml')
    if os.path.exists(out_yaml) and not args.force:
        print(f'{name} is already registered ({out_yaml}). Use --force to overwrite.')
    else:
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        if args.no_copy:
            model_path = pth
        else:
            model_path = os.path.join(empanada_dir, f'{name}.pth')
            if os.path.abspath(model_path) != pth:
                print(f'copying {pth} -> {model_path}')
                shutil.copy2(pth, model_path)

        config['model'] = model_path
        # the fine-tuned model has no quantized (CPU int8) twin
        if not config.get('model_quantized') or not os.path.isfile(str(config.get('model_quantized'))):
            config['model_quantized'] = None

        with open(out_yaml, 'w') as f:
            yaml.dump(config, f)
        print(f'wrote {out_yaml}')

    # test load
    import torch
    with open(out_yaml) as f:
        config = yaml.safe_load(f)
    model = torch.jit.load(config['model'], map_location='cpu')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model loads OK: {config["model"]}  ({n_params/1e6:.1f} M parameters)')
    print(f'classes: {config.get("class_names")}  norms: {config.get("norms")}')

    try:
        from empanada_napari.utils import get_configs
        print('models registered for empanada-napari:', ', '.join(sorted(get_configs().keys())))
    except Exception as e:  # pragma: no cover
        print(f'(could not list empanada-napari models: {e})')


if __name__ == '__main__':
    main()
