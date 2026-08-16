import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


DATASETS = {
    "fineweb": {
        "repo_id": "HuggingFaceFW/fineweb",
        "allow_patterns": "sample/10BT/*",
    },
    "fineweb-edu": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "allow_patterns": "sample/10BT/*",
    },
    "smoltalk": {
        "repo_id": "HuggingFaceTB/smoltalk",
        "allow_patterns": None,
    },
    "smol-smoltalk": {
        "repo_id": "HuggingFaceTB/smol-smoltalk",
        "allow_patterns": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download textual datasets for TinyLLM."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        default="fineweb",
        help=f"Dataset variant: {list(DATASETS.keys())} (default: fineweb)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("/media/datasets"),
        help="Base directory path (dataset name will be appended) (default: /media/datasets)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = DATASETS[args.dataset]
    target_dir = args.output_dir / args.dataset

    print(f"Downloading {config['repo_id']} -> {target_dir.resolve()}")
    
    kwargs = {
        "repo_id": config["repo_id"],
        "repo_type": "dataset",
        "local_dir": str(target_dir),
    }
    
    if config["allow_patterns"]:
        kwargs["allow_patterns"] = config["allow_patterns"]

    snapshot_download(**kwargs)


if __name__ == "__main__":
    main()